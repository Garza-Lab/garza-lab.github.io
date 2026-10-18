"""Pure-numpy helpers that turn MD frames into a seamless video loop.

Runs with numpy >= 1.20, so it works inside PyMOL's bundled Python and under a
regular Python for the tests. Coordinate arrays are (frames, atoms, 3), in Å.
"""
import numpy as np


def replica_ranges(lengths):
    """Global [start, stop) frame ranges of replicas concatenated in order."""
    ranges, start = [], 0
    for n in lengths:
        ranges.append((start, start + int(n)))
        start += int(n)
    return ranges


def superpose(mobile, ref):
    """Kabsch fit: rotation r and translation t with mobile @ r.T + t ~= ref."""
    mc, rc = mobile.mean(axis=0), ref.mean(axis=0)
    u, _, vt = np.linalg.svd((mobile - mc).T @ (ref - rc))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return r, rc - mc @ r.T


def align_frames(coords, ref, idx):
    """Superpose every frame onto ref using only the atoms in idx."""
    out = np.empty_like(coords)
    for i, frame in enumerate(coords):
        r, t = superpose(frame[idx], ref[idx])
        out[i] = frame @ r.T + t
    return out


def rmsd(a, b):
    """RMSD between two (atoms, 3) arrays, without fitting."""
    return float(np.sqrt(((a - b) ** 2).sum(axis=-1).mean()))


def gaussian_smooth(coords, sigma):
    """Gaussian-smooth along the frame axis; sigma in frames, ends reflected."""
    if sigma <= 0:
        return coords.copy()
    k = int(np.ceil(3 * sigma))
    if len(coords) <= k + 1:
        raise ValueError('need more than %d frames for sigma=%g' % (k + 1, sigma))
    w = np.exp(-0.5 * (np.arange(-k, k + 1) / sigma) ** 2)
    w /= w.sum()
    padded = np.concatenate([coords[k:0:-1], coords, coords[-2:-k - 2:-1]])
    out = np.zeros(coords.shape, dtype=float)
    for j, wj in enumerate(w):
        out += wj * padded[j:j + len(coords)]
    return out.astype(coords.dtype)


def pick_window(aperture, coords, length, blend):
    """Start of the best loop of `length` frames (plus `blend` seam frames).

    A start i is a candidate when frames i .. i+length+blend-1 exist. Among the
    5% of candidates whose seam RMSD (frame i vs frame i+length) is lowest, the
    largest pocket swing wins:
        swing = max(aperture[i:i+length]) - max(aperture[i], aperture[i+length])
    Returns (start, seam_rmsd, swing).
    """
    n = len(coords) - length - blend + 1
    if n <= 0:
        raise ValueError('need at least length + blend frames')
    seam = np.array([rmsd(coords[i], coords[i + length]) for i in range(n)])
    peak = np.array([aperture[i:i + length].max() for i in range(n)])
    swing = peak - np.maximum(aperture[:n], aperture[length:length + n])
    keep = np.flatnonzero(seam <= np.quantile(seam, 0.05))
    best = keep[np.argmax(swing[keep])]
    return int(best), float(seam[best]), float(swing[best])


def seam_blend(seq, length, blend):
    """Turn length + blend frames into `length` frames that loop seamlessly.

    The first `blend` frames fade from seq[length + t] (what naturally follows
    the last frame) into seq[t], so last -> first is one ordinary step.
    """
    if len(seq) < length + blend:
        raise ValueError('seq must hold length + blend frames')
    out = np.array(seq[:length], dtype=float)
    if blend > 1:
        a = 0.5 - 0.5 * np.cos(np.pi * np.arange(blend) / (blend - 1))
        a = a.reshape((-1,) + (1,) * (seq.ndim - 1))
        out[:blend] = a * seq[:blend] + (1 - a) * seq[length:length + blend]
    return out.astype(seq.dtype)


# --- Ping-pong loops, for stretches whose ends don't match (e.g. disordered chains) ---

def align_sequential(coords, idx):
    """Superpose each frame onto the previous (already superposed) one using idx.

    Removes overall tumbling but keeps internal motion; suits disordered chains
    that have no rigid core to align on.
    """
    out = np.empty_like(coords)
    out[0] = coords[0]
    for i in range(1, len(coords)):
        r, t = superpose(coords[i][idx], out[i - 1][idx])
        out[i] = coords[i] @ r.T + t
    return out


def eased_times(n_src, n_out, ease):
    """Source-frame positions (0 .. n_src-1) for n_out output frames.

    Constant pace, with cosine velocity ramps over the first and last `ease`
    frames so the motion comes to rest before a ping-pong turn.
    """
    v = np.ones(n_out - 1)
    if ease:
        ramp = 0.5 - 0.5 * np.cos(np.pi * (np.arange(ease) + 0.5) / ease)
        v[:ease] = ramp
        v[-ease:] = ramp[::-1]
    pos = np.concatenate([[0.0], np.cumsum(v)])
    return pos * (n_src - 1) / pos[-1]


def pingpong_order(n):
    """Play frames 0..n-1 and back down to 1, so no frame repeats at a turn."""
    return list(range(n)) + list(range(n - 2, 0, -1))


def interpolate(coords, times):
    """Linearly interpolate (frames, atoms, 3) at fractional frame positions."""
    times = np.asarray(times, dtype=float)
    lo = np.clip(np.floor(times).astype(int), 0, len(coords) - 1)
    hi = np.minimum(lo + 1, len(coords) - 1)
    w = (times - lo).reshape((-1,) + (1,) * (coords.ndim - 1))
    return ((1 - w) * coords[lo] + w * coords[hi]).astype(coords.dtype)


# --- Flipbooks: still renders that hold, then dissolve into the next ---

def flipbook_schedule(n, hold, fade):
    """(pose, next pose, weight) for every output frame.

    Each pose is held for `hold` frames, then dissolves into the next over
    `fade` frames; the last pose dissolves back into the first.
    """
    out = []
    for i in range(n):
        out += [(i, i, 0.0)] * hold
        out += [(i, (i + 1) % n, k / (fade + 1)) for k in range(1, fade + 1)]
    return out


def crossfade(a, b, w):
    """Mix two straight-alpha RGBA uint8 images as if each were composited.

    Premultiplies, mixes by w, then un-premultiplies, so a fading molecule's
    edges don't darken against a transparent background.
    """
    a = a.astype(float) / 255
    b = b.astype(float) / 255
    alpha = (1 - w) * a[..., 3:] + w * b[..., 3:]
    mixed = (1 - w) * a[..., :3] * a[..., 3:] + w * b[..., :3] * b[..., 3:]
    rgb = np.where(alpha > 0, mixed / np.maximum(alpha, 1e-12), 0.0)
    return np.round(np.concatenate([rgb, alpha], axis=-1) * 255).astype(np.uint8)


def radial_fade(img, inner, outer):
    """Fade an RGBA image's alpha to zero between `inner` and `outer`.

    Radii are fractions of the half-size from the image centre, so anything
    reaching past the window dissolves instead of being cut at the edge.
    """
    h, w = img.shape[:2]
    y, x = np.mgrid[0:h, 0:w]
    r = np.hypot((x + 0.5 - w / 2) / (w / 2), (y + 0.5 - h / 2) / (h / 2))
    t = np.clip((r - inner) / (outer - inner), 0.0, 1.0)
    out = img.copy()
    out[..., 3] = np.round(img[..., 3] * (1 - t * t * (3 - 2 * t))).astype(np.uint8)
    return out
