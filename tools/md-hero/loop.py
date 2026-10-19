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


def true_runs(mask):
    """[start, stop) of each run of True values in a 1-D boolean array."""
    edges = np.diff(np.concatenate([[0], np.asarray(mask, dtype=int), [0]]))
    return list(zip(np.nonzero(edges == 1)[0].tolist(), np.nonzero(edges == -1)[0].tolist()))


def restore_helices(frames, raw, src, helix, residues, fit, pad=2):
    """Give each displayed helix the shape it has in its own raw frame.

    Smoothing and interpolating a disordered chain crumples a helix that lasts a
    frame or two. For output frame i, each run of helical residues in helix[src[i]]
    takes its coordinates from raw[src[i]], superposed onto frames[i] using the
    `fit` atoms of the run; the `pad` residues on either side blend linearly from
    that raw shape back to the smoothed chain.

    frames: (n_out, atoms, 3) display coordinates; raw: (n_src, atoms, 3); src:
    source frame shown at each output frame; helix: (n_src, residues) bools;
    residues / fit: per residue, the atom indices to move / to superpose on.
    """
    out = frames.copy()
    for i, k in enumerate(src):
        weight = np.zeros(len(residues))
        target = {}
        for a, b in true_runs(helix[k]):
            idx = np.concatenate([fit[r] for r in range(a, b)])
            rot, t = superpose(raw[k][idx], frames[i][idx])
            for r in range(max(0, a - pad), min(len(residues), b + pad)):
                w = 1.0 - max(a - r, r - (b - 1), 0) / (pad + 1)
                if w > weight[r]:
                    weight[r] = w
                    target[r] = raw[k][residues[r]] @ rot.T + t
        for r, goal in target.items():
            atoms = residues[r]
            out[i][atoms] = weight[r] * goal + (1 - weight[r]) * frames[i][atoms]
    return out


# --- Camera: a looping zoom from the whole molecule into a site and back ---

def zoom_schedule(hold_out, move, hold_in):
    """Zoom amount per output frame (0 = wide, 1 = close): hold wide, ease in over
    `move` frames, hold close, ease back out; the last frame leads into the first."""
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(1, move + 1) / (move + 1))
    return np.concatenate([np.zeros(hold_out), ramp, np.ones(hold_in), ramp[::-1]])


def zoom_camera(wide, close, s):
    """Picture center and half-height at zoom amount s between two (center,
    half-height) views. The half-height changes geometrically (an even zoom rate)
    and the center moves so the close-up glides steadily into the middle. Also
    returns how far the move has gone (0..1), for anything that moves with it."""
    (c0, h0), (c1, h1) = wide, close
    h = h0 ** (1 - s) * h1 ** s
    w = (h0 - h) / (h0 - h1)
    return np.asarray(c0) + w * (np.asarray(c1) - np.asarray(c0)), h, w


def fibonacci_sphere(n):
    """n nearly evenly spread unit vectors."""
    k = np.arange(n) + 0.5
    z = 1 - 2 * k / n
    phi = np.pi * (1 + 5 ** 0.5) * k
    r = np.sqrt(1 - z ** 2)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def clear_view(points, target, radius=6.0, n=600):
    """Unit vector toward the viewer along which least is in the way of `target`:
    each point in front of it counts by how near it passes the line of sight
    (a Gaussian of width `radius`)."""
    rel = np.asarray(points, dtype=float) - target
    dirs = fibonacci_sphere(n)
    along = rel @ dirs.T
    perp2 = (rel ** 2).sum(axis=1)[:, None] - along ** 2
    blocked = np.where(along > 0, np.exp(-perp2 / (2 * radius ** 2)), 0.0)
    return dirs[np.argmin(blocked.sum(axis=0))]


def hidden_counts(points, targets, dirs, radius):
    """For each direction (rows of dirs, pointing at the viewer), how many of
    `targets` have one of `points` in front of them within `radius` of their line
    of sight."""
    points = np.asarray(points, dtype=float)
    counts = np.zeros(len(dirs), dtype=int)
    for t in np.asarray(targets, dtype=float):
        rel = points - t
        along = rel @ dirs.T
        perp2 = (rel ** 2).sum(axis=1)[:, None] - along ** 2
        counts += ((along > 0) & (perp2 < radius ** 2)).any(axis=0)
    return counts


def projected_spread(points, dirs):
    """For each direction, the radius of gyration of `points` seen along it
    (small when an extended or flat molecule is seen end-on or edge-on)."""
    rel = np.asarray(points, dtype=float) - np.mean(points, axis=0)
    return np.sqrt(((rel ** 2).sum(axis=1)[:, None] - (rel @ dirs.T) ** 2).mean(axis=0))


def view_basis(d, along, up):
    """Rows x, y, z of a camera rotation looking down -d (z = d points at the
    viewer), with the direction `along` horizontal and the offset `up` pointing
    up the picture."""
    d = np.asarray(d, dtype=float) / np.linalg.norm(d)
    x = np.asarray(along, dtype=float) - np.dot(along, d) * d
    x /= np.linalg.norm(x)
    y = np.cross(d, x)
    if np.dot(up, y) < 0:
        x, y = -x, -y
    return np.stack([x, y, d])


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
