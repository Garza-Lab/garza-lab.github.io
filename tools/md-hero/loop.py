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
