"""Render a seamless, transparent-background E6 trajectory loop with PyMOL.

Run with PyMOL's bundled Python (it provides `pymol` and numpy), e.g.

  /Applications/PyMOL.app/Contents/bin/python tools/md-hero/render.py \
      --top topology_35us.pdb --traj trajectory_35us.xtc \
      --cvs cvs_35us.dat --out /tmp/md-hero/final

Writes frame_0000.png ... (RGBA) and meta.json describing the chosen window.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from pymol import cmd, stored

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loop import (align_frames, gaussian_smooth, pick_window,  # noqa: E402
                  replica_ranges, rmsd, seam_blend)

# trajectory_35us.xtc is four replicas of apo HPV-16 E6, concatenated
REPLICA_LENGTHS = [10000, 10000, 5001, 10001]
CORE_RESI = (8, 140)   # ordered core (Cα), as in the FEL pipeline
CORE_SEL = 'name CA and resi %d-%d' % CORE_RESI


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--top', required=True, help='topology PDB')
    p.add_argument('--traj', required=True, help='trajectory PyMOL can read')
    p.add_argument('--cvs', required=True,
                   help='per-frame CVs; third column = pocket aperture (nm)')
    p.add_argument('--out', required=True, help='folder for frame_####.png')
    p.add_argument('--replica', type=int, default=2, help='1-based replica')
    p.add_argument('--frames', type=int, default=360, help='loop length, output frames')
    # 16 x 420 frames spans ~2/3 of replica 2: ~3x the motion of stride 2 at a
    # similar per-frame speed (slow breathing instead of faster jitter)
    p.add_argument('--stride', type=int, default=16, help='source frames per output frame')
    p.add_argument('--sigma', type=float, default=5.0, help='smoothing, output frames')
    p.add_argument('--blend', type=int, default=60, help='seam fade, output frames')
    p.add_argument('--start', type=int, help='force the window start (skip the search)')
    p.add_argument('--turn', type=float, nargs=2, default=[0.0, 0.0], metavar=('X', 'Y'),
                   help='extra camera rotation in degrees after orienting on the core')
    p.add_argument('--size', type=int, default=1080, help='square image size, px')
    p.add_argument('--every', type=int, default=1, help='render every k-th frame')
    p.add_argument('--draft', action='store_true', help='fast preview quality')
    p.add_argument('--part', type=int, default=0, help='this worker, 0-based')
    p.add_argument('--parts', type=int, default=1, help='number of parallel workers')
    return p.parse_args()


def load_replica(args):
    """Strided coordinates and pocket aperture for one replica, plus core indices."""
    start, stop = replica_ranges(REPLICA_LENGTHS)[args.replica - 1]
    cmd.load(args.top, 'traj')
    cmd.load_traj(args.traj, 'traj', start=start + 1, stop=stop, interval=args.stride)
    n = cmd.count_states('traj')
    coords = np.array([cmd.get_coords('traj', state=s) for s in range(1, n + 1)])
    aperture = np.loadtxt(args.cvs)[start:stop:args.stride, 2][:n]
    stored.atoms = []
    cmd.iterate('traj', 'stored.atoms.append((resv, name))')
    core = np.array([i for i, (resv, name) in enumerate(stored.atoms)
                     if name == 'CA' and CORE_RESI[0] <= resv <= CORE_RESI[1]])
    return coords, aperture, core


def build_loop(args, coords, aperture, core):
    """Aligned, smoothed, seam-blended frames plus a summary of the window."""
    coords = align_frames(coords, coords[0], core)
    if args.start is None:
        start, seam, swing = pick_window(aperture, coords, args.frames, args.blend)
    else:
        start = args.start
        seam = rmsd(coords[start], coords[start + args.frames])
        swing = float(aperture[start:start + args.frames].max()
                      - max(aperture[start], aperture[start + args.frames]))
    margin = int(np.ceil(3 * args.sigma))
    lo = max(0, start - margin)
    hi = min(len(coords), start + args.frames + args.blend + margin)
    smooth = gaussian_smooth(coords[lo:hi], args.sigma)
    seq = smooth[start - lo:start - lo + args.frames + args.blend]
    first = replica_ranges(REPLICA_LENGTHS)[args.replica - 1][0] + start * args.stride
    info = {'window_start': start, 'seam_rmsd_A': round(seam, 3),
            'pocket_swing_nm': round(swing, 3),
            'global_frames': [first, first + (args.frames + args.blend - 1) * args.stride]}
    return seam_blend(seq, args.frames, args.blend), info


def apply_style(draft):
    """Matte white surface, soft shadows, transparent background."""
    settings = {
        'ray_opaque_background': 0, 'ray_trace_mode': 0, 'depth_cue': 0,
        'ray_trace_fog': 0, 'orthoscopic': 0, 'field_of_view': 20,
        'surface_quality': 0 if draft else 1, 'antialias': 1 if draft else 2,
        'ambient': 0.30, 'direct': 0.55, 'reflect': 0.45,
        'specular': 0.12, 'shininess': 12, 'spec_count': 1,
        'light_count': 4 if draft else 8,
        'ray_shadows': 1, 'ray_shadow_decay_factor': 0.1, 'ray_shadow_decay_range': 2.0,
        'ambient_occlusion_mode': 1, 'ambient_occlusion_scale': 15,
        'ambient_occlusion_smooth': 10,
    }
    for name, value in settings.items():
        cmd.set(name, value)


def setup_scene(loop, args):
    """Frame the camera on every loop frame, then keep one state to render."""
    cmd.set('auto_zoom', 0)
    cmd.create('frames', 'traj', 1, 1)
    for i, xyz in enumerate(loop, start=1):
        cmd.load_coordset(xyz, 'frames', state=i)
    cmd.orient('frames and ' + CORE_SEL, state=1)
    cmd.turn('x', args.turn[0])
    cmd.turn('y', args.turn[1])
    cmd.zoom('frames', 2.0, state=0, complete=1)   # state=0 fits all states
    view = cmd.get_view()
    cmd.create('hero', 'frames', 1, 1)
    cmd.delete('frames')
    cmd.delete('traj')
    cmd.hide('everything')
    cmd.show('surface', 'hero')
    cmd.set_color('clay', [0.93, 0.93, 0.94])
    cmd.color('clay', 'hero')
    apply_style(args.draft)
    cmd.set_view(view)


def render(loop, args):
    os.makedirs(args.out, exist_ok=True)
    picks = list(range(0, len(loop), args.every))[args.part::args.parts]
    cmd.set('max_threads', max(1, (os.cpu_count() or 1) // args.parts))
    t0 = time.time()
    for k, i in enumerate(picks, start=1):
        cmd.load_coordset(loop[i], 'hero', state=1)
        path = os.path.join(args.out, 'frame_%04d.png' % (i // args.every))
        cmd.png(path, width=args.size, height=args.size, ray=1, quiet=1)
        if k % 10 == 0 or k == len(picks):
            print('part %d: %d/%d frames, %.1f s/frame'
                  % (args.part, k, len(picks), (time.time() - t0) / k), flush=True)


def main():
    args = parse_args()
    cmd.feedback('disable', 'all', 'everything')
    coords, aperture, core = load_replica(args)
    loop, info = build_loop(args, coords, aperture, core)
    setup_scene(loop, args)
    if args.part == 0:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, 'meta.json'), 'w') as f:
            json.dump(dict(info, args=vars(args)), f, indent=2)
        print(json.dumps(info), flush=True)
    render(loop, args)


if __name__ == '__main__':
    main()
