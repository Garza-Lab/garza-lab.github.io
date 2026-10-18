"""Render a looping, transparent cartoon movie of a disordered chain that shows the
secondary structure of every frame (MDTraj DSSP, simplified H/E/C).

Run with PyMOL's bundled Python (it provides pymol, numpy and mdtraj), e.g.

  /Applications/PyMOL.app/Contents/bin/python tools/md-hero/render_cartoon.py \
      --top NUPR1_bAIes_all_120000_frames.pdb --traj NUPR1_bAIes_all_120000_frames.xtc \
      --first 70725 --count 60 --sigma 2 --color cyan \
      --style ~/.pymol/startup/IDP_style.py --out /tmp/md-hero/nupr1
  /Applications/PyMOL.app/Contents/bin/python tools/md-hero/render_cartoon.py \
      --out /tmp/md-hero/nupr1 --mirror

The stretch plays forward, then backward (ping-pong), easing at each turn, so the
loop is seamless without blending unrelated conformations. Secondary structure is
assigned on the raw frames and shown only where it occurs; coordinates are smoothed
for display, but each helix keeps the shape it has in its own raw frame. Run once
more with --mirror (after all workers finish) to add the return pass as hard links.
"""
import argparse
import json
import os
import runpy
import sys
import time

import mdtraj as md
import numpy as np
from pymol import cmd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loop import (align_sequential, eased_times, gaussian_smooth,  # noqa: E402
                  interpolate, pingpong_order, restore_helices)

TO_PYMOL_SS = {'H': 'H', 'E': 'S', 'C': 'L'}   # MDTraj simplified DSSP -> PyMOL


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--top', help='topology PDB, atoms in trajectory order')
    p.add_argument('--traj', help='trajectory MDTraj can read')
    p.add_argument('--first', type=int, help='0-based index of the first source frame')
    p.add_argument('--count', type=int, help='number of consecutive source frames')
    p.add_argument('--out', required=True, help='folder for frame_####.png')
    p.add_argument('--half', type=int, default=360, help='output frames per pass')
    p.add_argument('--ease', type=int, default=30, help='easing frames at each turn')
    p.add_argument('--sigma', type=float, default=1.0, help='display smoothing, source frames')
    p.add_argument('--color', default='hotpink', help='PyMOL color name')
    p.add_argument('--style', help="script defining IDP_style() (default: the homepage's matte look)")
    p.add_argument('--size', type=int, default=1000, help='square image size, px')
    p.add_argument('--draft', action='store_true', help='fast preview quality')
    p.add_argument('--only', type=int, nargs='+', help='render just these output frames')
    p.add_argument('--part', type=int, default=0, help='this worker, 0-based')
    p.add_argument('--parts', type=int, default=1, help='number of parallel workers')
    p.add_argument('--mirror', action='store_true',
                   help='add the return pass as hard links, then exit')
    return p.parse_args()


def load_window(args):
    """Coordinates (Å), simplified DSSP per frame, residue numbers, CA indices, the
    trajectory's first frame (to check atom order), and per residue its atoms and
    backbone atoms. Reads only the frames needed."""
    chunks, have = [], 0
    for chunk in md.iterload(args.traj, top=args.top, chunk=500, skip=args.first):
        chunks.append(chunk)
        have += len(chunk)
        if have >= args.count:
            break
    if have < args.count:
        raise SystemExit('trajectory ends before frame %d' % (args.first + args.count - 1))
    win = md.join(chunks)[:args.count]
    ss = md.compute_dssp(win, simplified=True)
    resseq = [r.resSeq for r in win.topology.residues]
    ca = win.topology.select('name CA')
    first = md.load_frame(args.traj, 0, top=args.top)
    residues = [np.array([a.index for a in r.atoms]) for r in win.topology.residues]
    backbone = [np.array([a.index for a in r.atoms if a.name in ('N', 'CA', 'C', 'O')])
                for r in win.topology.residues]
    return win.xyz * 10.0, ss, resseq, ca, first.xyz[0] * 10.0, residues, backbone


def build_pass(args, xyz, ca, ss, residues, backbone):
    """Display coordinates for one forward pass, and the source frame of each."""
    raw = align_sequential(xyz, ca)
    times = eased_times(args.count, args.half, args.ease)
    src = np.rint(times).astype(int)
    frames = interpolate(gaussian_smooth(raw, args.sigma), times)
    # Smoothing crumples a helix that lasts a frame or two: show it in its raw shape.
    return restore_helices(frames, raw, src, ss == 'H', residues, backbone), src


def apply_style(draft):
    """Matte cartoon, soft shadows, transparent background (as in render.py)."""
    settings = {
        'ray_opaque_background': 0, 'ray_trace_mode': 0, 'depth_cue': 0,
        'ray_trace_fog': 0, 'orthoscopic': 0, 'field_of_view': 20,
        'antialias': 1 if draft else 2,
        'ambient': 0.30, 'direct': 0.55, 'reflect': 0.45,
        'specular': 0.12, 'shininess': 12, 'spec_count': 1,
        'light_count': 4 if draft else 8,
        'ray_shadows': 1, 'ray_shadow_decay_factor': 0.1, 'ray_shadow_decay_range': 2.0,
        'cartoon_fancy_helices': 1, 'cartoon_loop_radius': 0.5,
        'cartoon_oval_length': 1.6, 'cartoon_oval_width': 0.35,
        'cartoon_rect_length': 1.6, 'cartoon_rect_width': 0.5,
    }
    for name, value in settings.items():
        cmd.set(name, value)


def apply_idp_style(path, draft):
    """The lab's IDP_style() look (outlined illustration cartoon)."""
    runpy.run_path(os.path.expanduser(path))['IDP_style']()
    # IDP_style's chunky disordered-chain tube (its full=1 settings) keeps the
    # mostly-loop chain readable at web size; the rest of the look is the default.
    cmd.set('cartoon_loop_radius', 0.9)
    cmd.set('cartoon_sampling', 14)
    if draft:
        cmd.set('antialias', 1)


def fit_view(frames, margin=3.0):
    """Keep the camera's rotation; center and zoom it so every frame fits the square
    picture with `margin` Å to spare (tighter than cmd.zoom's bounding sphere)."""
    view = list(cmd.get_view())
    rot = np.array(view[:9]).reshape(3, 3)          # model -> camera (column-major)
    cam = frames.reshape(-1, 3) @ rot
    lo, hi = cam.min(axis=0), cam.max(axis=0)
    half = max(hi[0] - lo[0], hi[1] - lo[1]) / 2 + margin
    depth = (hi[2] - lo[2]) / 2 + margin
    dist = half / np.tan(np.radians(cmd.get_setting_float('field_of_view')) / 2)
    view[9:12] = [0.0, 0.0, -dist]
    view[12:15] = [float(v) for v in rot @ ((lo + hi) / 2)]
    view[15:17] = [dist - depth, dist + depth]
    cmd.set_view(view)


def setup_scene(args, frames, first_frame):
    """Frame the camera on every output frame, then keep one state to render."""
    cmd.set('retain_order', 1)     # keep file atom order so coordinates line up
    cmd.set('auto_dss', 0)         # secondary structure comes from MDTraj
    cmd.set('auto_zoom', 0)
    cmd.load(args.top, 'mol')
    if not np.allclose(cmd.get_coords('mol'), first_frame, atol=0.01):
        raise SystemExit('PyMOL atom order does not match the trajectory')
    cmd.create('frames', 'mol', 1, 1)
    for i, xyz in enumerate(frames, start=1):
        cmd.load_coordset(xyz, 'frames', state=i)
    cmd.orient('frames and name CA', state=0)       # principal axes of every frame
    fit_view(frames)
    view = cmd.get_view()
    cmd.create('peptide', 'frames', 1, 1)
    cmd.delete('frames')
    cmd.delete('mol')
    cmd.hide('everything')
    cmd.show('cartoon', 'peptide')
    cmd.dss('peptide')   # PyMOL ignores ss set by alter until dss has run once
    if args.style:
        apply_idp_style(args.style, args.draft)
    else:
        apply_style(args.draft)
    cmd.color(args.color, 'peptide')
    cmd.set_view(view)


def render(args, frames, src, ss, resseq):
    os.makedirs(args.out, exist_ok=True)
    picks = args.only or list(range(len(frames)))[args.part::args.parts]
    cmd.set('max_threads', max(1, (os.cpu_count() or 1) // args.parts))
    t0 = time.time()
    for k, i in enumerate(picks, start=1):
        cmd.load_coordset(frames[i], 'peptide', state=1)
        table = {r: TO_PYMOL_SS[c] for r, c in zip(resseq, ss[src[i]])}
        cmd.alter('peptide', 'ss = table[resv]', space={'table': table})
        cmd.rebuild('peptide', 'cartoon')
        cmd.png(os.path.join(args.out, 'frame_%04d.png' % i),
                width=args.size, height=args.size, ray=1, quiet=1)
        if k % 10 == 0 or k == len(picks):
            print('part %d: %d/%d frames, %.1f s/frame'
                  % (args.part, k, len(picks), (time.time() - t0) / k), flush=True)


def mirror(out, half):
    """Hard-link the forward frames in reverse order to make the return pass."""
    for k, src in enumerate(pingpong_order(half)[half:], start=half):
        dst = os.path.join(out, 'frame_%04d.png' % k)
        if os.path.exists(dst):
            os.remove(dst)
        os.link(os.path.join(out, 'frame_%04d.png' % src), dst)
    print('loop: %d frames' % (2 * half - 2))


def main():
    args = parse_args()
    if args.mirror:
        mirror(args.out, args.half)
        return
    cmd.feedback('disable', 'all', 'everything')
    xyz, ss, resseq, ca, first_frame, residues, backbone = load_window(args)
    frames, src = build_pass(args, xyz, ca, ss, residues, backbone)
    setup_scene(args, frames, first_frame)
    if args.part == 0 and not args.only:
        os.makedirs(args.out, exist_ok=True)
        has = (ss == 'H') | (ss == 'E')
        info = {'source_frames': [args.first, args.first + args.count - 1],
                'frames_with_helix_or_strand': int(has.any(1).sum()),
                'dssp': [''.join(row) for row in ss], 'args': vars(args)}
        with open(os.path.join(args.out, 'meta.json'), 'w') as f:
            json.dump(info, f, indent=1)
        print('source frames %d-%d, %d with helix/strand'
              % (args.first, args.first + args.count - 1, info['frames_with_helix_or_strand']),
              flush=True)
    render(args, frames, src, ss, resseq)


if __name__ == '__main__':
    main()
