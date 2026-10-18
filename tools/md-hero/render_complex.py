"""Render a seamless, transparent cartoon loop of a protein-ligand complex from MD,
in the lab's IDP_style (the E6 hero's loop: a window whose ends nearly match,
smoothed, with a blended seam).

Run with PyMOL's bundled Python (it provides pymol, numpy and mdtraj), e.g.

  /Applications/PyMOL.app/Contents/bin/python tools/md-hero/render_complex.py \
      --top 9BJK_NAM_merged_protein_ligands.pdb \
      --traj 9BJK_NAM_protein_ligands_every10.xtc --segment 10001 15001 \
      --stride 8 --sigma 5 --style ~/.pymol/startup/IDP_style.py \
      --ligand 501 lime --ligand 502 hotpink --ligand-radius 0.45 \
      --azimuth 15 --tilt 15 --out /tmp/md-hero/mor

--segment must be one continuous simulation. The window is the one (among the 5%
whose first and last frames match best) where the ligands stray furthest from their
usual pose and come back. The camera looks along a membrane whose normal is z, with
the -z side (extracellular, for a GPCR whose N-terminus points to -z) on top, and
frames every loop frame with a little room to spare.
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
from loop import (align_frames, gaussian_smooth, pick_window,  # noqa: E402
                  rmsd, seam_blend)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--top', required=True, help='complex PDB, atoms in trajectory order')
    p.add_argument('--traj', required=True, help='trajectory MDTraj can read')
    p.add_argument('--segment', type=int, nargs=2, required=True, metavar=('START', 'STOP'),
                   help='frames [START, STOP) of --traj that form one continuous run')
    p.add_argument('--frames', type=int, default=360, help='loop length, output frames')
    p.add_argument('--stride', type=int, default=1, help='trajectory frames per output frame')
    p.add_argument('--sigma', type=float, default=3.0, help='smoothing, output frames')
    p.add_argument('--blend', type=int, default=60, help='seam fade, output frames')
    p.add_argument('--start', type=int, help='force the window start (skip the search)')
    p.add_argument('--style', required=True, help='script defining IDP_style()')
    p.add_argument('--protein-color', default='white')
    p.add_argument('--ligand', nargs=2, action='append', default=[], metavar=('RESI', 'COLOR'),
                   help='ligand residue number and its color; repeat per ligand')
    p.add_argument('--ligand-radius', type=float,
                   help="ligand stick radius, Å (default: the style's)")
    p.add_argument('--azimuth', type=float, default=0.0,
                   help='viewing direction around the membrane normal, degrees')
    p.add_argument('--tilt', type=float, default=0.0,
                   help='camera height above the membrane plane, degrees')
    p.add_argument('--size', type=int, default=1000, help='square image size, px')
    p.add_argument('--every', type=int, default=1, help='render every k-th frame')
    p.add_argument('--draft', action='store_true', help='fast preview quality')
    p.add_argument('--part', type=int, default=0, help='this worker, 0-based')
    p.add_argument('--parts', type=int, default=1, help='number of parallel workers')
    p.add_argument('--out', required=True, help='folder for frame_####.png')
    return p.parse_args()


def side_view(azimuth, tilt):
    """Rotation putting -z (extracellular) up on screen: looking along the membrane
    from `azimuth` around the normal, raised `tilt` degrees above the membrane plane."""
    a, b = np.radians(azimuth), np.radians(tilt)
    ex = np.array([np.cos(a), np.sin(a), 0.0])
    ey = np.array([0.0, 0.0, -1.0])
    ez = np.cross(ex, ey)
    return np.stack([ex, np.cos(b) * ey - np.sin(b) * ez, np.cos(b) * ez + np.sin(b) * ey])


def load_segment(args):
    """Segment coordinates (Å), helix-core Cα, ligand heavy atoms, the atoms that
    outline the picture (backbone and ligands), and (resSeq, element) per atom."""
    start, stop = args.segment
    need = len(range(start, stop, args.stride))
    chunks, have = [], 0
    for chunk in md.iterload(args.traj, top=args.top, chunk=500, skip=start, stride=args.stride):
        chunks.append(chunk)
        have += len(chunk)
        if have >= need:
            break
    seg = md.join(chunks)[:need]
    top = seg.topology
    # amino acids only: MDTraj also counts an unnamed ligand (UNK) as protein
    peptide = [r for r in top.residues
               if r.is_protein and {'N', 'CA', 'C'} <= {a.name for a in r.atoms}]
    ss = md.compute_dssp(seg[0].atom_slice([a.index for r in peptide for a in r.atoms]),
                         simplified=True)[0]
    helix = {r.resSeq for r, c in zip(peptide, ss) if c == 'H'}
    core = np.array([a.index for a in top.atoms
                     if a.name == 'CA' and a.residue.is_protein and a.residue.resSeq in helix])
    lig = np.concatenate([top.select('resSeq %s and not element H' % r) for r, _ in args.ligand])
    shown = np.union1d(top.select('protein and backbone'), lig)
    atoms = [(a.residue.resSeq, a.element.symbol) for a in top.atoms]   # names may differ
    return seg.xyz * 10.0, core, lig, shown, atoms


def build_loop(args, coords, core, lig):
    """Aligned, smoothed, seam-blended frames plus a summary of the window."""
    coords = align_frames(coords, coords[0], core)
    mean_pose = coords[:, lig].mean(axis=0)
    excursion = np.sqrt(((coords[:, lig] - mean_pose) ** 2).sum(-1).mean(-1))
    if args.start is None:
        start, seam, swing = pick_window(excursion, coords, args.frames, args.blend)
    else:
        start = args.start
        seam = rmsd(coords[start], coords[start + args.frames])
        swing = float(excursion[start:start + args.frames].max()
                      - max(excursion[start], excursion[start + args.frames]))
    margin = int(np.ceil(3 * args.sigma))
    lo = max(0, start - margin)
    hi = min(len(coords), start + args.frames + args.blend + margin)
    seq = gaussian_smooth(coords[lo:hi], args.sigma)[start - lo:start - lo + args.frames + args.blend]
    first = args.segment[0] + start * args.stride
    info = {'window_start': start, 'seam_rmsd_A': round(seam, 3),
            'ligand_excursion_A': round(swing, 3),
            'traj_frames': [first, first + (args.frames + args.blend - 1) * args.stride]}
    return seam_blend(seq, args.frames, args.blend), info


def fit_view(xyz, margin=3.0):
    """Square camera, centered on xyz (frames, atoms, 3; already in screen axes), whose
    picture holds every frame's atoms with `margin` Å to spare."""
    lo, hi = xyz.min(axis=(0, 1)), xyz.max(axis=(0, 1))
    half = max(hi[0] - lo[0], hi[1] - lo[1]) / 2 + margin
    depth = (hi[2] - lo[2]) / 2 + margin
    dist = half / np.tan(np.radians(cmd.get_setting_float('field_of_view')) / 2)
    view = list(cmd.get_view())
    view[:12] = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, -dist]
    view[12:15] = [float(v) for v in (lo + hi) / 2]
    view[15:17] = [dist - depth, dist + depth]
    cmd.set_view(view)


def setup_scene(args, loop, atoms, shown):
    """Load the complex, apply IDP_style and colors, frame every loop frame."""
    cmd.set('retain_order', 1)     # keep file atom order so coordinates line up
    cmd.set('auto_zoom', 0)
    cmd.load(args.top, 'cx')
    if [(int(a.resi), a.symbol) for a in cmd.get_model('cx').atom] != atoms:
        raise SystemExit('PyMOL atom order does not match the trajectory')
    cmd.load_coordset(loop[0], 'cx', state=1)
    runpy.run_path(os.path.expanduser(args.style))['IDP_style']()
    cmd.color(args.protein_color, 'cx and polymer.protein')
    for resi, color in args.ligand:
        lig = 'cx and resi %s' % resi
        cmd.show('sticks', lig)
        cmd.color(color, lig)
        if args.ligand_radius:
            cmd.set_bond('stick_radius', args.ligand_radius, lig)
    cmd.hide('everything', 'hydro')
    cmd.dss('cx')                  # helices from the first frame; they persist
    if args.draft:
        cmd.set('antialias', 1)
    fit_view(loop[:, shown])


def render(args, loop):
    os.makedirs(args.out, exist_ok=True)
    picks = list(range(0, len(loop), args.every))[args.part::args.parts]
    cmd.set('max_threads', max(1, (os.cpu_count() or 1) // args.parts))
    t0 = time.time()
    for k, i in enumerate(picks, start=1):
        cmd.load_coordset(loop[i], 'cx', state=1)
        cmd.png(os.path.join(args.out, 'frame_%04d.png' % (i // args.every)),
                width=args.size, height=args.size, ray=1, quiet=1)
        if k % 10 == 0 or k == len(picks):
            print('part %d: %d/%d frames, %.1f s/frame'
                  % (args.part, k, len(picks), (time.time() - t0) / k), flush=True)


def main():
    args = parse_args()
    cmd.feedback('disable', 'all', 'everything')
    coords, core, lig, shown, atoms = load_segment(args)
    loop, info = build_loop(args, coords, core, lig)
    loop = (loop @ side_view(args.azimuth, args.tilt).T).astype(np.float32)
    setup_scene(args, loop, atoms, shown)
    if args.part == 0:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, 'meta.json'), 'w') as f:
            json.dump(dict(info, args=vars(args)), f, indent=2)
        print(json.dumps(info), flush=True)
    render(args, loop)


if __name__ == '__main__':
    main()
