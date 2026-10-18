"""Render an ensemble flipbook: one docked complex per cluster, centered on the
ligand, each held and then dissolved into the next in a seamless loop.

Run with PyMOL's bundled Python (it provides pymol, numpy, mdtraj and PIL), e.g.

  /Applications/PyMOL.app/Contents/bin/python tools/md-hero/render_flipbook.py \
      --top NUPR1_ZZW115_all_20000_frames.pdb \
      --traj NUPR1_ZZW115_all_20000_frames.xtc --frame-map frame_map.tsv \
      --style ~/.pymol/startup/IDP_style.py --out /tmp/md-hero/nupr1

Uses the highest-DiffDock-confidence frame of each cluster. The frames are
independent docked complexes, not a trajectory: every image is a real complex,
and the dissolves blend pictures, never coordinates. The look comes from the
IDP_style() function in --style; the protein is then colored --color.
"""
import argparse
import csv
import json
import os
import runpy
import sys

import mdtraj as md
import numpy as np
from PIL import Image
from pymol import cmd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loop import crossfade, flipbook_schedule, radial_fade, superpose  # noqa: E402

TO_PYMOL_SS = {'H': 'H', 'E': 'S', 'C': 'L'}   # MDTraj simplified DSSP -> PyMOL


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--top', required=True, help='complex PDB, atoms in trajectory order')
    p.add_argument('--traj', required=True, help='docked-complex frames')
    p.add_argument('--frame-map', required=True,
                   help='TSV with trajectory_frame_0based, cluster_1based, winner_raw_confidence')
    p.add_argument('--style', required=True, help='script defining IDP_style()')
    p.add_argument('--ligand', default='ZZW', help='ligand residue name')
    p.add_argument('--color', default='cyan', help='protein color, PyMOL name')
    p.add_argument('--size', type=int, default=800, help='square image size, px')
    p.add_argument('--radius', type=float, default=55.0,
                   help='window half-width around the ligand, Å; farther tails fade out')
    p.add_argument('--hold', type=int, default=24, help='frames each complex is held')
    p.add_argument('--fade', type=int, default=12, help='frames of each dissolve')
    p.add_argument('--out', required=True, help='folder for frame_####.png')
    return p.parse_args()


def pick_frames(path):
    """Highest-confidence frame of each cluster, in cluster order."""
    best = {}
    with open(path) as f:
        for r in csv.DictReader(f, delimiter='\t'):
            c, conf = int(r['cluster_1based']), float(r['winner_raw_confidence'])
            if c not in best or conf > best[c]['confidence']:
                best[c] = {'cluster': c, 'confidence': conf,
                           'frame': int(r['trajectory_frame_0based']),
                           'replica': r['replica'], 'stored_frame': r['stored_frame_1based']}
    return [best[c] for c in sorted(best)]


def load_poses(args, picks):
    """Coordinates (Å) aligned on the ligand, DSSP per pose, residue numbers, atoms."""
    poses = md.load(args.traj, top=args.top)[[p['frame'] for p in picks]]
    top = poses.topology
    lig = top.select('resname %s' % args.ligand)
    prot = top.select('protein')
    ss = md.compute_dssp(poses.atom_slice(prot), simplified=True)
    resseq = [r.resSeq for r in top.subset(prot).residues]
    xyz = poses.xyz * 10.0
    ref = xyz[0][lig].copy()
    for i in range(len(xyz)):
        r, t = superpose(xyz[i][lig], ref)
        xyz[i] = xyz[i] @ r.T + t
    atoms = [(a.name, a.residue.resSeq) for a in top.atoms]
    return xyz, ss, resseq, atoms


def setup_scene(args, xyz, atoms):
    """Load the complex, apply IDP_style, center a fixed window on the ligand."""
    cmd.set('retain_order', 1)     # keep file atom order so coordinates line up
    cmd.set('auto_dss', 0)         # secondary structure comes from MDTraj
    cmd.set('auto_zoom', 0)
    cmd.load(args.top, 'cx')
    model = cmd.get_model('cx')
    if [(a.name, int(a.resi)) for a in model.atom] != atoms:
        raise SystemExit('PyMOL atom order does not match the trajectory')
    cmd.load_coordset(xyz[0], 'cx', state=1)
    runpy.run_path(os.path.expanduser(args.style))['IDP_style']()
    # IDP_style's chunky disordered-chain tube (its full=1 settings) keeps the
    # mostly-loop chain readable at web size; the rest of the look is the default.
    cmd.set('cartoon_loop_radius', 0.9)
    cmd.set('cartoon_sampling', 14)
    cmd.color(args.color, 'cx and polymer.protein')
    cmd.dss('cx')   # PyMOL ignores ss set by alter until dss has run once
    # Every pose is aligned on the ligand, so one ligand-centered view fits all.
    lig = 'cx and resn %s' % args.ligand
    cmd.orient(lig)
    center = [float(v) for v in np.mean(cmd.get_coords(lig), axis=0)]
    cmd.pseudoatom('window_center', pos=center)
    cmd.zoom('window_center', args.radius)
    cmd.delete('window_center')


def render_stills(args, xyz, ss, resseq):
    stills = os.path.join(args.out, 'stills')
    os.makedirs(stills, exist_ok=True)
    paths = []
    for i, pose in enumerate(xyz):
        cmd.load_coordset(pose, 'cx', state=1)
        table = {r: TO_PYMOL_SS[c] for r, c in zip(resseq, ss[i])}
        cmd.alter('cx and polymer.protein', 'ss = table[resv]', space={'table': table})
        cmd.rebuild()
        path = os.path.join(stills, 'pose_%02d.png' % i)
        cmd.png(path, width=args.size, height=args.size, ray=1, quiet=1)
        img = np.asarray(Image.open(path).convert('RGBA'))
        Image.fromarray(radial_fade(img, 0.82, 0.97), 'RGBA').save(path)
        paths.append(path)
        print('pose %d/%d' % (i + 1, len(xyz)), flush=True)
    return paths


def assemble(paths, out, hold, fade):
    """Write frame_####.png: holds as hard links, dissolves as alpha-correct mixes."""
    images = [np.asarray(Image.open(p).convert('RGBA')) for p in paths]
    schedule = flipbook_schedule(len(paths), hold, fade)
    for k, (i, j, w) in enumerate(schedule):
        dst = os.path.join(out, 'frame_%04d.png' % k)
        if os.path.exists(dst):
            os.remove(dst)
        if w == 0:
            os.link(paths[i], dst)
        else:
            Image.fromarray(crossfade(images[i], images[j], w), 'RGBA').save(dst)
    print('loop: %d frames' % len(schedule))


def main():
    args = parse_args()
    cmd.feedback('disable', 'all', 'everything')
    picks = pick_frames(args.frame_map)
    xyz, ss, resseq, atoms = load_poses(args, picks)
    setup_scene(args, xyz, atoms)
    os.makedirs(args.out, exist_ok=True)
    for p, row in zip(picks, ss):
        p['dssp'] = ''.join(row)
    with open(os.path.join(args.out, 'meta.json'), 'w') as f:
        json.dump({'poses': picks, 'args': vars(args)}, f, indent=1)
    assemble(render_stills(args, xyz, ss, resseq), args.out, args.hold, args.fade)


if __name__ == '__main__':
    main()
