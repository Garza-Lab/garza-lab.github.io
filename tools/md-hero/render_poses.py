"""Render a looping, transparent movie of several ligand poses in one binding site:
the receptor stays still as a surface while the ligands take turns, each fading
into the next, in the lab's IDP_style.

Run with PyMOL's bundled Python (it provides pymol, numpy and PIL), e.g.

  /Applications/PyMOL.app/Contents/bin/python tools/md-hero/render_poses.py \
      --style ~/.pymol/startup/IDP_style.py --out /tmp/md-hero/e6-ligands \
      4XR8_Ligand{1,2,3,4,5,6,7,8,9,10}.maegz

Each input is one receptor-ligand complex (any format PyMOL reads). Every complex
is superposed on the first one's receptor (Cα), so all poses share one still
receptor; water is removed and ions are not shown. The camera looks straight into
the pocket (the most open direction onto the ligands' common center), keeps the
pocket in the middle and frames the whole receptor, so nothing is cut at the
edge. Each pose is rendered once; the loop holds each and dissolves it into the
next (alpha-correct, so only the ligands change).
"""
import argparse
import json
import os
import runpy
import sys

import numpy as np
from pymol import cmd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loop import clear_view, superpose, view_basis  # noqa: E402
from render_flipbook import assemble  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('complexes', nargs='+', help='receptor-ligand complexes, one pose each')
    p.add_argument('--ligand', default='organic', help='PyMOL selection of the ligand')
    p.add_argument('--style', required=True, help='script defining IDP_style()')
    p.add_argument('--protein-color', default='white')
    p.add_argument('--ligand-color', default='brightorange')
    p.add_argument('--ligand-radius', type=float, default=0.45, help='ligand stick radius, Å')
    p.add_argument('--margin', type=float, default=4.0, help='room around the receptor, Å')
    p.add_argument('--hold', type=int, default=60, help='frames each pose is held')
    p.add_argument('--fade', type=int, default=45, help='frames of each dissolve')
    p.add_argument('--size', type=int, default=900, help='square image size, px')
    p.add_argument('--draft', action='store_true', help='fast preview quality')
    p.add_argument('--out', required=True, help='folder for frame_####.png')
    return p.parse_args()


def load_poses(args):
    """One receptor object 'rec' and one object per ligand, all in the first
    complex's frame."""
    cmd.set('retain_order', 1)
    cmd.set('auto_zoom', 0)
    names = []
    for i, path in enumerate(args.complexes, start=1):
        cx = 'cx%d' % i
        cmd.load(path, cx)
        cmd.remove('%s and solvent' % cx)
        ca = cmd.get_coords('%s and polymer.protein and name CA' % cx)
        if i == 1:
            ref = ca
            cmd.create('rec', '%s and polymer.protein' % cx)
        elif ca is None or len(ca) != len(ref):
            raise SystemExit('%s: receptor differs from the first complex' % path)
        r, t = superpose(ca, ref)
        name = 'lig%d' % i
        cmd.create(name, '%s and (%s)' % (cx, args.ligand))
        if not cmd.count_atoms(name):
            raise SystemExit('%s: no atoms match --ligand %r' % (path, args.ligand))
        cmd.load_coordset(cmd.get_coords(name) @ r.T + t, name, state=1)
        cmd.delete(cx)
        names.append(name)
    return names


def style(args, ligands):
    runpy.run_path(os.path.expanduser(args.style))['IDP_style']()
    cmd.hide('everything')
    cmd.set('transparency', 0)          # setPretty's 0.5 would show the surface's far side
    cmd.set('surface_quality', 1)       # finer mesh: no stray outline specks on the surface
    cmd.show('surface', 'rec')
    cmd.color(args.protein_color, 'rec')
    for name in ligands:
        cmd.show('sticks', '%s and not hydro' % name)
        cmd.color(args.ligand_color, name)   # one color, no element coloring
        cmd.set_bond('stick_radius', args.ligand_radius, name)
    if args.draft:
        cmd.set('antialias', 1)


def frame(args, ligands):
    """Turn everything so the pocket faces the camera; return the view."""
    rec = cmd.get_coords('rec and not hydro')
    lig = np.concatenate([cmd.get_coords('%s and not hydro' % n) for n in ligands])
    target = lig.mean(axis=0)
    d = clear_view(rec, target)
    rec_axis = np.linalg.svd(rec - rec.mean(axis=0), full_matrices=False)[2][0]
    rot = view_basis(d, rec_axis, np.cross(d, rec_axis))     # receptor's long axis level
    for name in ['rec'] + ligands:
        cmd.load_coordset(cmd.get_coords(name) @ rot.T, name, state=1)
    pts = np.concatenate([rec, lig]) @ rot.T
    c = target @ rot.T                                      # pocket in the middle
    half = np.abs(pts[:, :2] - c[:2]).max() + args.margin
    zmid = (pts[:, 2].min() + pts[:, 2].max()) / 2
    depth = (pts[:, 2].max() - pts[:, 2].min()) / 2 + args.margin
    dist = half / np.tan(np.radians(cmd.get_setting_float('field_of_view')) / 2)
    view = list(cmd.get_view())
    view[:12] = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, -dist]
    view[12:15] = [float(c[0]), float(c[1]), float(zmid)]
    view[15:17] = [dist - depth, dist + depth]
    return view, {'view_rotation': np.round(rot, 4).tolist(), 'half_A': round(float(half), 2)}


def main():
    args = parse_args()
    cmd.feedback('disable', 'all', 'everything')
    ligands = load_poses(args)
    style(args, ligands)
    view, info = frame(args, ligands)
    stills = os.path.join(args.out, 'stills')
    os.makedirs(stills, exist_ok=True)
    paths = []
    for name in ligands:
        cmd.disable('lig*')
        cmd.enable(name)
        cmd.set_view(view)
        path = os.path.join(stills, '%s.png' % name)
        cmd.png(path, width=args.size, height=args.size, ray=1, quiet=1)
        paths.append(path)
        print('rendered %s' % name, flush=True)
    with open(os.path.join(args.out, 'meta.json'), 'w') as f:
        json.dump(dict(info, poses=args.complexes, args=vars(args)), f, indent=1)
    assemble(paths, args.out, args.hold, args.fade)


if __name__ == '__main__':
    main()
