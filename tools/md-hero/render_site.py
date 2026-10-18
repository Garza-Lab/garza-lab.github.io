"""Render a looping, transparent movie of a protein-ligand complex that starts on the
whole complex, zooms into its active site, holds there and zooms back out, in the
lab's IDP_style.

Run with PyMOL's bundled Python (it provides pymol, numpy and mdtraj), e.g.

  /Applications/PyMOL.app/Contents/bin/python tools/md-hero/render_site.py \
      --top taxifolin_solute.pdb --traj taxifolin_solute.xtc --segment 0 1001 \
      --stride 2 --style ~/.pymol/startup/IDP_style.py --ligand 999 cyan \
      --spheres 'resn CU+PER' --out /tmp/md-hero/tyrosinase

The motion is the homepage's kind of loop (see render_complex.py): a window whose
ends nearly match, smoothed, with a blended seam. The active site is the ligand,
the --spheres atoms (metals and what bridges them, colored by element) and the side
chains of the residues that coordinate the metals, shown as sticks. Ligand and
side-chain carbons take the ligand and protein colors; other atoms PyMOL's element
colors. The camera looks at the site from its most open side that keeps the ligand
beside the metals rather than in front of them, with the metals level and the
ligand above. --clip cuts away what lies in front of the site in the close-up.
Metal coordination (any metal-to-heavy-atom pair within --contact Å in the median
frame) is drawn as dotted black lines.
"""
import argparse
import json
import os
import runpy
import sys
import time

import numpy as np
from pymol import cmd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loop import clear_view, view_basis, zoom_camera, zoom_schedule  # noqa: E402
from render_complex import build_loop, load_segment  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--top', required=True, help='complex PDB, atoms in trajectory order')
    p.add_argument('--traj', required=True, help='trajectory MDTraj can read')
    p.add_argument('--segment', type=int, nargs=2, required=True, metavar=('START', 'STOP'),
                   help='frames [START, STOP) of --traj that form one continuous run')
    p.add_argument('--stride', type=int, default=1, help='trajectory frames per output frame')
    p.add_argument('--sigma', type=float, default=3.0, help='smoothing, output frames')
    p.add_argument('--blend', type=int, default=60, help='seam fade, output frames')
    p.add_argument('--start', type=int, help='force the window start (skip the search)')
    p.add_argument('--hold-wide', type=int, default=75, help='frames on the whole complex')
    p.add_argument('--move', type=int, default=75, help='frames of each zoom')
    p.add_argument('--hold-close', type=int, default=135, help='frames on the active site')
    p.add_argument('--style', required=True, help='script defining IDP_style()')
    p.add_argument('--protein-color', default='white')
    p.add_argument('--ligand', nargs=2, action='append', default=[], metavar=('RESI', 'COLOR'),
                   help='ligand residue number and its color; repeat per ligand')
    p.add_argument('--spheres', required=True,
                   help='PyMOL selection shown as spheres, colored by element')
    p.add_argument('--sphere-scale', type=float, default=0.3, help='non-metal spheres')
    p.add_argument('--metal-scale', type=float, default=0.5, help='metal spheres')
    p.add_argument('--max-cos', type=float, default=0.6,
                   help='keep the view at least arccos(this) off the metals-to-ligand '
                        'axis, so the ligand sits beside the metals, not on them')
    p.add_argument('--contact', type=float, default=2.6,
                   help='metal-coordination distance, Å')
    p.add_argument('--clip', type=float,
                   help='in the close-up, clip away what is more than this many Å in '
                        'front of the site (default: no clipping)')
    p.add_argument('--size', type=int, default=900, help='square image size, px')
    p.add_argument('--every', type=int, default=1, help='render every k-th frame')
    p.add_argument('--only', type=int, nargs='+', help='render just these output frames')
    p.add_argument('--draft', action='store_true', help='fast preview quality')
    p.add_argument('--part', type=int, default=0, help='this worker, 0-based')
    p.add_argument('--parts', type=int, default=1, help='number of parallel workers')
    p.add_argument('--out', required=True, help='folder for frame_####.png')
    args = p.parse_args()
    args.frames = args.hold_wide + 2 * args.move + args.hold_close   # loop length
    return args


def indices(sel):
    """0-based atom indices (trajectory order) of a PyMOL selection on 'cx'."""
    return np.array([a.index - 1 for a in cmd.get_model('cx and (%s)' % sel).atom], dtype=int)


def ligand_selection(args):
    """PyMOL selection of every --ligand residue."""
    return ' or '.join('resi %s' % resi for resi, _ in args.ligand) or 'none'


def setup_scene(args, atoms):
    """Load the complex, apply IDP_style, show and color the site. Returns the
    selections of the coordinating side chains and of the site."""
    cmd.set('retain_order', 1)     # keep file atom order so coordinates line up
    cmd.set('auto_zoom', 0)
    cmd.load(args.top, 'cx')
    if [(int(a.resi), a.symbol) for a in cmd.get_model('cx').atom] != atoms:
        raise SystemExit('PyMOL atom order does not match the trajectory')
    runpy.run_path(os.path.expanduser(args.style))['IDP_style']()
    cmd.hide('everything')
    cmd.show('cartoon', 'cx and polymer.protein')
    cmd.color(args.protein_color, 'cx and polymer.protein')
    for resi, color in args.ligand:
        cmd.show('sticks', 'cx and resi %s' % resi)
        cmd.color(color, 'cx and resi %s and elem C' % resi)
        cmd.color('atomic', 'cx and resi %s and not elem C' % resi)   # N blue, O red, ...
    spheres = 'cx and (%s)' % args.spheres
    coord = ('byres ((cx and polymer.protein and not hydro) within %g of (%s and metals))'
             % (args.contact, spheres))
    sidechains = '(%s) and (sidechain or name CA) and not hydro' % coord
    cmd.show('sticks', sidechains)
    cmd.color('atomic', '(%s) and not elem C' % sidechains)          # carbons keep the protein color
    cmd.show('spheres', spheres)
    cmd.set('sphere_scale', args.sphere_scale, spheres)
    cmd.set('sphere_scale', args.metal_scale, spheres + ' and metals')
    cmd.color('atomic', spheres)   # PyMOL's element colors
    cmd.hide('everything', 'hydro')
    cmd.dss('cx')                  # helices from the first frame; they persist
    if args.draft:
        cmd.set('antialias', 1)
    residues = sorted({(a.resn, int(a.resi)) for a in cmd.get_model(sidechains).atom},
                      key=lambda r: r[1])
    site = '(%s) or (%s) or ((%s) and not hydro)' % (sidechains, args.spheres,
                                                      ligand_selection(args))
    return sidechains, site, residues


def coordination(loop, metals, partners, cutoff):
    """(metal, atom) index pairs whose median distance over the loop is <= cutoff."""
    pairs = []
    for m in metals:
        d = np.median(np.linalg.norm(loop[:, partners] - loop[:, [m]], axis=-1), axis=0)
        pairs += [(int(m), int(a), float(x)) for a, x in zip(partners, d) if x <= cutoff]
    return pairs


def draw_coordination(pairs):
    """Dotted black lines that follow the atoms as the coordinates change."""
    for k, (m, a, _) in enumerate(pairs):
        cmd.distance('coord%02d' % k, 'cx and index %d' % (m + 1), 'cx and index %d' % (a + 1))
    cmd.hide('labels', 'coord*')
    cmd.set('dash_color', 'black')
    cmd.set('dash_radius', 0.07)
    cmd.set('dash_length', 0.12)
    cmd.set('dash_gap', 0.22)       # IDP_style draws solid dashes (gap 0)
    cmd.set('dash_round_ends', 1)


def choose_view(loop, site, metals, ligand, occluders, max_cos):
    """Camera rotation: from the site's most open side that keeps the ligand beside
    the metals, metals level, ligand above."""
    mean = loop.mean(axis=0)
    center = mean[site].mean(axis=0)
    axis = mean[ligand].mean(axis=0) - mean[metals].mean(axis=0)
    d = clear_view(mean[occluders], center, across=axis, max_cos=max_cos)
    m = mean[metals] - mean[metals].mean(axis=0)
    along = np.linalg.svd(m, full_matrices=False)[2][0]     # the metals' long axis
    return view_basis(d, along, mean[ligand].mean(axis=0) - center)


def camera_path(args, loop, shown, site, margin=3.0, close_margin=5.0):
    """One PyMOL view per output frame (coordinates already in camera axes)."""
    pts, near = loop[:, shown].reshape(-1, 3), loop[:, site].reshape(-1, 3)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    slo, shi = near.min(axis=0), near.max(axis=0)
    wide = ((lo[:2] + hi[:2]) / 2, max(hi[0] - lo[0], hi[1] - lo[1]) / 2 + margin)
    close = ((slo[:2] + shi[:2]) / 2, max(shi[0] - slo[0], shi[1] - slo[1]) / 2 + close_margin)
    zmid = (lo[2] + hi[2]) / 2
    front_wide, back = hi[2] + margin, lo[2] - margin
    front_close = front_wide if args.clip is None else min(shi[2] + args.clip, front_wide)
    tan = np.tan(np.radians(cmd.get_setting_float('field_of_view')) / 2)
    views = []
    for s in zoom_schedule(args.hold_wide, args.move, args.hold_close):
        c, h, w = zoom_camera(wide, close, s)
        front = front_wide + w * (front_close - front_wide)   # clipping follows the zoom
        dist = h / tan
        views.append([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -dist,
                      float(c[0]), float(c[1]), float(zmid),
                      float(dist - (front - zmid)), float(dist + (zmid - back)),
                      cmd.get_view()[17]])
    return views, {'wide_half_A': round(float(wide[1]), 2), 'close_half_A': round(float(close[1]), 2)}


def render(args, loop, views):
    os.makedirs(args.out, exist_ok=True)
    picks = args.only or list(range(0, len(loop), args.every))[args.part::args.parts]
    cmd.set('max_threads', max(1, (os.cpu_count() or 1) // args.parts))
    t0 = time.time()
    for k, i in enumerate(picks, start=1):
        cmd.load_coordset(loop[i], 'cx', state=1)
        cmd.set_view(views[i])
        name = i if args.only else i // args.every
        cmd.png(os.path.join(args.out, 'frame_%04d.png' % name),
                width=args.size, height=args.size, ray=1, quiet=1)
        if k % 10 == 0 or k == len(picks):
            print('part %d: %d/%d frames, %.1f s/frame'
                  % (args.part, k, len(picks), (time.time() - t0) / k), flush=True)


def main():
    args = parse_args()
    cmd.feedback('disable', 'all', 'everything')
    coords, core, lig, shown, atoms = load_segment(args)
    loop, info = build_loop(args, coords, core, lig)
    sidechains, site, residues = setup_scene(args, atoms)
    site_idx, metals = indices(site), indices('(%s) and metals' % args.spheres)
    shown = np.union1d(shown, site_idx)
    occluders = indices('polymer.protein and not hydro and not (%s)' % sidechains)
    rot = choose_view(loop, site_idx, metals, lig, occluders, args.max_cos)
    loop = (loop @ rot.T).astype(np.float32)
    cmd.load_coordset(loop[0], 'cx', state=1)
    partners = indices('not hydro and not metals and (polymer.protein or (%s) or %s)'
                       % (args.spheres, ligand_selection(args)))
    pairs = coordination(loop, metals, partners, args.contact)
    draw_coordination(pairs)
    views, framing = camera_path(args, loop, shown, site_idx)
    if args.part == 0 and not args.only:
        model = cmd.get_model('cx').atom

        def label(i):
            return '%s%s:%s' % (model[i].resn, model[i].resi, model[i].name)
        info.update(framing, coordinating=['%s%d' % r for r in residues],
                    coordination=['%s-%s %.2f A' % (label(m), label(a), x) for m, a, x in pairs],
                    view_rotation=np.round(rot, 4).tolist(), args=vars(args))
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, 'meta.json'), 'w') as f:
            json.dump(info, f, indent=2)
        print(json.dumps({k: info[k] for k in ('window_start', 'seam_rmsd_A',
                                               'ligand_excursion_A', 'traj_frames',
                                               'coordinating', 'coordination',
                                               'wide_half_A', 'close_half_A')}),
              flush=True)
    render(args, loop, views)


if __name__ == '__main__':
    main()
