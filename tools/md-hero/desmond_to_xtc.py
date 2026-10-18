"""Convert a Desmond trajectory into a solute-only PDB + XTC for the md-hero scripts.

Run with PyMOL's bundled Python (PyMOL reads the .cms, MDTraj the .dtr), e.g.

  /Applications/PyMOL.app/Contents/bin/python tools/md-hero/desmond_to_xtc.py \
      --cms taxifolin_sample-out.cms --trj taxifolin_sample_trj --out /tmp/md-hero/taxifolin

Writes OUT.pdb (the kept atoms, at the first frame's coordinates) and OUT.xtc (every
frame of those atoms, with Desmond's time stamps). Water, ions and virtual sites
(force-field pseudo-atoms) are dropped by default, and everything that is not
protein is written as HETATM, so viewers treat a ligand as a ligand. Frames are
written as stored: molecules are whole, but the complex can jump between periodic
images, which aligning the frames removes.
"""
import argparse
import os

import mdtraj as md
import numpy as np
from pymol import cmd


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--cms', required=True, help='Desmond -out.cms (topology)')
    p.add_argument('--trj', required=True, help='the *_trj folder, or its clickme.dtr')
    p.add_argument('--keep', default='not (resn T3P+T4P+SPC+HOH+WAT+NA+CL+K or elem PS)',
                   help='PyMOL selection of the atoms to keep '
                        '(default: drop water, ions and virtual sites)')
    p.add_argument('--out', required=True, help='output path without extension')
    return p.parse_args()


def main():
    args = parse_args()
    cmd.feedback('disable', 'all', 'everything')
    cmd.set('retain_order', 1)     # keep file atom order: it is the trajectory's order
    cmd.load(args.cms, 'full')
    keep = 'full and (%s)' % args.keep
    idx = np.array([a.index - 1 for a in cmd.get_model(keep).atom])
    dtr = args.trj if args.trj.endswith('.dtr') else os.path.join(args.trj, 'clickme.dtr')
    with md.formats.DTRTrajectoryFile(dtr) as f:
        if f.read(n_frames=1)[0].shape[1] != cmd.count_atoms('full'):
            raise SystemExit('the trajectory and the .cms have different atom counts')
        f.seek(0)
        xyz, time, lengths, angles = f.read(atom_indices=idx)          # Å, ps
    cmd.create('solute', keep, 1, 1)
    cmd.delete('full')
    cmd.alter('solute and not polymer.protein', 'type = "HETATM"')
    cmd.load_coordset(xyz[0], 'solute', state=1)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cmd.save(args.out + '.pdb', 'solute')
    box = md.utils.lengths_and_angles_to_box_vectors(*(lengths.T / 10.0), *angles.T)
    with md.formats.XTCTrajectoryFile(args.out + '.xtc', 'w') as f:
        f.write(xyz / 10.0, time=time, box=np.stack(box, axis=1))
    print('kept %d atoms, %d frames, %.1f-%.1f ps' % (len(idx), len(xyz), time[0], time[-1]))


if __name__ == '__main__':
    main()
