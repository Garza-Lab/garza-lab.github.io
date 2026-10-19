# md-hero — homepage trajectory loop

Renders the transparent, looping molecular dynamics video at the top of the
homepage (`uploads/hero/`).

Current loop: apo HPV-16 E6, replica 2 of `trajectory_35us.xtc`, all-white
surface. `meta.json`, written next to the rendered frames, records the exact
source frames.

## Requirements (macOS)

- PyMOL 3.x — its bundled Python provides `pymol` and numpy
- Blender 4.4 or newer — encodes the WebM (VP9 with alpha) and the poster
- Xcode Command Line Tools (`clang`) — builds `encode_hevc.m`, which drives
  macOS's encoder for the HEVC-with-alpha file Safari needs

## Render

```bash
PYMOL_PY=/Applications/PyMOL.app/Contents/bin/python
ARGS="--top topology_35us.pdb --traj trajectory_35us.xtc --cvs cvs_35us.dat"

# fast preview: 480 px, every 2nd frame, draft quality
$PYMOL_PY tools/md-hero/render.py $ARGS --out /tmp/md-hero/preview \
    --size 480 --every 2 --draft

# final: 1080 px, 360 frames, split across 4 PyMOL processes
for k in 0 1 2 3; do
  $PYMOL_PY tools/md-hero/render.py $ARGS --out /tmp/md-hero/final \
      --part $k --parts 4 &
done; wait
```

Options: `--start N` reuses a window (see `meta.json`), `--turn X Y` rotates
the camera, `--stride` / `--sigma` / `--blend` set pace, smoothing and seam
fade, `--replica` picks another replica.

## Encode

```bash
tools/md-hero/encode.sh /tmp/md-hero/final uploads/hero          # 30 fps
tools/md-hero/encode.sh /tmp/md-hero/preview uploads/hero 15     # preview

# check the Safari file really carries alpha (expect codec=hvc1 alpha=true)
clang -fobjc-arc -framework Foundation -framework AVFoundation \
    -framework CoreMedia tools/md-hero/check_mov.m -o /tmp/check_mov
/tmp/check_mov uploads/hero/e6-hero.mov
```

Outputs `e6-hero.webm` (Chrome, Firefox, Edge), `e6-hero.mov` (Safari, iOS)
and `e6-hero-poster.webp`. The defaults keep each video under ~3 MB (WebM
2.7 MB, .mov 2.7 MB for the current 1080 px loop). To shrink further, raise the 4th
argument of `encode.sh` (VP9 CRF) or lower the 5th (HEVC color bitrate, kbps)
or 6th (HEVC alpha quality, 0–1).

## Research page: NUPR1 cartoon loop

`render_cartoon.py` makes the cyan NUPR1 movie beside Intrinsically Disordered
Proteins (`uploads/research/nupr1-baies.*`), in the lab's `IDP_style()` with its
thicker disordered-chain tube. It shows the simplified DSSP (H/E/C, MDTraj) of
every frame as cartoon helices/strands, so transient structure appears exactly
where and when it occurs. Coordinates are smoothed for display, but each helix is
drawn in the shape it has in its own raw frame; smoothing alone would crumple a
helix that lasts a frame or two. The stretch plays forward then backward
(ping-pong) with eased turns, because a disordered chain's start and end never
match.

Current loop: `NUPR1_bAIes_all_120000_frames.xtc` frames 70725–70784 (replica 71,
stored frames 726–785; 6 ns of bAIes time, which is not physical kinetics). Of the
6 ns stretches in all 120 replicas, it is the one in which a helix of two turns
or more (residues ~70–78, near the C-terminus) appears most often.

```bash
D=nupr1_baies_all_frames_20261017_v1
ARGS="--top $D/NUPR1_bAIes_all_120000_frames.pdb --traj $D/NUPR1_bAIes_all_120000_frames.xtc
      --first 70725 --count 60 --sigma 2 --color cyan
      --style ~/.pymol/startup/IDP_style.py --size 900"
for k in 0 1 2 3; do
  $PYMOL_PY tools/md-hero/render_cartoon.py $ARGS --out /tmp/md-hero/nupr1 --part $k --parts 4 &
done; wait
$PYMOL_PY tools/md-hero/render_cartoon.py --out /tmp/md-hero/nupr1 --mirror   # return pass
NAME=nupr1-baies tools/md-hero/encode.sh /tmp/md-hero/nupr1 uploads/research 30 40 450 0.5
```

The 24 s loop comes to WebM 1.7 MB and .mov 2.3 MB: the thin chain covers little of
the frame, so it affords a lower CRF than the receptor movie.

This replaced a hotpink Aβ42 loop in the homepage's matte look
(`AB42_bAIes_full_protein.xtc` frames 27441–27560), made with `render_cartoon.py`
as of commit 8955c8a.

## Research page: NUPR1–ZZW115 docking flipbook

`render_flipbook.py` made a flipbook that briefly sat beside the Research intro
(since taken down, with its files). `NUPR1_ZZW115_all_20000_frames.xtc` holds
independent DiffDock-L complexes, not a trajectory, so nothing is animated: the
script takes the highest-confidence complex of each of the 20 clusters, aligns
them on ZZW115 (the ligand stays put), renders each as a still with the lab's
`IDP_style()` (NUPR1 cyan, ZZW115 white sticks, each structure's own DSSP), and
holds each still before dissolving into the next. Tails beyond the 55 Å window
fade out at the edge.

`--style` points at your local `IDP_style.py` (normally
`~/.pymol/startup/IDP_style.py`); it is not stored in this repo.

```bash
D=NUPR1_ZZW115_all_frames_export_folder
$PYMOL_PY tools/md-hero/render_flipbook.py --top $D/NUPR1_ZZW115_all_20000_frames.pdb \
    --traj $D/NUPR1_ZZW115_all_20000_frames.xtc --frame-map $D/frame_map.tsv \
    --style ~/.pymol/startup/IDP_style.py --out /tmp/md-hero/nupr1
NAME=nupr1-zzw115 tools/md-hero/encode.sh /tmp/md-hero/nupr1 uploads/research 30 36 450 0.6
```

## Research page: mu-opioid receptor with naloxone and 368

`render_complex.py` makes the movie beside Protein Allostery
(`uploads/research/mor-naloxone-368.*`): the mu-opioid receptor (from PDB 9BJK)
as a white cartoon, with naloxone (lime) in the orthosteric pocket and the
negative allosteric modulator 368 (hotpink) above it, in the lab's
`IDP_style()`; only the ligand sticks are thickened (0.45 Å) so they read at
web size. The loop is built like the homepage's: within one continuous
replica, among the windows whose first and last frames match best, the one in
which the ligands move furthest from their usual pose and come back, aligned
on the helices, smoothed, with a blended seam. The camera looks along the
membrane, extracellular side up, raised 15°.

Current loop: replica 3 of `9BJK_NAM_merged_protein_ligands.xtc` (protein and
ligands only), every 8th frame of the 1 ns copy from 10585 to 13937 — 3.35 µs
of simulation per 12 s loop.

On the cluster, first keep every 10th frame (1 ns apart). In the result the
three replicas are frames [0, 5001), [5001, 10001) and [10001, 15001).

```bash
printf "System\n" | gmx trjconv -f 9BJK_NAM_merged_protein_ligands.xtc \
    -s 9BJK_NAM_merged_protein_ligands.gro -n 9BJK_NAM_merged_protein_ligands.ndx \
    -skip 10 -o 9BJK_NAM_protein_ligands_every10.xtc
```

Then, with that file and `9BJK_NAM_merged_protein_ligands.pdb` copied here:

```bash
ARGS="--top 9BJK_NAM_merged_protein_ligands.pdb --traj 9BJK_NAM_protein_ligands_every10.xtc
      --segment 10001 15001 --stride 8 --sigma 5 --style ~/.pymol/startup/IDP_style.py
      --ligand 501 lime --ligand 502 hotpink --ligand-radius 0.45
      --azimuth 15 --tilt 15 --size 900"
for k in 0 1 2 3; do
  $PYMOL_PY tools/md-hero/render_complex.py $ARGS --out /tmp/md-hero/mor --part $k --parts 4 &
done; wait
NAME=mor-naloxone-368 tools/md-hero/encode.sh /tmp/md-hero/mor uploads/research 30 48 450 0.5
```

Thin outlines moving every frame compress poorly: at 1080 px and CRF 36 the WebM
was 7.5 MB. 900 px with CRF 48 keeps it near 3 MB (.mov 2.3 MB) and still looks
clean at the ~550 px it is shown.

Add `--size 480 --every 2 --draft` (and a 15 fps encode) for a quick preview.

## Research page: tyrosinase–taxifolin zoom

`render_site.py` makes the movie in Enzyme Catalysis
(`uploads/research/tyrosinase-taxifolin.*`): human tyrosinase as a white cartoon
with taxifolin (cyan) in its pocket, in the lab's `IDP_style()`. It holds on the
whole enzyme, zooms into the dicopper active site, holds, and zooms back out. The
site shows the two copper ions and the bridging peroxide as spheres in PyMOL's
element colors, the side chains of the six copper-binding histidines (180, 202,
211, 363, 367, 390; white carbons, blue nitrogens; taxifolin has cyan carbons and
red oxygens; both drawn as thick 0.45 Å sticks, like the ligands in the other
movies) and dotted black lines for every metal bond (Cu–N and Cu–O within
2.6 Å in the median frame). Taxifolin's nearest oxygen stays 3.5–4.4 Å
from CuB in every frame, so it gets no coordination line. The camera looks at
the site from the side where least of it is hidden (site atoms behind the
cartoon, or the coppers and histidines behind taxifolin) without seeing
taxifolin edge-on.

Source: the Desmond run `taxifolin_sample-out.cms` + `taxifolin_sample_trj`
(100 ns, 1,001 frames). Current loop: every 2nd frame from 132 to 970, i.e. 84 ns
in a 12 s loop (window ends 2.1 Å apart, blended over 2 s).

`desmond_to_xtc.py` first turns the run into a PDB + XTC of the solute (it drops
water, ions and the force field's virtual sites):

```bash
D=path/to/desmond_run
$PYMOL_PY tools/md-hero/desmond_to_xtc.py --cms $D/taxifolin_sample-out.cms \
    --trj $D/taxifolin_sample_trj --out /tmp/md-hero/taxifolin
for k in 0 1 2 3; do
  $PYMOL_PY tools/md-hero/render_site.py --top /tmp/md-hero/taxifolin.pdb \
      --traj /tmp/md-hero/taxifolin.xtc --segment 0 1001 --stride 2 --sigma 3 \
      --style ~/.pymol/startup/IDP_style.py --ligand 999 cyan --spheres 'resn CU+PER' \
      --stick-radius 0.45 --size 800 --out /tmp/md-hero/tyrosinase --part $k --parts 4 &
done; wait
NAME=tyrosinase-taxifolin tools/md-hero/encode.sh /tmp/md-hero/tyrosinase uploads/research 30 54 450 0.4
```

During a zoom every pixel moves, which costs the encoder far more than slow MD
motion: at 900 px and CRF 48 this loop's WebM was 5.8 MB. 800 px with CRF 54
brings it to 3.7 MB (.mov 2.2 MB) and still looks clean at the ~550 px it is
shown.

## Research page: ten ligands docked to HPV-16 E6

`render_poses.py` makes the movie beside Computer-Aided Drug Design
(`uploads/research/e6-ligands.*`) from `4XR8_Ligand1.maegz` …
`4XR8_Ligand10.maegz`. Each file is one docked complex: HPV-16 E6 (from PDB
4XR8, chain H) with its two Zn ions and one ligand. The receptor is the same
rigid structure in every file, but files 2, 3, 6, 8 and 9 sit in a different
coordinate frame, so every complex is superposed on the first receptor (Cα)
before the ligands are placed. Water is removed and the ions are not shown.

E6 is an opaque white surface (`IDP_style()` with `transparency 0` and
`surface_quality 1`); each ligand is drawn as thick (0.45 Å) brightorange
sticks in one color, without element colors. The camera looks straight into
the pocket the ten ligands share, with the pocket in the middle and the whole
protein in frame, so nothing is cut at the edge and the background stays fully
transparent. Each pose is held for 2 s and dissolves into the next over 1.5 s;
the 35 s loop returns to the first.

```bash
$PYMOL_PY tools/md-hero/render_poses.py --style ~/.pymol/startup/IDP_style.py \
    --out /tmp/md-hero/e6-ligands 4XR8_Ligand{1,2,3,4,5,6,7,8,9,10}.maegz
NAME=e6-ligands tools/md-hero/encode.sh /tmp/md-hero/e6-ligands uploads/research 30 40 450 0.5
```

## Tests

```bash
python3 -m pytest -q -p no:cacheprovider tools/md-hero   # needs numpy + pytest
```

Never commit trajectories or rendered frames.
