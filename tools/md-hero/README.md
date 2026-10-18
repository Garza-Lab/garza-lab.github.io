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

## Research page: Aβ42 cartoon loop

`render_cartoon.py` makes the hotpink Aβ42 movie on the Research page
(`uploads/research/`). It shows the simplified DSSP (H/E/C, MDTraj) of every
frame as cartoon helices/strands, so transient structure appears exactly where
and when it occurs. The stretch plays forward then backward (ping-pong) with
eased turns, because a disordered chain's start and end never match.

Current loop: `AB42_bAIes_full_protein.xtc` frames 27441–27560 (replica 42,
frames 94–213, 12 ns), the 12 ns stretch with the most helix/strand frames.

```bash
ARGS="--top AB42_bAIes_full_protein.pdb --traj AB42_bAIes_full_protein.xtc --first 27441 --count 120"
for k in 0 1 2 3; do
  $PYMOL_PY tools/md-hero/render_cartoon.py $ARGS --out /tmp/md-hero/ab42 --part $k --parts 4 &
done; wait
$PYMOL_PY tools/md-hero/render_cartoon.py --out /tmp/md-hero/ab42 --mirror   # return pass
NAME=ab42-baies tools/md-hero/encode.sh /tmp/md-hero/ab42 uploads/research 30 36 450 0.6
```

## Research page: NUPR1–ZZW115 docking flipbook

`render_flipbook.py` makes the movie beside the Research intro
(`uploads/research/nupr1-zzw115.*`). `NUPR1_ZZW115_all_20000_frames.xtc` holds
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

## Tests

```bash
python3 -m pytest -q -p no:cacheprovider tools/md-hero   # needs numpy + pytest
```

Never commit trajectories or rendered frames.
