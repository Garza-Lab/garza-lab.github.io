# md-hero — homepage trajectory loop

Renders the transparent, looping molecular dynamics video at the top of the
homepage (`uploads/hero/`).

Current loop: apo HPV-16 E6, replica 2 of `trajectory_35us.xtc`, all-white
surface. `meta.json`, written next to the rendered frames, records the exact
source frames.

## Requirements (macOS)

- PyMOL 3.x — its bundled Python provides `pymol` and numpy
- Blender 4.4 or newer — encodes WebM/VP9 with alpha and ProRes 4444
- `avconvert` — ships with macOS; writes the HEVC-with-alpha file Safari needs

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
and `e6-hero-poster.webp`. Keep each video under ~3 MB: raise the 4th
argument of `encode.sh` (VP9 CRF) or lower the 5th (ProRes scale, percent).

## Tests

```bash
python3 -m pytest -q -p no:cacheprovider tools/md-hero   # needs numpy + pytest
```

Never commit trajectories or rendered frames.
