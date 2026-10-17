"""Encode frame_####.png (RGBA) into a transparent WebM with Blender's FFmpeg.

  Blender -b --factory-startup --python-exit-code 1 \
      --python tools/md-hero/encode_blender.py -- --frames DIR --fps 30 \
      --webm e6-hero.webm --poster e6-hero-poster.webp
"""
import argparse
import glob
import os
import shutil
import sys
import tempfile

import bpy


def parse_args():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--frames', required=True)
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--crf', type=int, default=32, help='VP9 quality, lower = larger')
    p.add_argument('--webm', help='output .webm (VP9 + alpha)')
    p.add_argument('--poster', help='output .webp of the first frame')
    return p.parse_args(argv)


def build_scene(frames, fps):
    files = sorted(glob.glob(os.path.join(frames, 'frame_*.png')))
    if not files:
        raise SystemExit('no frame_*.png in %s' % frames)
    scene = bpy.context.scene
    r = scene.render
    r.resolution_x, r.resolution_y = bpy.data.images.load(files[0]).size
    r.resolution_percentage = 100
    r.fps, r.fps_base = fps, 1.0
    r.film_transparent = True
    r.use_sequencer = True
    r.use_compositing = False
    scene.view_settings.view_transform = 'Standard'   # keep PyMOL's colors
    scene.view_settings.look = 'None'
    scene.frame_start, scene.frame_end = 1, len(files)
    strip = scene.sequence_editor_create().strips.new_image(
        name='frames', filepath=files[0], channel=1, frame_start=1)
    for f in files[1:]:
        strip.elements.append(os.path.basename(f))
    return scene, files


def encode_webm(scene, path, crf):
    r = scene.render
    r.image_settings.media_type = 'VIDEO'
    r.image_settings.file_format = 'FFMPEG'
    r.ffmpeg.format = 'WEBM'
    r.ffmpeg.codec = 'WEBM'                      # VP9
    r.ffmpeg.constant_rate_factor = 'CUSTOM'
    r.ffmpeg.custom_constant_rate_factor = crf
    r.ffmpeg.ffmpeg_preset = 'BEST'
    r.ffmpeg.gopsize = 2 * r.fps
    r.image_settings.color_mode = 'RGBA'
    tmp = tempfile.mkdtemp()
    r.filepath = os.path.join(tmp, 'clip')
    bpy.ops.render.render(animation=True)
    made = glob.glob(os.path.join(tmp, 'clip*'))
    if len(made) != 1:
        raise SystemExit('expected one encoded file, got %r' % made)
    shutil.move(made[0], path)
    shutil.rmtree(tmp)


def poster(scene, first_png, path):
    s = scene.render.image_settings
    s.media_type = 'IMAGE'
    s.file_format = 'WEBP'
    s.color_mode = 'RGBA'
    s.quality = 90
    bpy.data.images.load(first_png).save_render(path, scene=scene)


def main():
    a = parse_args()
    scene, files = build_scene(a.frames, a.fps)
    if a.webm:
        encode_webm(scene, a.webm, a.crf)
    if a.poster:
        poster(scene, files[0], a.poster)


main()
