import numpy as np
import pytest

from loop import (align_frames, align_sequential, crossfade, eased_times,
                  flipbook_schedule, gaussian_smooth,
                  interpolate, pick_window, pingpong_order, radial_fade,
                  replica_ranges, rmsd,
                  seam_blend, superpose)

RNG = np.random.default_rng(0)


def rotation(axis, deg):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    a = np.radians(deg)
    return np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * k @ k


def test_replica_ranges_match_e6_layout():
    assert replica_ranges([10000, 10000, 5001, 10001]) == [
        (0, 10000), (10000, 20000), (20000, 25001), (25001, 35002)]


def test_superpose_recovers_rigid_motion():
    p = RNG.normal(size=(50, 3))
    q = p @ rotation([1, 2, 3], 40).T + [5.0, -2.0, 1.0]
    r, t = superpose(p, q)
    assert np.allclose(p @ r.T + t, q, atol=1e-6)
    assert np.isclose(np.linalg.det(r), 1.0)


def test_superpose_never_reflects():
    p = RNG.normal(size=(50, 3))
    r, _ = superpose(p, p * [1, 1, -1])
    assert np.isclose(np.linalg.det(r), 1.0)


def test_align_frames_fits_on_core_only():
    ref = RNG.normal(size=(20, 3))
    moved = ref.copy()
    moved[-1] += [3.0, 0.0, 0.0]                         # a wandering tail atom
    frame = moved @ rotation([0, 0, 1], 70).T + [10.0, 0.0, 0.0]
    core = np.arange(19)
    out = align_frames(frame[None], ref, core)[0]
    assert np.allclose(out[core], ref[core], atol=1e-6)
    assert np.allclose(out[-1], ref[-1] + [3.0, 0.0, 0.0], atol=1e-6)


def test_rmsd_of_a_shift():
    a = RNG.normal(size=(10, 3))
    assert np.isclose(rmsd(a, a + [3.0, 4.0, 0.0]), 5.0)


def test_gaussian_smooth_keeps_lines_and_cuts_noise():
    t = np.arange(40, dtype=float)
    line = np.stack([t, 2 * t, -t], axis=-1)[:, None, :]
    assert np.allclose(gaussian_smooth(line, 2.0)[8:-8], line[8:-8])
    noise = RNG.normal(size=(400, 5, 3))
    assert gaussian_smooth(noise, 3.0).std() < 0.5 * noise.std()


def test_gaussian_smooth_zero_sigma_is_a_copy():
    x = RNG.normal(size=(10, 2, 3))
    assert np.array_equal(gaussian_smooth(x, 0), x)


def test_pick_window_rejects_bad_seams_even_with_a_big_swing():
    n, length, blend = 200, 40, 5
    coords = np.zeros((n, 1, 3))
    coords[70:170, 0, 0] = np.arange(100)        # ramp: windows over it never close
    coords[170:, 0, 0] = 100.0
    aperture = np.full(n, 1.3)
    aperture[10:16] = 1.5                        # small swing where seams are clean
    aperture[120:131] = 2.0                      # big swing where seams are bad
    start, seam, swing = pick_window(aperture, coords, length, blend)
    assert 0 <= start < 10
    assert seam == 0.0
    assert np.isclose(swing, 0.2)


def test_pick_window_takes_biggest_swing_among_clean_seams():
    n, length, blend = 200, 40, 5
    coords = np.zeros((n, 1, 3))
    aperture = np.full(n, 1.3)
    aperture[30:36] = 1.5
    aperture[100:106] = 2.0
    start, _, swing = pick_window(aperture, coords, length, blend)
    assert 66 <= start <= 99
    assert np.isclose(swing, 0.7)


def test_pick_window_needs_enough_frames():
    with pytest.raises(ValueError):
        pick_window(np.ones(40), np.zeros((40, 1, 3)), 40, 5)


def test_seam_blend_wraps_onto_the_natural_next_frame():
    seq = np.arange(50, dtype=float)[:, None, None] * np.ones((1, 2, 3))
    out = seam_blend(seq, 40, 10)
    assert out.shape == (40, 2, 3)
    assert np.allclose(out[0], seq[40])        # restart = the frame after the last
    assert np.allclose(out[9], seq[9])         # fade finished
    assert np.allclose(out[10:], seq[10:40])   # rest untouched


def test_seam_blend_needs_length_plus_blend_frames():
    with pytest.raises(ValueError):
        seam_blend(np.zeros((45, 1, 3)), 40, 10)


def test_align_sequential_removes_tumbling():
    base = RNG.normal(size=(15, 3))
    frames = np.stack([base @ rotation([1, 1, 0], 25 * k).T + [k, 0.0, 0.0] for k in range(6)])
    out = align_sequential(frames, np.arange(15))
    assert np.allclose(out, base, atol=1e-6)


def test_align_sequential_keeps_internal_motion():
    base = RNG.normal(size=(15, 3))
    frames = np.stack([base.copy() for _ in range(3)])
    frames[1, -1] += [2.0, 0.0, 0.0]
    frames[2] = frames[2] @ rotation([0, 0, 1], 90).T
    out = align_sequential(frames, np.arange(14))
    assert np.allclose(out[1, -1], base[-1] + [2.0, 0.0, 0.0], atol=1e-6)
    assert np.allclose(out[2], base, atol=1e-6)


def test_eased_times_span_and_pace():
    t = eased_times(120, 360, 30)
    assert t.shape == (360,)
    assert t[0] == 0.0 and np.isclose(t[-1], 119.0)
    steps = np.diff(t)
    assert (steps > 0).all()
    assert np.allclose(steps[40:-40], steps[180])            # constant pace mid-pass
    assert steps[0] < 0.1 * steps[180] and steps[-1] < 0.1 * steps[180]


def test_eased_times_without_easing_is_linear():
    assert np.allclose(eased_times(5, 9, 0), np.linspace(0, 4, 9))


def test_pingpong_order_never_repeats_a_frame_at_the_turns():
    order = pingpong_order(4)
    assert order == [0, 1, 2, 3, 2, 1]
    loop = order + order[:1]
    assert all(abs(a - b) == 1 for a, b in zip(loop, loop[1:]))


def test_interpolate_hits_frames_and_midpoints():
    coords = np.arange(3, dtype=float)[:, None, None] * np.ones((1, 2, 3))
    out = interpolate(coords, np.array([0.0, 0.5, 1.0, 2.0]))
    assert np.allclose(out[:, 0, 0], [0.0, 0.5, 1.0, 2.0])


def test_flipbook_schedule_holds_then_fades_and_wraps():
    assert flipbook_schedule(3, 2, 1) == [
        (0, 0, 0.0), (0, 0, 0.0), (0, 1, 0.5),
        (1, 1, 0.0), (1, 1, 0.0), (1, 2, 0.5),
        (2, 2, 0.0), (2, 2, 0.0), (2, 0, 0.5)]


def test_crossfade_endpoints_and_opaque_average():
    red = np.zeros((2, 2, 4), np.uint8); red[..., 0] = 255; red[..., 3] = 255
    blue = np.zeros((2, 2, 4), np.uint8); blue[..., 2] = 255; blue[..., 3] = 255
    assert np.array_equal(crossfade(red, blue, 0.0), red)
    assert np.array_equal(crossfade(red, blue, 1.0), blue)
    mid = crossfade(red, blue, 0.5)
    assert tuple(mid[0, 0]) == (128, 0, 128, 255)


def test_crossfade_to_transparent_keeps_color_and_halves_alpha():
    red = np.zeros((2, 2, 4), np.uint8); red[..., 0] = 255; red[..., 3] = 255
    clear = np.zeros((2, 2, 4), np.uint8)
    assert tuple(crossfade(red, clear, 0.5)[0, 0]) == (255, 0, 0, 128)


def test_radial_fade_keeps_centre_and_clears_corners():
    img = np.full((100, 100, 4), 255, np.uint8)
    out = radial_fade(img, 0.8, 0.95)
    assert out[50, 50, 3] == 255                     # centre untouched
    assert out[0, 0, 3] == 0                         # corners gone
    assert 0 < out[50, 94, 3] < 255                  # partway through the fade
    assert np.array_equal(out[..., :3], img[..., :3])   # colour untouched
