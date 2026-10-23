"""Offline checks for the scripted pick-and-place controller (no robot needed).

Runs under pytest, or directly: `python3 -m tests.test_auto_pickplace` from
the lerobot_ur5e_keyteleop directory.
"""

import numpy as np

from scripts.core.auto_pickplace import (
    _bezier_length,
    _cubic_bezier,
    _min_jerk,
    _transit_control_points,
)


def test_min_jerk_endpoints_and_symmetry():
    assert _min_jerk(0.0) == 0.0
    assert _min_jerk(1.0) == 1.0
    assert _min_jerk(-1.0) == 0.0 and _min_jerk(2.0) == 1.0
    assert abs(_min_jerk(0.5) - 0.5) < 1e-12
    for tau in np.linspace(0.0, 1.0, 11):
        assert abs(_min_jerk(tau) + _min_jerk(1.0 - tau) - 1.0) < 1e-12


def test_min_jerk_starts_and_ends_at_rest():
    eps = 1e-4
    assert _min_jerk(eps) / eps < 1e-6
    assert (1.0 - _min_jerk(1.0 - eps)) / eps < 1e-6
    # Peak speed at the midpoint is 1.875x the average.
    peak = (_min_jerk(0.5 + eps) - _min_jerk(0.5 - eps)) / (2 * eps)
    assert abs(peak - 1.875) < 1e-6


def test_transit_path_lifts_then_lowers():
    start = np.array([0.45, -0.3, 0.05])
    end = np.array([0.2, -0.44, 0.09])
    points = _transit_control_points(start, end, cruise_z=0.25)
    assert np.allclose(_cubic_bezier(points, 0.0), start)
    assert np.allclose(_cubic_bezier(points, 1.0), end)
    zs = [_cubic_bezier(points, s)[2] for s in np.linspace(0.0, 1.0, 21)]
    peak_idx = int(np.argmax(zs))
    assert 0 < peak_idx < 20
    assert zs[peak_idx] > max(start[2], end[2]) + 0.1
    assert _bezier_length(points) > np.linalg.norm(end - start)


def _run_all():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")


if __name__ == "__main__":
    _run_all()
