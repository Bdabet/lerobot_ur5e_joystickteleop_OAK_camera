"""Offline checks for the scripted pick-and-place controller (no robot needed).

Runs under pytest, or directly: `python3 -m tests.test_auto_pickplace` from
the lerobot_ur5e_keyteleop directory.
"""

import random

import numpy as np
from scipy.spatial.transform import Rotation as R

from scripts.core.auto_pickplace import (
    AutoPickPlaceConfig,
    AutoPickPlaceController,
    Phase,
    _bezier_length,
    _cubic_bezier,
    _min_jerk,
    _transit_control_points,
)

FPS = 30
CANONICAL_ROTVEC = [np.pi, 0.0, 0.0]
CONFIG = {
    "pick_region": {"x_range": [0.4, 0.5], "y_range": [-0.55, -0.06]},
    "place_region": {"x_range": [0.14, 0.27], "y_range": [-0.47, -0.418]},
    "force_stop_enabled": True,
    "contact_force_threshold": 10.0,
}


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


class FakeRobot:
    """Integrates each commanded delta with a one-tick-style tracking lag, and
    reports a contact force once the TCP is pushed below contact_z."""

    def __init__(self, pose, tracking=0.8, contact_z=-np.inf):
        self.position = np.array(pose[:3], dtype=float)
        self.rotation = R.from_rotvec(pose[3:])
        self.tracking = tracking
        self.contact_z = contact_z

    def get_ee_pose(self):
        return [*self.position.tolist(), *self.rotation.as_rotvec().tolist()]

    def get_tcp_force(self):
        return [0.0, 0.0, 20.0 if self.position[2] < self.contact_z else 0.0]

    def apply(self, action):
        delta = np.array([action["delta_x"], action["delta_y"], action["delta_z"]])
        if self.position[2] + delta[2] < self.contact_z - 0.001:
            delta[2] = self.contact_z - 0.001 - self.position[2]  # can't push through a rigid surface
        self.position += self.tracking * delta
        rot = np.array([action["delta_rx"], action["delta_ry"], action["delta_rz"]])
        self.rotation = R.from_rotvec(self.tracking * rot) * self.rotation


def _make_controller(robot, clock, object_pose, extra_cfg=None):
    controller = AutoPickPlaceController(
        AutoPickPlaceConfig({**CONFIG, **(extra_cfg or {})}), use_gripper=True, clock=clock
    )
    controller.set_robot(robot)
    controller.set_canonical_orientation_from_current_pose()
    controller.register_loaded_object(object_pose, object_pose[2])
    return controller


def _run(controller, robot, clock, max_ticks=20000):
    """Step until the sequence completes; returns per-tick (phase, gripper, position)."""
    trace = []
    for _ in range(max_ticks):
        action = controller.get_action()
        if controller.episode_complete:
            return trace
        trace.append((controller.current_phase, action["gripper_position"], robot.position.copy()))
        robot.apply(action)
        clock.t += 1.0 / FPS
    raise AssertionError("sequence did not complete")


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


def test_episode_picks_and_places_smoothly():
    np.random.seed(0)
    clock = FakeClock()
    object_pose = [0.45, -0.3, 0.03, *CANONICAL_ROTVEC]
    robot = FakeRobot([0.22, -0.47, 0.17, *CANONICAL_ROTVEC])
    controller = _make_controller(robot, clock, object_pose)
    controller.reset_step_size()
    trace = _run(controller, robot, clock)

    phases = [phase for phase, _, _ in trace]
    order = [p for i, p in enumerate(phases) if i == 0 or p != phases[i - 1]]
    assert order == [
        Phase.TRANSIT, Phase.DESCEND, Phase.GRASP, Phase.LIFT,
        Phase.TRANSIT, Phase.DESCEND, Phase.RELEASE, Phase.RETREAT, Phase.DONE,
    ], order

    # Suction is on from just before contact until the release.
    grippers = [g for _, g, _ in trace]
    first_on = grippers.index(1.0)
    assert phases[first_on] == Phase.DESCEND
    assert np.linalg.norm(trace[first_on][2] - np.array(object_pose[:3])) < 0.01
    assert phases[grippers.index(0.0, first_on)] == Phase.RELEASE

    # Ended above the sampled place pose.
    place_pose = controller._object_positions[0]
    assert np.allclose(robot.position[:2], place_pose[:2], atol=0.003)

    # Smooth: no tick-to-tick speed jumps beyond a few mm/tick^2.
    positions = np.array([p for _, _, p in trace])
    speeds = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    assert speeds.max() <= controller.config.max_step_size + 1e-9
    assert np.abs(np.diff(speeds)).max() < 0.002

    # Near-object phases are a meaningful share of the episode.
    near = sum(p in (Phase.DESCEND, Phase.GRASP, Phase.LIFT, Phase.RELEASE) for p in phases)
    assert near / len(phases) > 0.25, near / len(phases)


def test_force_stop_skips_rest_of_descent():
    np.random.seed(1)
    clock = FakeClock()
    object_pose = [0.45, -0.3, 0.03, *CANONICAL_ROTVEC]
    # The object is really 1.5cm taller than registered.
    robot = FakeRobot([0.45, -0.3, 0.12, *CANONICAL_ROTVEC], contact_z=0.045)
    controller = _make_controller(robot, clock, object_pose)
    controller.reset_step_size()
    trace = _run(controller, robot, clock)

    grasp_positions = [p for phase, _, p in trace if phase == Phase.GRASP]
    assert grasp_positions, "grasp never ran"
    assert all(p[2] > 0.04 for p in grasp_positions)
    # Suction still came on even though contact cut the descent short.
    assert any(g == 1.0 for phase, g, _ in trace if phase == Phase.GRASP)


def _phase_runs(trace):
    phases = [phase for phase, _, _ in trace]
    return [p for i, p in enumerate(phases) if i == 0 or p != phases[i - 1]]


def test_misaligned_approach_is_corrected():
    np.random.seed(2)
    random.seed(2)
    clock = FakeClock()
    object_pose = [0.45, -0.3, 0.03, *CANONICAL_ROTVEC]
    robot = FakeRobot([0.22, -0.47, 0.17, *CANONICAL_ROTVEC])
    controller = _make_controller(robot, clock, object_pose, {"misalign_prob": 1.0})
    controller.reset_step_size()
    trace = _run(controller, robot, clock)

    assert _phase_runs(trace) == [
        Phase.PERTURB, Phase.CORRECT, Phase.DESCEND, Phase.GRASP, Phase.LIFT,
        Phase.PERTURB, Phase.CORRECT, Phase.DESCEND, Phase.RELEASE, Phase.RETREAT, Phase.DONE,
    ], _phase_runs(trace)
    # The perturbation really is off target, and the grasp happens on it.
    perturb_end = next(i for i, (phase, _, _) in enumerate(trace) if phase == Phase.CORRECT) - 1
    off = np.linalg.norm(trace[perturb_end][2][:2] - np.array(object_pose[:2]))
    assert 0.012 < off < 0.033, off
    grasp = next(p for phase, _, p in trace if phase == Phase.GRASP)
    assert np.linalg.norm(grasp[:2] - np.array(object_pose[:2])) < 0.003


def test_missed_grasp_retries():
    np.random.seed(3)
    random.seed(3)
    clock = FakeClock()
    object_pose = [0.45, -0.3, 0.03, *CANONICAL_ROTVEC]
    robot = FakeRobot([0.22, -0.47, 0.17, *CANONICAL_ROTVEC])
    controller = _make_controller(robot, clock, object_pose, {"miss_grasp_prob": 1.0})
    controller.reset_step_size()
    trace = _run(controller, robot, clock)

    assert _phase_runs(trace)[:4] == [Phase.PERTURB, Phase.CORRECT, Phase.DESCEND, Phase.GRASP]
    # Suction on during the miss, off for the correction, on again for the real grasp.
    grippers = [(phase, g) for phase, g, _ in trace]
    assert any(phase == Phase.PERTURB and g == 1.0 for phase, g in grippers)
    assert all(g == 0.0 for phase, g in grippers if phase == Phase.CORRECT)
    lowest_miss = min((p for phase, _, p in trace if phase == Phase.PERTURB), key=lambda p: p[2])
    assert abs(np.linalg.norm(lowest_miss[:2] - np.array(object_pose[:2])) - 0.05) < 0.003
    assert lowest_miss[2] < object_pose[2] + 0.003


def test_scene_reset_never_perturbs():
    np.random.seed(4)
    random.seed(4)
    clock = FakeClock()
    robot = FakeRobot([0.22, -0.47, 0.17, *CANONICAL_ROTVEC])
    controller = _make_controller(
        robot, clock, [0.45, -0.3, 0.03, *CANONICAL_ROTVEC], {"misalign_prob": 1.0, "miss_grasp_prob": 1.0}
    )
    controller.reset_step_size()
    _run(controller, robot, clock)
    assert controller.build_scene_reset_waypoints()
    trace = _run(controller, robot, clock)
    assert not any(phase in (Phase.PERTURB, Phase.CORRECT) for phase, _, _ in trace)


def test_pause_restarts_current_segment():
    np.random.seed(5)
    random.seed(5)
    clock = FakeClock()
    object_pose = [0.45, -0.3, 0.03, *CANONICAL_ROTVEC]
    robot = FakeRobot([0.22, -0.47, 0.17, *CANONICAL_ROTVEC])
    controller = _make_controller(robot, clock, object_pose)
    controller.reset_step_size()

    # Interrupt the first transit halfway, then resume 5s later.
    for _ in range(30):
        robot.apply(controller.get_action())
        clock.t += 1.0 / FPS
    segment = controller.current_segment
    assert segment.phase == Phase.TRANSIT
    clock.t += 5.0
    controller.get_action()
    assert controller.current_segment is segment
    assert controller._active.start_time == clock.t
    assert np.allclose(controller._active.start_position, robot.position)

    trace = _run(controller, robot, clock)
    grasp = next(p for phase, _, p in trace if phase == Phase.GRASP)
    assert np.linalg.norm(grasp[:2] - np.array(object_pose[:2])) < 0.003


def test_episode_ends_holding_above_last_object():
    np.random.seed(6)
    random.seed(6)
    clock = FakeClock()
    robot = FakeRobot([0.22, -0.47, 0.17, *CANONICAL_ROTVEC])
    controller = _make_controller(robot, clock, [0.45, -0.3, 0.03, *CANONICAL_ROTVEC])
    controller.register_loaded_object([0.47, -0.15, 0.03, *CANONICAL_ROTVEC], 0.03)
    controller.reset_step_size()
    trace = _run(controller, robot, clock)

    done = [p for phase, _, p in trace if phase == Phase.DONE]
    assert abs(len(done) - FPS * controller.config.end_hold_s) <= 2, len(done)
    assert np.ptp(np.array(done), axis=0).max() < 0.001  # holding still
    last_place = controller._last_placements[-1][1]
    assert np.allclose(done[-1][:2], last_place[:2], atol=0.003)
    assert done[-1][2] > last_place[2] + controller.config.approach_height - 0.005
    # The completing tick is labelled with the final segment, not IDLE.
    assert controller.current_phase == Phase.DONE

    # Scene resets don't hold at the end.
    assert controller.build_scene_reset_waypoints()
    assert not any(phase == Phase.DONE for phase, _, _ in _run(controller, robot, clock))


def _run_all():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")


if __name__ == "__main__":
    _run_all()
