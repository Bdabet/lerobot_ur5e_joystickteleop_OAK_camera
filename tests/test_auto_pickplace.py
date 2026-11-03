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
    last_place = controller._object_positions[controller._last_placements[-1]]
    assert np.allclose(done[-1][:2], last_place[:2], atol=0.003)
    assert done[-1][2] > last_place[2] + controller.config.approach_height - 0.005
    # The completing tick is labelled with the final segment, not IDLE.
    assert controller.current_phase == Phase.DONE

    # Scene resets don't hold at the end.
    assert controller.build_scene_reset_waypoints()
    assert not any(phase == Phase.DONE for phase, _, _ in _run(controller, robot, clock))


def test_stack_drops_from_release_height_and_unstacks_by_contact():
    np.random.seed(7)
    random.seed(7)
    clock = FakeClock()
    robot = FakeRobot([0.22, -0.47, 0.17, *CANONICAL_ROTVEC])
    release_z = 0.12
    controller = _make_controller(
        robot,
        clock,
        [0.45, -0.3, 0.03, *CANONICAL_ROTVEC],
        {"place_mode": "stack", "stack_release_z": release_z, "misalign_prob": 1.0},
    )
    controller.register_loaded_object([0.47, -0.15, 0.03, *CANONICAL_ROTVEC], 0.03)
    controller.reset_step_size()
    trace = _run(controller, robot, clock)

    # Both objects are released at the same spot, at stack_release_z - never lower.
    releases = [p for phase, _, p in trace if phase == Phase.RELEASE]
    assert np.ptp(np.array(releases)[:, :2], axis=0).max() < 0.004
    assert all(abs(p[2] - release_z) < 0.003 for p in releases)
    # Once the carried object has cleared stack_release_z, it never goes below
    # it again before the release (misaligned approach included).
    lifted, above = False, False
    for phase, _, p in trace:
        if phase == Phase.LIFT:
            lifted = True
        elif phase == Phase.RELEASE:
            lifted, above = False, False
        above = above or (lifted and p[2] > release_z)
        if above:
            assert p[2] > release_z - 0.003, p

    # Reset: the top object sits 2cm above the table height; the descent
    # starts above stack_release_z with suction already on and stops on contact.
    robot.contact_z = 0.05
    assert controller.build_scene_reset_waypoints()
    reset_trace = _run(controller, robot, clock)
    stack_descent = [(g, p) for phase, g, p in reset_trace if phase == Phase.DESCEND and p[2] > 0.06]
    assert max(p[2] for _, p in stack_descent) > release_z + controller.config.align_height - 0.005
    assert all(g == 1.0 for g, _ in stack_descent)
    grasp = next(p for phase, _, p in reset_trace if phase == Phase.GRASP)
    assert 0.045 < grasp[2] < 0.055, grasp


def test_stack_release_below_table_sets_down_slowly_instead_of_aborting():
    np.random.seed(8)
    random.seed(8)
    clock = FakeClock()
    robot = FakeRobot([0.22, -0.47, 0.17, *CANONICAL_ROTVEC], contact_z=0.029)
    controller = _make_controller(
        robot,
        clock,
        [0.45, -0.3, 0.03, *CANONICAL_ROTVEC],
        {"place_mode": "stack", "stack_release_z": 0.0, "stack_descend_speed": 0.01},
    )
    controller.reset_step_size()
    trace = _run(controller, robot, clock)

    # Placed at table height, never below it, with a slow descent.
    release = next(p for phase, _, p in trace if phase == Phase.RELEASE)
    assert 0.027 < release[2] < 0.035, release
    after_lift = next(i for i, (phase, _, _) in enumerate(trace) if phase == Phase.LIFT)
    place_descent = np.array([p for phase, _, p in trace[after_lift:] if phase == Phase.DESCEND])
    peak_speed = np.linalg.norm(np.diff(place_descent, axis=0), axis=1).max() * FPS
    assert peak_speed < 0.012, peak_speed

    assert controller.build_scene_reset_waypoints()
    reset_trace = _run(controller, robot, clock)
    assert any(phase == Phase.GRASP for phase, _, _ in reset_trace)


def _run_all():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")




def test_manual_scene_reset_replaces_objects():
    clock = FakeClock()
    robot = FakeRobot([0.3, -0.3, 0.2, *CANONICAL_ROTVEC])
    controller = _make_controller(
        robot, clock, [0.45, -0.3, 0.02, *CANONICAL_ROTVEC], {"scene_reset": "manual", "place_mode": "stack",
                                                             "stack_release_z": 0.13, "force_stop_enabled": False}
    )
    controller.reset_step_size()
    controller.clear_loaded_objects()
    assert not controller.build_scene_reset_waypoints()
    controller.register_loaded_object([0.42, -0.2, 0.03, *CANONICAL_ROTVEC], 0.03)
    controller.reset_step_size()
    assert controller._object_heights == [0.03]


def test_place_region_center_radius():
    from scripts.core.auto_pickplace import Region

    region = Region({"center": [0.2, -0.44], "radius": 0.03}, "place_region")
    points = np.array([region.sample_xy() for _ in range(2000)])
    distances = np.linalg.norm(points - np.array([0.2, -0.44]), axis=1)
    assert distances.max() <= 0.03 + 1e-9
    assert distances.max() > 0.025  # actually spreads out to the edge
    assert Region({"center": [0.2, -0.44]}, "place_region").sample_xy() == (0.2, -0.44)


def _run_named(controller, robot, clock, max_ticks=40000):
    """Like _run, but per tick (segment name, position)."""
    trace = []
    for _ in range(max_ticks):
        name = controller.current_segment.name
        action = controller.get_action()
        if controller.episode_complete:
            return trace
        trace.append((name, robot.position.copy()))
        robot.apply(action)
        clock.t += 1.0 / FPS
    raise AssertionError("sequence did not complete")


def test_scene_reset_moves_straight_up_across_down():
    np.random.seed(5)
    random.seed(5)
    clock = FakeClock()
    robot = FakeRobot([0.22, -0.47, 0.17, *CANONICAL_ROTVEC], contact_z=0.03)
    controller = _make_controller(robot, clock, [0.45, -0.3, 0.03, *CANONICAL_ROTVEC])
    controller.reset_step_size()
    _run(controller, robot, clock)
    assert controller.build_scene_reset_waypoints()
    trace = _run_named(controller, robot, clock)
    names = {name for name, _ in trace}
    assert not any("transit" in name for name in names)
    straight_names = [name for name in names if name.startswith(("reset_rise_to", "reset_cross_to", "reset_lower_to"))]
    assert len(straight_names) == 6  # rise/cross/lower, to the pick and to the place
    for segment_name in straight_names:
        segment_positions = np.array([pos for name, pos in trace if name == segment_name])
        moved = segment_positions.max(axis=0) - segment_positions.min(axis=0)
        if "cross_to" in segment_name:
            assert moved[2] < 0.005, (segment_name, moved)  # stays at cruise height
        else:
            assert moved[:2].max() < 0.005, (segment_name, moved)  # purely vertical


def test_descent_waits_for_a_lagging_arm_to_get_above_the_object():
    np.random.seed(6)
    random.seed(6)
    clock = FakeClock()
    # Tracks only 15% of each commanded delta: well behind every profile.
    robot = FakeRobot([0.22, -0.47, 0.17, *CANONICAL_ROTVEC], tracking=0.15, contact_z=0.03)
    controller = _make_controller(robot, clock, [0.45, -0.3, 0.03, *CANONICAL_ROTVEC])
    controller.reset_step_size()
    trace = _run_named(controller, robot, clock)
    for descent in ("descend_pick", "descend_place"):
        first = next(pos for name, pos in trace if name.startswith(descent))
        last = [pos for name, pos in trace if name.startswith(descent)][-1]
        assert np.linalg.norm(first[:2] - last[:2]) < 0.004, (descent, first, last)


def test_region_distance():
    from scripts.core.auto_pickplace import Region

    box = Region({"x_range": [0.1, 0.2], "y_range": [-0.3, -0.4]}, "box")  # reversed y range
    assert box.distance_to((0.15, -0.35)) == 0.0
    assert np.isclose(box.distance_to((0.25, -0.35)), 0.05)
    assert np.isclose(box.distance_to((0.23, -0.26)), 0.05)  # 3-4-5 from the corner
    disc = Region({"center": [0.0, 0.0], "radius": 0.1}, "disc")
    assert disc.distance_to((0.05, 0.0)) == 0.0
    assert np.isclose(disc.distance_to((0.0, 0.25)), 0.15)


def test_scene_reset_keeps_pick_spots_clear_of_place_region():
    np.random.seed(7)
    random.seed(7)
    # pick_region fully contains place_region (center [0.3, -0.3], radius 0.03).
    extra = {
        "pick_region": {"x_range": [0.1, 0.5], "y_range": [-0.5, -0.1]},
        "place_region": {"center": [0.3, -0.3], "radius": 0.03},
        "pick_place_region_clearance": 0.1,
        "pick_min_separation": 0.0,
    }
    clock = FakeClock()
    robot = FakeRobot([0.3, -0.3, 0.2, *CANONICAL_ROTVEC])
    controller = _make_controller(robot, clock, [0.45, -0.3, 0.03, *CANONICAL_ROTVEC], extra)
    for _ in range(100):
        for pose in controller._plan_pick_poses([0.03, 0.03, 0.03]):
            assert np.hypot(pose[0] - 0.3, pose[1] + 0.3) > 0.03 + 0.1


def test_pick_plan_fits_objects_that_only_fit_spread_out():
    np.random.seed(8)
    random.seed(8)
    # 3 objects 0.14m apart in a 0.30m strip: only fits near both ends + middle.
    extra = {
        "pick_region": {"x_range": [0.3, 0.6], "y_range": [-0.2, -0.2]},
        "pick_min_separation": 0.14,
    }
    clock = FakeClock()
    robot = FakeRobot([0.3, -0.3, 0.2, *CANONICAL_ROTVEC])
    controller = _make_controller(robot, clock, [0.45, -0.3, 0.03, *CANONICAL_ROTVEC], extra)
    for _ in range(20):
        xs = sorted(pose[0] for pose in controller._plan_pick_poses([0.03, 0.03, 0.03]))
        assert np.diff(xs).min() >= 0.14, xs


def test_load_from_stack_unstacks_into_pick_region():
    np.random.seed(9)
    random.seed(9)
    extra = {"load_mode": "stack", "stack_plate_pitch": 0.01, "pick_min_separation": 0.1}
    clock = FakeClock()
    robot = FakeRobot([0.3, -0.3, 0.2, *CANONICAL_ROTVEC], contact_z=0.03)
    controller = _make_controller(robot, clock, [0.45, -0.3, 0.03, *CANONICAL_ROTVEC], extra)
    # Cup on the top of a 3-high stack at (0.2, -0.45), top z 0.05.
    top_z = controller.register_loaded_stack([0.2, -0.45, 0.05, *CANONICAL_ROTVEC], 3)
    assert np.allclose(controller._object_heights, 0.03)
    assert controller.build_scene_reset_waypoints(stack_top_z=top_z)
    trace = _run_named(controller, robot, clock)
    stack_descents = {name for name, _ in trace if name.startswith("reset_descend_pick_stack")}
    assert len(stack_descents) == 3
    # Each unstack descent starts above the stack's top.
    for descent in stack_descents:
        first = next(pos for name, pos in trace if name == descent)
        assert first[2] >= top_z, (descent, first)
    region = controller.config.pick_region
    for pose in controller._object_positions.values():
        assert region.distance_to(pose[:2]) == 0.0
        assert np.isclose(pose[2], 0.03)
    controller.reset_step_size()  # the first episode can be built from there


def test_load_top_of_stack_keeps_the_base():
    # 3-high stack, only the top one is picked off: it's registered alone, at
    # the table height (two pitches below the top), not at the top.
    extra = {"load_mode": "stack", "stack_plate_pitch": 0.01}
    clock = FakeClock()
    robot = FakeRobot([0.3, -0.3, 0.2, *CANONICAL_ROTVEC], contact_z=0.05)
    controller = _make_controller(robot, clock, [0.45, -0.3, 0.03, *CANONICAL_ROTVEC], extra)
    top_z = controller.register_loaded_stack([0.2, -0.45, 0.05, *CANONICAL_ROTVEC], 1, stack_size=3)
    assert np.allclose(controller._object_heights, [0.03])
    assert controller.build_scene_reset_waypoints(stack_top_z=top_z)
    trace = _run_named(controller, robot, clock)
    grasp = next(pos for name, pos in trace if name.startswith("reset_grasp"))
    assert np.isclose(grasp[2], 0.05, atol=0.002), grasp  # picked off the top by force contact
    assert np.isclose(controller._object_positions[0][2], 0.03)
    try:
        controller.register_loaded_stack([0.2, -0.45, 0.05, *CANONICAL_ROTVEC], 4, stack_size=3)
    except ValueError:
        pass
    else:
        raise AssertionError("picking more objects than the stack holds should fail")


def test_load_from_stack_centers_place_region_on_first_stack():
    extra = {
        "load_mode": "stack",
        "stack_plate_pitch": 0.01,
        "place_region": {"center": [0.3, -0.3], "radius": 0.015, "exclusion_radius": 0.05},
    }
    clock = FakeClock()
    robot = FakeRobot([0.3, -0.3, 0.2, *CANONICAL_ROTVEC])
    controller = _make_controller(robot, clock, [0.45, -0.3, 0.03, *CANONICAL_ROTVEC], extra)
    controller.register_loaded_stack([0.2, -0.45, 0.05, *CANONICAL_ROTVEC], 3)
    region = controller.config.place_region
    assert np.allclose(region.center, [0.2, -0.45])
    assert region.radius == 0.015 and region.exclusion_radius == 0.05
    # A re-taught stack (scene_reset "manual") keeps the first center.
    controller.register_loaded_stack([0.25, -0.4, 0.05, *CANONICAL_ROTVEC], 3)
    assert np.allclose(region.center, [0.2, -0.45])


def test_place_region_exclusion_radius():
    np.random.seed(10)
    random.seed(10)
    extra = {
        "pick_region": {"x_range": [0.1, 0.5], "y_range": [-0.5, -0.1]},
        "place_region": {"center": [0.3, -0.3], "radius": 0.015, "exclusion_radius": 0.12},
        "pick_min_separation": 0.0,
    }
    clock = FakeClock()
    robot = FakeRobot([0.3, -0.3, 0.2, *CANONICAL_ROTVEC])
    controller = _make_controller(robot, clock, [0.45, -0.3, 0.03, *CANONICAL_ROTVEC], extra)
    for _ in range(100):
        for pose in controller._plan_pick_poses([0.03, 0.03, 0.03]):
            assert np.hypot(pose[0] - 0.3, pose[1] + 0.3) >= 0.12
    try:
        AutoPickPlaceConfig({**CONFIG, "place_region": {"center": [0.3, -0.3], "radius": 0.05, "exclusion_radius": 0.02}})
    except ValueError:
        pass
    else:
        raise AssertionError("exclusion_radius < radius must be rejected")


if __name__ == "__main__":
    _run_all()
