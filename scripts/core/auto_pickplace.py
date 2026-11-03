"""Scripted pick-and-place controller that stands in for a live teleoperator.

Exposes the same duck-typed surface `record_loop`/`run_record` expect from `teleop`
(`connect`, `disconnect`, `get_action`, `reset_step_size`, `set_robot`) plus a
`set_events` hook so it can signal episode completion itself, exactly like a human
pressing the "next episode" key.

Objects are not tied to a single fixed pick/place pose. Instead, `pick_region`
and `place_region` each define an x/y bounding box (plus a fixed orientation)
that positions are sampled from. Before the first episode, `object_loading.py`
drives this same controller through an interactive loading sequence (see
`build_wait_for_object_waypoints`/`build_retreat_after_manual_place_waypoints`/
`register_loaded_object`) that asks the operator how many objects there are;
for each one, the operator hands it to the arm and then manually jogs it down
onto the table wherever they like - the height it's released at becomes that
object's fixed height for the rest of the run.

Motion is a sequence of time-parameterized segments rather than waypoints
chased at a capped speed: each segment follows a minimum-jerk profile (the
bell-shaped speed curve of a human reach) from wherever the arm actually is
when it starts to its end pose, and every tick streams the delta from the
current pose to where the profile says the arm should be next. Free-space
transit is fast and curved (lift, carry, lower); the last few centimeters
above an object are slow, so the frames that matter most for grasping and
releasing are well represented in the data.
"""

import logging
import random
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

logger = logging.getLogger(__name__)


class Region:
    """An x/y area (meters, robot base frame) that pick/place positions are
    sampled uniformly from: either a bounding box (x_range + y_range) or a
    disc (center [x, y] + radius; radius 0 always gives exactly center).
    Orientation is not configured here - every sampled pose uses whatever
    orientation the robot actually has at robot.init_pose (see
    AutoPickPlaceController.set_canonical_orientation_from_current_pose()),
    so it can never drift from init_pose."""

    def __init__(self, cfg: dict[str, Any], name: str):
        self.center: tuple[float, float] | None = None
        self.radius = 0.0
        # place_region only: pick spots within this distance of center are
        # rejected (overrides pick_place_region_clearance); >= radius.
        self.exclusion_radius: float | None = None
        if "center" in cfg:
            if "x_range" in cfg or "y_range" in cfg:
                raise ValueError(f"auto_pick_place.{name}: use either center + radius or x_range + y_range, not both.")
            self.center = tuple(float(v) for v in cfg["center"])
            if len(self.center) != 2:
                raise ValueError(f"auto_pick_place.{name}.center must be [x, y].")
            self.radius = float(cfg.get("radius", 0.0))
            if self.radius < 0.0:
                raise ValueError(f"auto_pick_place.{name}.radius must be >= 0.")
            if cfg.get("exclusion_radius") is not None:
                self.exclusion_radius = float(cfg["exclusion_radius"])
                if self.exclusion_radius < self.radius:
                    raise ValueError(f"auto_pick_place.{name}.exclusion_radius must be >= its radius.")
        elif "x_range" not in cfg or "y_range" not in cfg:
            raise ValueError(f"auto_pick_place.{name} requires x_range and y_range, or center (+ radius).")
        else:
            self.x_range: tuple[float, float] = tuple(float(v) for v in cfg["x_range"])
            self.y_range: tuple[float, float] = tuple(float(v) for v in cfg["y_range"])

    def distance_to(self, xy) -> float:
        """Distance (m) from an x/y point to the nearest point of the region;
        0 if it's inside."""
        if self.center is not None:
            return max(0.0, float(np.hypot(xy[0] - self.center[0], xy[1] - self.center[1])) - self.radius)
        dx = max(min(self.x_range) - xy[0], 0.0, xy[0] - max(self.x_range))
        dy = max(min(self.y_range) - xy[1], 0.0, xy[1] - max(self.y_range))
        return float(np.hypot(dx, dy))

    def sample_xy(self) -> tuple[float, float]:
        if self.center is not None:
            # sqrt keeps the density uniform over the disc's area.
            r = self.radius * np.sqrt(np.random.uniform(0.0, 1.0))
            angle = np.random.uniform(0.0, 2.0 * np.pi)
            return self.center[0] + r * float(np.cos(angle)), self.center[1] + r * float(np.sin(angle))
        return (
            float(np.random.uniform(self.x_range[0], self.x_range[1])),
            float(np.random.uniform(self.y_range[0], self.y_range[1])),
        )


class AutoPickPlaceConfig:
    def __init__(self, cfg: dict[str, Any]):
        self.pick_region = Region(cfg.get("pick_region", {}), "pick_region")
        self.place_region = Region(cfg.get("place_region", {}), "place_region")

        # How long (seconds) the arm holds at robot.init_pose with the gripper
        # turned on while the operator hands it each object during loading.
        self.object_load_hold_s: float = cfg.get("object_load_hold_s", 3.0)
        # Max time (seconds) the operator has to jog each object down onto the
        # table and release it during loading.
        self.manual_place_time_sec: float = cfg.get("manual_place_time_sec", 300)

        # scene_reset "auto" picks every object back into pick_region between
        # episodes. "manual" leaves it to the operator: before each episode
        # (including the first, in place of object loading) they arrange the
        # objects by hand, then jog the cup down onto each one to teach where
        # it is (see object_loading.teach_scene()).
        self.scene_reset: str = cfg.get("scene_reset", "auto")
        if self.scene_reset not in ("auto", "manual"):
            raise ValueError(f"auto_pick_place.scene_reset must be 'auto' or 'manual', got {self.scene_reset!r}.")

        # Transit between objects arcs up to approach_height above the higher
        # of its two endpoints, and ends align_height above the object, where
        # the slow descent takes over.
        self.approach_height: float = cfg.get("approach_height", 0.16)
        self.align_height: float = cfg.get("align_height", 0.04)
        # Peak speeds (m/s) of each segment's minimum-jerk profile. Transit
        # speed is resampled once per episode from transit_speed_range so the
        # data covers a spread of speeds without jitter inside a single move.
        self.transit_speed_range: tuple[float, float] = tuple(cfg.get("transit_speed_range", [0.15, 0.30]))
        self.descend_speed: float = cfg.get("descend_speed", 0.04)
        self.rot_speed: float = cfg.get("rot_speed", 0.6)  # rad/s
        self.min_segment_s: float = cfg.get("min_segment_s", 0.3)
        # Safety clamp on the per-tick commanded delta (m / rad), on top of
        # the robot's own max_pos_speed/max_rot_speed.
        self.max_step_size: float = cfg.get("max_step_size", 0.02)
        self.max_rot_step_size: float = cfg.get("max_rot_step_size", 0.05)
        # How far ahead on the profile each tick aims (s), to make up for the
        # arm lagging one control period behind its last command.
        self.tracking_lead_s: float = cfg.get("tracking_lead_s", 0.05)
        # A segment ends once its profile has finished and the arm is within
        # tolerance of the end pose; settle_timeout_s after the profile ends
        # it moves on regardless (with a warning).
        self.position_tolerance: float = cfg.get("position_tolerance", 0.003)
        self.rotation_tolerance: float = cfg.get("rotation_tolerance", 0.02)
        self.settle_timeout_s: float = cfg.get("settle_timeout_s", 1.5)
        # The move that ends above an object (right before the descent onto
        # it) waits up to this long instead, so a lagging arm reaches the spot
        # above the object and the descent goes straight down, instead of
        # coming in at an angle and touching down (force stop) off target.
        self.align_settle_timeout_s: float = cfg.get("align_settle_timeout_s", 5.0)
        # If get_action() hasn't been called for longer than this (s) - the
        # loop driving it was interrupted, e.g. Esc mid scene reset, and a
        # later loop picks the same segments back up - the current segment
        # restarts from wherever the arm is instead of treating its profile
        # as long overdue and skipping the rest of the move.
        self.resume_gap_s: float = cfg.get("resume_gap_s", 0.5)

        # Suction is switched on suction_lead_m before reaching the object
        # (like a person closing the hand on the way in), then held
        # grasp_seal_s to let the seal form. release_s is the hold after
        # switching suction off at the place point.
        self.suction_lead_m: float = cfg.get("suction_lead_m", 0.005)
        self.grasp_seal_s: float = cfg.get("grasp_seal_s", 0.25)
        self.release_s: float = cfg.get("release_s", 0.25)
        # Hold still for this long (s) after the last object's retreat, right
        # above it, so every episode ends with a recognizable "done -> stay
        # put" pattern instead of stopping mid-motion. Episodes only, not
        # scene resets. 0 disables it.
        self.end_hold_s: float = cfg.get("end_hold_s", 1.0)

        # Stop the downward pick/place move early on TCP force contact instead
        # of servoing all the way to the (possibly optimistic) target depth.
        self.force_stop_enabled: bool = cfg.get("force_stop_enabled", False)
        self.contact_force_threshold: float = cfg.get("contact_force_threshold", 15.0)
        # Right after grasp, the TCP first lifts this far straight up (slowly)
        # from wherever it actually is before the transit arc starts.
        self.force_stop_clearance_height: float = cfg.get("force_stop_clearance_height", 0.02)

        # Recovery demonstrations, so the policy sees how to get back on track
        # from the kinds of mistakes it will make. Per object and per
        # approach (pick or place), with probability misalign_prob the
        # transit ends misalign_range meters off to the side, starts down,
        # then corrects back over the target; at the pick only, with
        # probability miss_grasp_prob it instead goes all the way down
        # miss_offset beside the object (cup in the air, nothing to grab),
        # switches suction on, lifts, switches it off and goes back to the
        # real object. miss_offset must clear the object's half-width plus
        # the cup radius. The wrong part is recorded as Phase.PERTURB, the
        # recovery as Phase.CORRECT. Scene resets never use these.
        self.misalign_prob: float = cfg.get("misalign_prob", 0.0)
        self.misalign_range: tuple[float, float] = tuple(cfg.get("misalign_range", [0.015, 0.03]))
        self.miss_grasp_prob: float = cfg.get("miss_grasp_prob", 0.0)
        self.miss_offset: float = cfg.get("miss_offset", 0.05)
        self.correct_speed: float = cfg.get("correct_speed", 0.06)

        self.shuffle_pick_order: bool = cfg.get("shuffle_pick_order", True)
        self.pick_min_separation: float = cfg.get("pick_min_separation", 0.05)
        # Pick spots (scene reset) closer than this to place_region - measured
        # from the object's center to the region's nearest point, 0 = only
        # spots inside it - are rejected, so pick_region may overlap
        # place_region without objects being put back where the next episode
        # places (or stacks) them. null disables the check.
        self.pick_place_region_clearance: float | None = cfg.get("pick_place_region_clearance", 0.0)
        self.place_min_separation: float = cfg.get("place_min_separation", 0.05)
        self.place_sampling_max_attempts: int = cfg.get("place_sampling_max_attempts", 200)

        # place_mode "scatter" samples a separate spot in place_region per
        # object. "stack" samples one spot per episode and stacks every object
        # there by dropping it: the place motion stops at the fixed height
        # stack_release_z (base frame, TCP z) above the stack and releases, so
        # it never presses into the stack. stack_release_z must clear the
        # tallest stack plus one object. Since the plates' resting heights are
        # then unknown, the scene reset unstacks top-down by approaching
        # from stack_release_z and descending at stack_descend_speed (suction
        # already on) until force contact with the top object.
        self.place_mode: str = cfg.get("place_mode", "scatter")
        if self.place_mode not in ("scatter", "stack"):
            raise ValueError(f"auto_pick_place.place_mode must be 'scatter' or 'stack', got {self.place_mode!r}.")
        self.stack_release_z: float | None = cfg.get("stack_release_z")
        # load_mode "per_object" hands over / teaches each object separately.
        # "stack": the operator stacks all objects anywhere and only jogs the
        # cup onto the top one; the robot then unstacks them top-down (force
        # contact) into pick_region. Each object's table height is derived as
        # top - (n - 1) * stack_plate_pitch (the height each object adds to a
        # stack); too large is safe, the force-stop descent catches the table.
        self.load_mode: str = cfg.get("load_mode", "per_object")
        if self.load_mode not in ("per_object", "stack"):
            raise ValueError(f"auto_pick_place.load_mode must be 'per_object' or 'stack', got {self.load_mode!r}.")
        self.stack_plate_pitch: float | None = cfg.get("stack_plate_pitch")
        if self.load_mode == "stack":
            if self.stack_plate_pitch is None or self.stack_plate_pitch < 0.0:
                raise ValueError("auto_pick_place.load_mode 'stack' requires stack_plate_pitch >= 0.")
            if not cfg.get("force_stop_enabled", False):
                raise ValueError(
                    "auto_pick_place.load_mode 'stack' requires force_stop_enabled: unstacking finds each "
                    "object by force contact."
                )
        self.stack_descend_speed: float = cfg.get("stack_descend_speed", self.descend_speed)
        if self.place_mode == "stack":
            if self.stack_release_z is None:
                raise ValueError("auto_pick_place.place_mode 'stack' requires stack_release_z.")
            if self.scene_reset == "auto" and not self.force_stop_enabled:
                raise ValueError(
                    "auto_pick_place.place_mode 'stack' with scene_reset 'auto' requires force_stop_enabled: "
                    "the scene reset finds each stacked object by force contact."
                )


class Phase(IntEnum):
    """Per-frame label of what the scripted controller is doing, recorded into
    the dataset (feature "phase") so training can reweight or exclude frames
    by phase. PERTURB marks deliberately wrong motion (a misaligned approach,
    a missed grasp) that should not be imitated; CORRECT marks the recovery
    from it, which should."""

    IDLE = 0
    TRANSIT = 1
    DESCEND = 2
    GRASP = 3
    LIFT = 4
    RELEASE = 5
    RETREAT = 6
    PERTURB = 7
    CORRECT = 8
    DONE = 9


@dataclass
class Segment:
    """One time-parameterized move. The start pose is captured live from the
    robot when the segment begins (so a move that follows an early force-stop
    contact starts from wherever contact actually happened); the end pose is
    either absolute (position/rotvec) or relative to that start (offset), and
    a segment with neither is a hold in place for duration_s. A NaN in
    position keeps that coordinate where the arm is when the segment starts
    (e.g. [nan, nan, z] goes straight up or down to z)."""

    name: str
    phase: Phase
    gripper: float | None  # set at segment start; None keeps the current value
    position: np.ndarray | None = None
    rotvec: np.ndarray | None = None
    offset: np.ndarray | None = None
    speed: float | None = None  # peak linear speed (m/s) of the min-jerk profile
    duration_s: float | None = None  # fixed duration, for holds
    cruise_z: float | None = None  # if set, follow a cubic Bezier lifted to about this height
    force_stop: bool = False  # TCP-force contact ends this segment (and any directly following force_stop ones)
    settle_timeout_s: float | None = None  # overrides config.settle_timeout_s


# Peak/average velocity ratio of the minimum-jerk profile: a move of length d
# at peak speed v takes MIN_JERK_PEAK_RATIO * d / v seconds.
MIN_JERK_PEAK_RATIO = 1.875


@dataclass
class _ActiveSegment:
    """Runtime state of the segment being executed, fixed when it begins."""

    start_time: float
    duration_s: float
    start_position: np.ndarray
    end_position: np.ndarray
    slerp: Slerp
    control_points: np.ndarray | None  # cubic Bezier path, or None for a straight line

    def reference(self, elapsed: float) -> tuple[np.ndarray, R]:
        s = _min_jerk(elapsed / self.duration_s) if self.duration_s > 0.0 else 1.0
        if self.control_points is not None:
            position = _cubic_bezier(self.control_points, s)
        else:
            position = self.start_position + s * (self.end_position - self.start_position)
        return position, self.slerp([s])[0]


def _sample_region_position(region: Region, height: float, canonical_rotvec: np.ndarray) -> list[float]:
    """A random pose [x, y, z, rx, ry, rz] (rotvec) with x/y uniform over the
    region, z fixed at height, and the canonical orientation (whatever the
    robot's actual orientation is at robot.init_pose)."""
    x, y = region.sample_xy()
    return [x, y, float(height), *canonical_rotvec.tolist()]


class AutoPickPlaceController:
    """Scripted pick-and-place teleop stand-in.

    Objects are loaded interactively (see object_loading.py) before the first
    episode: each object gets an index, a fixed height, and an initial position
    sampled inside pick_region. Every episode then picks each loaded object up
    from wherever it currently sits and places it at a fresh random spot inside
    place_region; per object the segments are transit -> descend -> grasp ->
    lift -> transit -> descend -> release -> retreat (see _pick_place_block).

    Between episodes, build_scene_reset_waypoints() builds a mirror-image
    sequence that picks each object back up from where it was just placed and
    sets it down at a fresh randomized spot inside pick_region, so the scene is
    ready for the next episode without human intervention.
    """

    name = "auto_pick_place"

    def __init__(self, config: AutoPickPlaceConfig, use_gripper: bool, clock=time.perf_counter):
        self.config = config
        self.use_gripper = use_gripper
        self.robot = None
        self.events: dict | None = None
        self._connected = False
        self._clock = clock

        self._segments: list[Segment] = []
        self._current_idx = 0
        self._active: _ActiveSegment | None = None
        self._last_tick_time: float | None = None
        self._episode_complete = False
        self._last_gripper = 1.0
        # Per-object fixed height, set once during loading (index == object id).
        self._object_heights: list[float] = []
        # object index -> its current pose [x,y,z,rx,ry,rz] (rotvec), updated
        # whenever a pick/place block moves it (loading, episode placement, or
        # scene reset), so the next episode/step picks it up from where it
        # really is instead of a fresh, independently-sampled guess.
        self._object_positions: dict[int, list[float]] = {}
        # Object indices in placement order from the most recently built
        # episode, used by build_scene_reset_waypoints() to bring objects back
        # from their goal position (read from _object_positions) to a fresh
        # spot in pick_region.
        self._last_placements: list[int] = []
        # Set once register_loaded_stack() has moved place_region's center
        # onto the first hand-built stack.
        self._place_center_from_stack = False
        # The single orientation (rotvec) used for every pick/place/loaded-
        # object pose, captured live from the robot at robot.init_pose (see
        # set_canonical_orientation_from_current_pose()) so it's always
        # exactly what init_pose actually is, never a separately configured
        # value that can drift from it.
        self._canonical_rotvec: np.ndarray | None = None

    def set_canonical_orientation_from_current_pose(self) -> None:
        """Capture the robot's current EE orientation as the fixed orientation
        used for every pick_region/place_region sample and every loaded
        object. Call this once, right after the robot has been moved to
        robot.init_pose, before object loading or any episode starts."""
        current_pose = self.robot.get_ee_pose()
        self._canonical_rotvec = np.array(current_pose[3:], dtype=float)

    # ======= duck-typed teleop interface =======

    def set_robot(self, robot) -> None:
        self.robot = robot

    def set_events(self, events: dict) -> None:
        self.events = events

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def current_segment(self) -> "Segment | None":
        if self._episode_complete or not self._segments:
            return None
        return self._segments[self._current_idx]

    @property
    def current_phase(self) -> int:
        """Phase of the segment that produced the most recent action - on the
        tick that completes the sequence, that's still the final segment."""
        if not self._segments:
            return int(Phase.IDLE)
        return int(self._segments[self._current_idx].phase)

    @property
    def episode_complete(self) -> bool:
        return self._episode_complete

    def reset_step_size(self) -> None:
        """Per-episode reset hook (called once before each episode starts)."""
        self._set_segments(self._build_episode_segments())
        self._last_gripper = 0.0

    def build_scene_reset_waypoints(self, stack_top_z: float | None = None) -> bool:
        """Build a segment sequence that returns each object from where the last
        episode placed it to a freshly randomized position inside pick_region.
        Objects on a stack (place_mode "stack", or stack_top_z given, the
        known z of its top) are taken off top-down by force contact.

        Intended to be run through record_loop with dataset=None (not recorded as
        a demonstration) between episodes, so the scene is ready for the next
        episode's pick-and-place without human intervention. Returns False (no
        segments built) if there is no completed episode's placements to reset.
        """
        if not self._last_placements:
            return False

        segments: list[Segment] = []
        # Not recorded, so no need for speed variety - just go briskly.
        transit_speed = self.config.transit_speed_range[1]
        # A stack has to come apart top-down, i.e. in reverse placement order.
        reset_order = self._last_placements[::-1] if self.config.place_mode == "stack" else self._last_placements
        if stack_top_z is None and self.config.place_mode == "stack":
            stack_top_z = self.config.stack_release_z
        reset_targets = self._plan_pick_poses([self._object_heights[object_idx] for object_idx in reset_order])
        for i, (object_idx, reset_target) in enumerate(zip(reset_order, reset_targets, strict=True), start=1):
            placed_pose = self._object_positions[object_idx]
            self._object_positions[object_idx] = reset_target
            segments.extend(
                self._pick_place_block(
                    i,
                    placed_pose,
                    reset_target,
                    transit_speed,
                    name_prefix="reset_",
                    allow_recovery=False,
                    straight=True,
                    source_stack_top_z=stack_top_z,
                )
            )
            logger.info(
                "====== [AUTO] Reset %d/%d: from=%s to=%s ======",
                i,
                len(self._last_placements),
                [round(v, 4) for v in placed_pose],
                [round(v, 4) for v in reset_target],
            )

        self._set_segments(segments)
        self._last_gripper = 0.0
        self._last_placements = []
        return True

    def get_action(self) -> dict[str, float]:
        if self.robot is None:
            raise ValueError("AutoPickPlaceController requires set_robot() before get_action().")

        if self._episode_complete:
            self._signal_done()
            return self._zero_action()

        current_pose = self.robot.get_ee_pose()
        current_position = np.array(current_pose[:3], dtype=float)
        current_rotation = R.from_rotvec(current_pose[3:])
        now = self._clock()
        if (
            self._active is not None
            and self._last_tick_time is not None
            and now - self._last_tick_time > self.config.resume_gap_s
        ):
            logger.info(
                "====== [AUTO] Resuming segment '%s' after a %.1fs pause ======",
                self._segments[self._current_idx].name,
                now - self._last_tick_time,
            )
            self._active = None
        self._last_tick_time = now

        segment = self._segments[self._current_idx]
        if (
            self._active is not None
            and segment.force_stop
            and self.config.force_stop_enabled
            and self._contact_detected()
        ):
            # Skip the rest of the descent; whatever follows starts from
            # here, since segment start poses are captured live.
            while self._current_idx < len(self._segments) - 1 and self._segments[self._current_idx].force_stop:
                self._current_idx += 1
            self._active = None
            segment = self._segments[self._current_idx]

        while True:
            if self._active is None:
                self._begin_segment(segment, current_position, current_rotation, now)

            elapsed = now - self._active.start_time
            if elapsed >= self._active.duration_s:
                pos_error = float(np.linalg.norm(self._active.end_position - current_position))
                rot_error = float(
                    np.linalg.norm((self._active.slerp([1.0])[0] * current_rotation.inv()).as_rotvec())
                )
                settled = pos_error <= self.config.position_tolerance and rot_error <= self.config.rotation_tolerance
                settle_timeout_s = (
                    self.config.settle_timeout_s if segment.settle_timeout_s is None else segment.settle_timeout_s
                )
                timed_out = elapsed >= self._active.duration_s + settle_timeout_s
                if settled or timed_out:
                    if not settled:
                        logger.warning(
                            "====== [AUTO] Segment '%s' did not settle (pos err %.4fm, rot err %.4frad); "
                            "moving on ======",
                            segment.name,
                            pos_error,
                            rot_error,
                        )
                    if self._current_idx == len(self._segments) - 1:
                        self._episode_complete = True
                        self._signal_done()
                        return self._zero_action()
                    self._current_idx += 1
                    self._active = None
                    segment = self._segments[self._current_idx]
                    continue
            break

        reference_position, reference_rotation = self._active.reference(elapsed + self.config.tracking_lead_s)
        delta_position = _clip_to_norm(reference_position - current_position, self.config.max_step_size)
        # Axis-angle (rotvec) error, not an "xyz" Euler decomposition: for a
        # large correction, decomposing into separate per-axis Euler angles
        # can distort badly and inject rotation around an axis that isn't
        # actually part of the true, minimal correction.
        delta_rotation = _clip_to_norm(
            (reference_rotation * current_rotation.inv()).as_rotvec(), self.config.max_rot_step_size
        )

        action = {
            "delta_x": float(delta_position[0]),
            "delta_y": float(delta_position[1]),
            "delta_z": float(delta_position[2]),
            "delta_rx": float(delta_rotation[0]),
            "delta_ry": float(delta_rotation[1]),
            "delta_rz": float(delta_rotation[2]),
        }
        if self.use_gripper:
            action["gripper_position"] = self._last_gripper
        return action

    # ======= object loading (see scripts/core/object_loading.py) =======

    def build_wait_for_object_waypoints(self, hold_s: float) -> None:
        """A single segment that holds the robot's current pose with the
        gripper turned on (suction active) for hold_s seconds, so the operator
        can hand the arm an object to load.

        Call this only once the robot is already stationary at init_pose (via
        a direct robot.reset_to_init_pose() call, not through this
        controller). Holding the live start pose (rather than a target rebuilt
        from robot.init_pose) matters because init_pose is specified in the
        TCP frame while get_ee_pose() - what every segment here is tracked
        against - returns the pose with the tool's TCP offset removed.
        """
        self._set_segments([Segment("load_wait", Phase.IDLE, gripper=1.0, duration_s=hold_s)])
        self._last_gripper = 1.0

    def canonicalize_pick_pose(self, placed_pose: list[float]) -> list[float]:
        """Replace a manually-placed pose's orientation with the canonical
        orientation (see set_canonical_orientation_from_current_pose()),
        keeping only its x/y/z.

        Manual jogging can leave the wrist rotated by whatever incidental
        amount the operator happened to approach at, even if they never meant
        to reorient it. Every later pick (every automatic scene reset)
        samples a fresh position but always uses the canonical orientation -
        so if a loaded object's registered pose kept the raw jogged rotation
        instead, the very first pick of that object would need a real (and
        often large) reorientation to align with it that no other pick ever
        needs. Canonicalizing here makes every pick, from the first one on,
        use the same orientation - which is exactly the orientation the robot
        already has at robot.init_pose.
        """
        if self._canonical_rotvec is None:
            raise ValueError(
                "AutoPickPlaceController has no canonical orientation. "
                "set_canonical_orientation_from_current_pose() must run first."
            )
        position = np.array(placed_pose[:3], dtype=float)
        return [*position.tolist(), *self._canonical_rotvec.tolist()]

    def build_retreat_after_manual_place_waypoints(self, placed_pose: list[float]) -> None:
        """Explicitly release the gripper at a manually-placed object's pose
        (rotvec), then lift straight up, so the object is guaranteed to be let
        go before the arm moves away - not left dependent on the operator
        having manually toggled the gripper open during jogging."""
        position = np.array(placed_pose[:3], dtype=float)
        rotvec = np.array(placed_pose[3:], dtype=float)
        self._set_segments(
            [
                Segment(
                    "load_manual_align",
                    Phase.IDLE,
                    gripper=None,
                    position=position,
                    rotvec=rotvec,
                    speed=self.config.descend_speed,
                ),
                Segment("load_manual_release", Phase.RELEASE, gripper=0.0, duration_s=self.config.release_s),
                Segment(
                    "load_manual_retreat",
                    Phase.RETREAT,
                    gripper=None,
                    offset=np.array([0.0, 0.0, self.config.approach_height]),
                    speed=self.config.transit_speed_range[0],
                ),
            ]
        )

    def register_loaded_object(self, placed_pose: list[float], height: float) -> None:
        """Record a newly-loaded object's fixed height and current position so
        episodes can pick it up. Called once per object, after the operator
        has manually placed it and released the gripper."""
        object_idx = len(self._object_heights)
        self._object_heights.append(height)
        self._object_positions[object_idx] = placed_pose

    def register_loaded_stack(self, top_pose: list[float], num_objects: int, stack_size: int | None = None) -> float:
        """load_mode "stack": register the top num_objects of a stack_size-high
        stack (default: all of it) at top_pose (the cup on the top object,
        canonicalized), as if an episode had just stacked them there, so
        build_scene_reset_waypoints(stack_top_z) unstacks them into
        pick_region. The other stack_size - num_objects stay as the base the
        episodes stack back onto. Returns the stack's top z.

        The first call also moves place_region's center to the stack's x/y
        (radius and exclusion_radius unchanged), so episodes rebuild the stack
        where the operator first built it. Later calls (scene_reset "manual")
        keep that center."""
        place_region = self.config.place_region
        if not self._place_center_from_stack:
            if place_region.center is None:
                logger.warning(
                    "====== [STACK] place_region is a box (x_range/y_range), not center + radius; "
                    "keeping it instead of centering it on the stack ======"
                )
            else:
                place_region.center = (float(top_pose[0]), float(top_pose[1]))
                logger.info(
                    "====== [STACK] place_region center set to the stack position %s (radius %.4f) ======",
                    [round(v, 4) for v in place_region.center],
                    place_region.radius,
                )
            self._place_center_from_stack = True

        if stack_size is None:
            stack_size = num_objects
        if not 0 < num_objects <= stack_size:
            raise ValueError(f"Can't pick {num_objects} object(s) off a stack of {stack_size}.")
        top_z = float(top_pose[2])
        # Every object is picked off into pick_region, i.e. ends up on the
        # table: stack_size - 1 pitches below the top, base or no base.
        table_z = top_z - (stack_size - 1) * self.config.stack_plate_pitch
        self.clear_loaded_objects()
        for _ in range(num_objects):
            self.register_loaded_object([top_pose[0], top_pose[1], table_z, *top_pose[3:]], table_z)
        self._last_placements = list(range(num_objects))
        return top_z

    def clear_loaded_objects(self) -> None:
        """Forget every registered object, so a manually prepared scene can be
        registered from scratch (scene_reset "manual")."""
        self._object_heights = []
        self._object_positions = {}
        self._last_placements = []

    # ======= internals =======

    def _set_segments(self, segments: list[Segment]) -> None:
        self._segments = segments
        self._current_idx = 0
        self._last_tick_time = None
        self._active = None
        self._episode_complete = False

    def _begin_segment(
        self, segment: Segment, current_position: np.ndarray, current_rotation: R, now: float
    ) -> None:
        if segment.gripper is not None:
            self._last_gripper = segment.gripper

        start_position = current_position.copy()
        if segment.position is not None:
            end_position = np.array(segment.position, dtype=float)
            end_position = np.where(np.isnan(end_position), start_position, end_position)
        elif segment.offset is not None:
            end_position = start_position + segment.offset
        else:
            end_position = start_position.copy()
        end_rotation = R.from_rotvec(segment.rotvec) if segment.rotvec is not None else current_rotation

        control_points = None
        if segment.cruise_z is not None:
            control_points = _transit_control_points(start_position, end_position, segment.cruise_z)
            length = _bezier_length(control_points)
        else:
            length = float(np.linalg.norm(end_position - start_position))
        angle = float(np.linalg.norm((end_rotation * current_rotation.inv()).as_rotvec()))

        if segment.duration_s is not None:
            duration_s = segment.duration_s
        else:
            duration_s = max(
                MIN_JERK_PEAK_RATIO * length / segment.speed,
                MIN_JERK_PEAK_RATIO * angle / self.config.rot_speed,
                self.config.min_segment_s,
            )

        self._active = _ActiveSegment(
            start_time=now,
            duration_s=duration_s,
            start_position=start_position,
            end_position=end_position,
            slerp=Slerp([0.0, 1.0], R.concatenate([current_rotation, end_rotation])),
            control_points=control_points,
        )

    def _contact_detected(self) -> bool:
        """True if live TCP force exceeds contact_force_threshold, meaning the
        gripper has touched an object/surface during a force_stop segment."""
        tcp_force = self.robot.get_tcp_force()
        force_norm = float(np.linalg.norm(tcp_force[:3]))
        if force_norm >= self.config.contact_force_threshold:
            logger.info(
                "====== [AUTO] Contact detected at segment '%s': force=%.2fN >= threshold=%.2fN ======",
                self._segments[self._current_idx].name,
                force_norm,
                self.config.contact_force_threshold,
            )
            return True
        return False

    def _signal_done(self) -> None:
        if self.events is not None:
            self.events["exit_early"] = True

    def _zero_action(self) -> dict[str, float]:
        action = {key: 0.0 for key in ("delta_x", "delta_y", "delta_z", "delta_rx", "delta_ry", "delta_rz")}
        if self.use_gripper:
            action["gripper_position"] = self._last_gripper
        return action

    def _build_episode_segments(self) -> list[Segment]:
        if not self._object_heights:
            raise ValueError(
                "AutoPickPlaceController has no loaded objects. "
                "load_objects_into_regions() must run before the first episode."
            )

        object_order = list(range(len(self._object_heights)))
        if self.config.shuffle_pick_order:
            random.shuffle(object_order)

        transit_speed = float(np.random.uniform(*self.config.transit_speed_range))
        logger.info("====== [AUTO] Episode transit speed: %.3f m/s ======", transit_speed)

        stacking = self.config.place_mode == "stack"
        # Stacking: one spot for the whole episode, every object goes on top.
        stack_xy = None
        if stacking:
            stack_xy = _sample_region_position(self.config.place_region, 0.0, self._canonical_rotvec)[:2]
            logger.info("====== [AUTO] Episode stack position: %s ======", [round(v, 4) for v in stack_xy])

        placed_positions: list[np.ndarray] = []
        placements: list[int] = []
        segments: list[Segment] = []

        for i, object_idx in enumerate(object_order, start=1):
            # Objects sit exactly where loading (or the previous episode's
            # reset) left them - pick them up from there.
            pick_pose = self._object_positions[object_idx]

            height = self._object_heights[object_idx]
            place_descend_speed = None
            if stacking:
                # Released (dropped) at stack_release_z, above the stack.
                release_z = self.config.stack_release_z
                if release_z <= height:
                    # Misconfigured (at or below the table): don't abort, set
                    # it down instead - aim for its table height and descend
                    # slowly, so the force stop catches the stack (or table)
                    # gently and it's released wherever it touches.
                    logger.warning(
                        "====== [AUTO] stack_release_z=%.4f is not above object %d's table height %.4f; "
                        "setting it down with a slow force-stop descent instead of dropping it ======",
                        release_z,
                        object_idx,
                        height,
                    )
                    release_z = height
                    place_descend_speed = self.config.stack_descend_speed
                place_sampled = [*stack_xy, release_z, *self._canonical_rotvec.tolist()]
            else:
                place_sampled = self._sample_region_pose(
                    self.config.place_region, height, placed_positions, self.config.place_min_separation
                )
            placed_positions.append(np.array(place_sampled[:3], dtype=float))
            placements.append(object_idx)
            # Where it lands on the stack isn't known; its own table height is
            # the lowest it can be, which is what the reset's unstacking
            # descent goes down towards.
            if stacking:
                self._object_positions[object_idx] = [*place_sampled[:2], height, *place_sampled[3:]]
            else:
                self._object_positions[object_idx] = place_sampled

            segments.extend(
                self._pick_place_block(
                    i, pick_pose, place_sampled, transit_speed, place_descend_speed=place_descend_speed
                )
            )
            logger.info(
                "====== [AUTO] Object %d/%d: pick=%s place=%s ======",
                i,
                len(object_order),
                [round(v, 4) for v in pick_pose],
                [round(v, 4) for v in place_sampled],
            )

        if self.config.end_hold_s > 0.0:
            segments.append(Segment("done", Phase.DONE, gripper=None, duration_s=self.config.end_hold_s))

        self._last_placements = placements
        return segments

    def _pick_place_block(
        self,
        index: int,
        source_pose: list[float],
        target_pose: list[float],
        transit_speed: float,
        name_prefix: str = "",
        allow_recovery: bool = True,
        source_stack_top_z: float | None = None,
        place_descend_speed: float | None = None,
        straight: bool = False,
    ) -> list[Segment]:
        """Segments moving one object from source_pose to target_pose, starting
        from wherever the arm is: arc over to just above the object, descend
        slowly (suction on just before contact), seal, lift clear, arc over to
        just above the target, descend, release, retreat up. With
        allow_recovery, either approach may first go wrong and recover (see
        AutoPickPlaceConfig.misalign_prob/miss_grasp_prob).

        With source_stack_top_z, the source sits somewhere on a stack whose
        top is at most that high (source z is only its lowest possible height):
        the pick approach ends align_height above it, and the descent goes
        down from there at stack_descend_speed with suction already on until
        force contact with the top object. place_descend_speed overrides the
        place descent's speed (default descend_speed). With straight (scene
        resets: not recorded, so nothing needs to look human), each transit is
        straight up, across and down instead of an arc."""
        source_position = np.array(source_pose[:3], dtype=float)
        source_rotvec = np.array(source_pose[3:], dtype=float)
        target_position = np.array(target_pose[:3], dtype=float)
        target_rotvec = np.array(target_pose[3:], dtype=float)
        source_stacked = source_stack_top_z is not None
        if source_stacked:
            pick_approach_position = source_position.copy()
            # Never below the object's own table height, even if
            # stack_release_z is misconfigured below it.
            pick_approach_position[2] = max(source_stack_top_z, source_position[2])
        else:
            pick_approach_position = source_position
        cruise_z = max(pick_approach_position[2], target_position[2]) + self.config.approach_height

        def name(step: str) -> str:
            return f"{name_prefix}{step}_{index}"

        if source_stacked:
            pick_descent = [
                Segment(
                    name("descend_pick_stack"),
                    Phase.DESCEND,
                    gripper=1.0,
                    position=source_position,
                    rotvec=source_rotvec,
                    speed=self.config.stack_descend_speed,
                    force_stop=True,
                )
            ]
        else:
            pick_descent = self._descend_segments(name, source_position, source_rotvec, suction=True)

        segments = [
            *self._approach_segments(
                name,
                pick_approach_position,
                source_rotvec,
                transit_speed,
                cruise_z,
                pick=True,
                allow_recovery=allow_recovery,
                straight=straight,
            ),
            *pick_descent,
            Segment(name("grasp"), Phase.GRASP, gripper=1.0, duration_s=self.config.grasp_seal_s),
            Segment(
                name("lift"),
                Phase.LIFT,
                gripper=None,
                offset=np.array([0.0, 0.0, self.config.force_stop_clearance_height]),
                speed=self.config.descend_speed,
            ),
            *self._approach_segments(
                name,
                target_position,
                target_rotvec,
                transit_speed,
                cruise_z,
                pick=False,
                allow_recovery=allow_recovery,
                straight=straight,
            ),
            *self._descend_segments(name, target_position, target_rotvec, suction=False, speed=place_descend_speed),
            Segment(name("release"), Phase.RELEASE, gripper=0.0, duration_s=self.config.release_s),
            Segment(
                name("retreat"),
                Phase.RETREAT,
                gripper=None,
                offset=np.array([0.0, 0.0, self.config.approach_height]),
                speed=transit_speed,
            ),
        ]
        return segments

    def _approach_segments(
        self,
        name,
        position: np.ndarray,
        rotvec: np.ndarray,
        transit_speed: float,
        cruise_z: float,
        pick: bool,
        allow_recovery: bool,
        straight: bool = False,
    ) -> list[Segment]:
        """Arc over (or with straight, go up, across and down) to align_height
        above position, possibly via a recovered mistake. The normal descent
        onto position follows these segments; the last one waits up to
        align_settle_timeout_s to actually get there."""
        segments = self._approach_segments_unaligned(
            name, position, rotvec, transit_speed, cruise_z, pick, allow_recovery, straight
        )
        segments[-1].settle_timeout_s = self.config.align_settle_timeout_s
        return segments

    def _approach_segments_unaligned(
        self,
        name,
        position: np.ndarray,
        rotvec: np.ndarray,
        transit_speed: float,
        cruise_z: float,
        pick: bool,
        allow_recovery: bool,
        straight: bool,
    ) -> list[Segment]:
        above = np.array([0.0, 0.0, self.config.align_height])
        step = "pick" if pick else "place"
        # Open the gripper on the way to a pick; keep holding on the way to a place.
        gripper = 0.0 if pick else None

        def transit(phase: Phase, offset: np.ndarray) -> Segment:
            return Segment(
                name(f"transit_to_{step}"),
                phase,
                gripper=gripper,
                position=position + offset + above,
                rotvec=rotvec,
                speed=transit_speed,
                cruise_z=cruise_z,
            )

        def correct_to_above(height: float) -> Segment:
            return Segment(
                name(f"correct_{step}"),
                Phase.CORRECT,
                gripper=gripper,
                position=position + np.array([0.0, 0.0, height]),
                rotvec=rotvec,
                speed=self.config.correct_speed,
            )

        if allow_recovery and pick and random.random() < self.config.miss_grasp_prob:
            miss = _sample_horizontal_offset((self.config.miss_offset, self.config.miss_offset))
            logger.info("====== [AUTO] Recovery: missed grasp %s off target ======", np.round(miss[:2], 3))
            return [
                transit(Phase.PERTURB, miss),
                *self._descend_segments(name, position + miss, rotvec, suction=True, phase=Phase.PERTURB),
                Segment(name("missed_grasp"), Phase.PERTURB, gripper=1.0, duration_s=self.config.grasp_seal_s),
                Segment(
                    name("missed_lift"),
                    Phase.PERTURB,
                    gripper=None,
                    position=position + miss + above,
                    rotvec=rotvec,
                    speed=self.config.descend_speed,
                ),
                correct_to_above(self.config.align_height),
            ]

        if allow_recovery and random.random() < self.config.misalign_prob:
            misalign = _sample_horizontal_offset(self.config.misalign_range)
            logger.info("====== [AUTO] Recovery: misaligned %s by %s ======", step, np.round(misalign[:2], 3))
            halfway = 0.5 * self.config.align_height
            return [
                transit(Phase.PERTURB, misalign),
                Segment(
                    name(f"misaligned_{step}"),
                    Phase.PERTURB,
                    gripper=None,
                    position=position + misalign + np.array([0.0, 0.0, halfway]),
                    rotvec=rotvec,
                    speed=self.config.descend_speed,
                ),
                correct_to_above(halfway),
            ]

        if straight:
            return [
                Segment(
                    name(f"rise_to_{step}"),
                    Phase.TRANSIT,
                    gripper=gripper,
                    position=np.array([np.nan, np.nan, cruise_z]),
                    speed=transit_speed,
                ),
                Segment(
                    name(f"cross_to_{step}"),
                    Phase.TRANSIT,
                    gripper=gripper,
                    position=np.array([position[0], position[1], cruise_z]),
                    rotvec=rotvec,
                    speed=transit_speed,
                ),
                Segment(
                    name(f"lower_to_{step}"),
                    Phase.TRANSIT,
                    gripper=gripper,
                    position=position + above,
                    rotvec=rotvec,
                    speed=transit_speed,
                ),
            ]

        return [transit(Phase.TRANSIT, np.zeros(3))]

    def _descend_segments(
        self,
        name,
        position: np.ndarray,
        rotvec: np.ndarray,
        suction: bool,
        phase: Phase = Phase.DESCEND,
        speed: float | None = None,
    ) -> list[Segment]:
        """Slow force-stop descent onto position, at descend_speed unless speed
        is given. When suction is True, the last suction_lead_m is split off
        and switches suction on as it starts."""
        speed = self.config.descend_speed if speed is None else speed
        lead = self.config.suction_lead_m if suction else 0.0
        step = "descend_pick" if suction else "descend_place"
        if phase == Phase.PERTURB:
            step = f"missed_{step}"
        segments = []
        if lead > 0.0:
            segments.append(
                Segment(
                    name(step),
                    phase,
                    gripper=None,
                    position=position + np.array([0.0, 0.0, lead]),
                    rotvec=rotvec,
                    speed=speed,
                    force_stop=True,
                )
            )
        segments.append(
            Segment(
                name(f"{step}_contact"),
                phase,
                gripper=1.0 if suction else None,
                position=position,
                rotvec=rotvec,
                speed=speed,
                force_stop=True,
            )
        )
        return segments

    def _sample_region_pose(
        self,
        region: Region,
        height: float,
        existing_positions: list[np.ndarray],
        min_separation: float,
    ) -> list[float]:
        """Sample a pose uniformly inside region at the given height, rejecting
        candidates too close to positions already placed earlier."""
        self._require_canonical_orientation()
        best, best_separation = None, -1.0
        for _ in range(self.config.place_sampling_max_attempts):
            candidate = _sample_region_position(region, height, self._canonical_rotvec)
            separation = _min_xy_distance(candidate, existing_positions)
            if separation >= min_separation:
                return candidate
            if separation > best_separation:
                best, best_separation = candidate, separation

        logger.warning(
            "====== [AUTO] Could not find a position >= %.3fm from prior placements "
            "after %d attempts; using the farthest one found (%.3fm). ======",
            min_separation,
            self.config.place_sampling_max_attempts,
            best_separation,
        )
        return best

    def _plan_pick_poses(self, heights: list[float]) -> list[list[float]]:
        """One pick_region pose per object (scene reset), planned together:
        every pose is at least pick_min_separation from the others and at
        least place_region.exclusion_radius from its center (if set), else,
        unless pick_place_region_clearance is null, more than that far from
        place_region.

        Objects are placed one at a time with up to PER_OBJECT_ATTEMPTS
        tries each; if one doesn't fit, the whole plan is thrown away and
        redrawn from scratch (up to place_sampling_max_attempts plans), since
        earlier objects landing badly can block the rest even when all of
        them fit. If no plan fits, the one whose closest pair is farthest
        apart is used, with a warning."""
        self._require_canonical_orientation()
        region = self.config.pick_region
        min_separation = self.config.pick_min_separation
        clearance = self.config.pick_place_region_clearance

        place_region = self.config.place_region

        def clear_of_place_region(candidate: list[float]) -> bool:
            if place_region.exclusion_radius is not None:
                return (
                    np.hypot(candidate[0] - place_region.center[0], candidate[1] - place_region.center[1])
                    >= place_region.exclusion_radius
                )
            return clearance is None or place_region.distance_to(candidate[:2]) > clearance

        def candidate_for(height: float) -> list[float] | None:
            for _ in range(PER_OBJECT_ATTEMPTS):
                candidate = _sample_region_position(region, height, self._canonical_rotvec)
                if clear_of_place_region(candidate):
                    return candidate
            return None

        best_plan, best_separation = None, -1.0
        for _ in range(self.config.place_sampling_max_attempts):
            plan: list[list[float]] = []
            for height in heights:
                # Among a few tries, keep the one farthest from the plan so far.
                chosen, chosen_separation = None, -1.0
                for _ in range(PER_OBJECT_ATTEMPTS):
                    candidate = candidate_for(height)
                    if candidate is None:
                        continue
                    separation = _min_xy_distance(candidate, plan)
                    if separation > chosen_separation:
                        chosen, chosen_separation = candidate, separation
                    if separation >= min_separation:
                        break
                if chosen is None:
                    raise ValueError(
                        "auto_pick_place: no spot in pick_region is clear of place_region - enlarge "
                        "pick_region or lower place_region.exclusion_radius / pick_place_region_clearance."
                    )
                plan.append(chosen)
            plan_separation = min((_min_xy_distance(pose, plan[:k]) for k, pose in enumerate(plan)), default=np.inf)
            if plan_separation >= min_separation:
                return plan
            if plan_separation > best_separation:
                best_plan, best_separation = plan, plan_separation

        logger.warning(
            "====== [AUTO] No plan with all %d objects >= %.3fm apart after %d tries; using the best one "
            "(closest pair %.3fm apart). ======",
            len(heights),
            min_separation,
            self.config.place_sampling_max_attempts,
            best_separation,
        )
        return best_plan

    def _require_canonical_orientation(self) -> None:
        if self._canonical_rotvec is None:
            raise ValueError(
                "AutoPickPlaceController has no canonical orientation. "
                "set_canonical_orientation_from_current_pose() must run before sampling."
            )


# Tries per object within one pick plan (see _plan_pick_poses).
PER_OBJECT_ATTEMPTS = 30


def _min_xy_distance(pose: list[float], others: list) -> float:
    """Horizontal distance from pose to the closest of others (inf if none)."""
    return min((float(np.hypot(pose[0] - o[0], pose[1] - o[1])) for o in others), default=np.inf)


def _min_jerk(tau: float) -> float:
    """Minimum-jerk progress s(tau) for tau in [0, 1]: zero velocity and
    acceleration at both ends, bell-shaped speed in between - the profile
    human point-to-point reaches follow."""
    tau = min(max(tau, 0.0), 1.0)
    return tau**3 * (10.0 - 15.0 * tau + 6.0 * tau**2)


def _cubic_bezier(points: np.ndarray, s: float) -> np.ndarray:
    p0, p1, p2, p3 = points
    return (1 - s) ** 3 * p0 + 3 * (1 - s) ** 2 * s * p1 + 3 * (1 - s) * s**2 * p2 + s**3 * p3


def _transit_control_points(start: np.ndarray, end: np.ndarray, cruise_z: float) -> np.ndarray:
    """Control points of a lift -> carry -> lower path: leaves start going up,
    arrives at end coming down, peaking a bit below cruise_z in between."""
    cruise_z = max(cruise_z, start[2], end[2])
    return np.array([start, [start[0], start[1], cruise_z], [end[0], end[1], cruise_z], end], dtype=float)


def _bezier_length(points: np.ndarray, samples: int = 32) -> float:
    curve = np.array([_cubic_bezier(points, s) for s in np.linspace(0.0, 1.0, samples + 1)])
    return float(np.sum(np.linalg.norm(np.diff(curve, axis=0), axis=1)))


def _sample_horizontal_offset(magnitude_range: tuple[float, float]) -> np.ndarray:
    angle = np.random.uniform(0.0, 2.0 * np.pi)
    magnitude = np.random.uniform(*magnitude_range)
    return np.array([magnitude * np.cos(angle), magnitude * np.sin(angle), 0.0])


def _clip_to_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= max_norm or norm == 0.0:
        return vector
    return vector * (max_norm / norm)
