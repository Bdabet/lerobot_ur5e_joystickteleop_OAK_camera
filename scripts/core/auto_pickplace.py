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
    """An x/y bounding box (meters, robot base frame) that pick/place
    positions are sampled uniformly from. Orientation is not configured here -
    every sampled pose uses whatever orientation the robot actually has at
    robot.init_pose (see AutoPickPlaceController.set_canonical_orientation_
    from_current_pose()), so it can never drift from init_pose."""

    def __init__(self, cfg: dict[str, Any], name: str):
        if "x_range" not in cfg or "y_range" not in cfg:
            raise ValueError(f"auto_pick_place.{name} requires x_range and y_range.")
        self.x_range: tuple[float, float] = tuple(float(v) for v in cfg["x_range"])
        self.y_range: tuple[float, float] = tuple(float(v) for v in cfg["y_range"])


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

        # Suction is switched on suction_lead_m before reaching the object
        # (like a person closing the hand on the way in), then held
        # grasp_seal_s to let the seal form. release_s is the hold after
        # switching suction off at the place point.
        self.suction_lead_m: float = cfg.get("suction_lead_m", 0.005)
        self.grasp_seal_s: float = cfg.get("grasp_seal_s", 0.25)
        self.release_s: float = cfg.get("release_s", 0.25)

        # Stop the downward pick/place move early on TCP force contact instead
        # of servoing all the way to the (possibly optimistic) target depth.
        self.force_stop_enabled: bool = cfg.get("force_stop_enabled", False)
        self.contact_force_threshold: float = cfg.get("contact_force_threshold", 15.0)
        # Right after grasp, the TCP first lifts this far straight up (slowly)
        # from wherever it actually is before the transit arc starts.
        self.force_stop_clearance_height: float = cfg.get("force_stop_clearance_height", 0.02)

        self.shuffle_pick_order: bool = cfg.get("shuffle_pick_order", True)
        self.pick_min_separation: float = cfg.get("pick_min_separation", 0.05)
        self.place_min_separation: float = cfg.get("place_min_separation", 0.05)
        self.place_sampling_max_attempts: int = cfg.get("place_sampling_max_attempts", 200)


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


@dataclass
class Segment:
    """One time-parameterized move. The start pose is captured live from the
    robot when the segment begins (so a move that follows an early force-stop
    contact starts from wherever contact actually happened); the end pose is
    either absolute (position/rotvec) or relative to that start (offset), and
    a segment with neither is a hold in place for duration_s."""

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
    region's bounding box, z fixed at height, and the canonical orientation
    (whatever the robot's actual orientation is at robot.init_pose)."""
    x = float(np.random.uniform(region.x_range[0], region.x_range[1]))
    y = float(np.random.uniform(region.y_range[0], region.y_range[1]))
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
        self._episode_complete = False
        self._last_gripper = 1.0
        # Per-object fixed height, set once during loading (index == object id).
        self._object_heights: list[float] = []
        # object index -> its current pose [x,y,z,rx,ry,rz] (rotvec), updated
        # whenever a pick/place block moves it (loading, episode placement, or
        # scene reset), so the next episode/step picks it up from where it
        # really is instead of a fresh, independently-sampled guess.
        self._object_positions: dict[int, list[float]] = {}
        # (object_index, actual_place_pose) per object from the most recently
        # built episode, used by build_scene_reset_waypoints() to bring objects
        # back from their goal position to a fresh spot in pick_region.
        self._last_placements: list[tuple[int, list[float]]] = []
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
        """Phase of the segment that produced the most recent action."""
        segment = self.current_segment
        return int(segment.phase if segment is not None else Phase.IDLE)

    @property
    def episode_complete(self) -> bool:
        return self._episode_complete

    def reset_step_size(self) -> None:
        """Per-episode reset hook (called once before each episode starts)."""
        self._set_segments(self._build_episode_segments())
        self._last_gripper = 0.0

    def build_scene_reset_waypoints(self) -> bool:
        """Build a segment sequence that returns each object from where the last
        episode placed it to a freshly randomized position inside pick_region.

        Intended to be run through record_loop with dataset=None (not recorded as
        a demonstration) between episodes, so the scene is ready for the next
        episode's pick-and-place without human intervention. Returns False (no
        segments built) if there is no completed episode's placements to reset.
        """
        if not self._last_placements:
            return False

        segments: list[Segment] = []
        reset_positions: list[np.ndarray] = []
        # Not recorded, so no need for speed variety - just go briskly.
        transit_speed = self.config.transit_speed_range[1]
        for i, (object_idx, placed_pose) in enumerate(self._last_placements, start=1):
            height = self._object_heights[object_idx]
            reset_target = self._sample_region_pose(
                self.config.pick_region, height, reset_positions, self.config.pick_min_separation
            )
            reset_positions.append(np.array(reset_target[:3], dtype=float))
            self._object_positions[object_idx] = reset_target
            segments.extend(
                self._pick_place_block(i, placed_pose, reset_target, transit_speed, name_prefix="reset_")
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
                timed_out = elapsed >= self._active.duration_s + self.config.settle_timeout_s
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

    # ======= internals =======

    def _set_segments(self, segments: list[Segment]) -> None:
        self._segments = segments
        self._current_idx = 0
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

        placed_positions: list[np.ndarray] = []
        placements: list[tuple[int, list[float]]] = []
        segments: list[Segment] = []

        for i, object_idx in enumerate(object_order, start=1):
            # Objects sit exactly where loading (or the previous episode's
            # reset) left them - pick them up from there.
            pick_pose = self._object_positions[object_idx]

            height = self._object_heights[object_idx]
            place_sampled = self._sample_region_pose(
                self.config.place_region, height, placed_positions, self.config.place_min_separation
            )
            placed_positions.append(np.array(place_sampled[:3], dtype=float))
            placements.append((object_idx, place_sampled))
            self._object_positions[object_idx] = place_sampled

            segments.extend(self._pick_place_block(i, pick_pose, place_sampled, transit_speed))
            logger.info(
                "====== [AUTO] Object %d/%d: pick=%s place=%s ======",
                i,
                len(object_order),
                [round(v, 4) for v in pick_pose],
                [round(v, 4) for v in place_sampled],
            )

        self._last_placements = placements
        return segments

    def _pick_place_block(
        self,
        index: int,
        source_pose: list[float],
        target_pose: list[float],
        transit_speed: float,
        name_prefix: str = "",
    ) -> list[Segment]:
        """Segments moving one object from source_pose to target_pose, starting
        from wherever the arm is: arc over to just above the object, descend
        slowly (suction on just before contact), seal, lift clear, arc over to
        just above the target, descend, release, retreat up."""
        source_position = np.array(source_pose[:3], dtype=float)
        source_rotvec = np.array(source_pose[3:], dtype=float)
        target_position = np.array(target_pose[:3], dtype=float)
        target_rotvec = np.array(target_pose[3:], dtype=float)
        above = np.array([0.0, 0.0, self.config.align_height])
        cruise_z = max(source_position[2], target_position[2]) + self.config.approach_height

        def name(step: str) -> str:
            return f"{name_prefix}{step}_{index}"

        segments = [
            Segment(
                name("transit_to_pick"),
                Phase.TRANSIT,
                gripper=0.0,
                position=source_position + above,
                rotvec=source_rotvec,
                speed=transit_speed,
                cruise_z=cruise_z,
            ),
            *self._descend_segments(name, source_position, source_rotvec, suction=True),
            Segment(name("grasp"), Phase.GRASP, gripper=1.0, duration_s=self.config.grasp_seal_s),
            Segment(
                name("lift"),
                Phase.LIFT,
                gripper=None,
                offset=np.array([0.0, 0.0, self.config.force_stop_clearance_height]),
                speed=self.config.descend_speed,
            ),
            Segment(
                name("transit_to_place"),
                Phase.TRANSIT,
                gripper=None,
                position=target_position + above,
                rotvec=target_rotvec,
                speed=transit_speed,
                cruise_z=cruise_z,
            ),
            *self._descend_segments(name, target_position, target_rotvec, suction=False),
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

    def _descend_segments(self, name, position: np.ndarray, rotvec: np.ndarray, suction: bool) -> list[Segment]:
        """Slow force-stop descent onto position. When suction is True, the
        last suction_lead_m is split off and switches suction on as it starts."""
        lead = self.config.suction_lead_m if suction else 0.0
        step = "descend_pick" if suction else "descend_place"
        segments = []
        if lead > 0.0:
            segments.append(
                Segment(
                    name(step),
                    Phase.DESCEND,
                    gripper=None,
                    position=position + np.array([0.0, 0.0, lead]),
                    rotvec=rotvec,
                    speed=self.config.descend_speed,
                    force_stop=True,
                )
            )
        segments.append(
            Segment(
                name(f"{step}_contact"),
                Phase.DESCEND,
                gripper=1.0 if suction else None,
                position=position,
                rotvec=rotvec,
                speed=self.config.descend_speed,
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
        if self._canonical_rotvec is None:
            raise ValueError(
                "AutoPickPlaceController has no canonical orientation. "
                "set_canonical_orientation_from_current_pose() must run before sampling."
            )
        candidate = _sample_region_position(region, height, self._canonical_rotvec)
        for _ in range(self.config.place_sampling_max_attempts):
            candidate_position = np.array(candidate[:3], dtype=float)
            if all(
                np.linalg.norm(candidate_position - prev) >= min_separation
                for prev in existing_positions
            ):
                return candidate
            candidate = _sample_region_position(region, height, self._canonical_rotvec)

        logger.warning(
            "====== [AUTO] Could not find a position >= %.3fm from prior placements "
            "after %d attempts; using closest candidate found. ======",
            min_separation,
            self.config.place_sampling_max_attempts,
        )
        return candidate


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
