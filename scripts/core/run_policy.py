"""Run a pi0.5 policy served from a GPU machine on the UR5e (remote inference client).

The policy runs on the GPU machine via openpi's `scripts/serve_policy.py`
(https://github.com/THI-Robotics-Lab/openpi, branch the checkpoint was trained on,
examples/ur5_pick_place). This script only reads cameras + robot state, queries the server
over websocket, and executes the returned action chunks.

It is the recording loop (`scripts/core/record_loop.py`) with the policy in place of the
scripted `auto_pick_place` teleop: the same `UR5e` LeRobot robot class built from the same
`scripts/config/cfg.yaml`, `get_observation()` -> action -> `send_action()`. Observations and
action execution are therefore identical to recording: flange pose with XYZ-Euler rotation,
base-frame deltas through the speed-limited pose interpolator + servoL, suction gripper,
OAK cameras at 640x360.

Run from the repo root (so `scripts` and the vendored `openpi_client` are importable):

  python -m scripts.core.run_policy --host <GPU_PC_IP> --fake-obs --num-steps 50
  python -m scripts.core.run_policy --host <GPU_PC_IP> --dry-run --prompt "stack the plates"

Modes:
  --fake-obs   No robot/cameras needed: sends random observations (link test).
  --dry-run    Real cameras + robot state, but never commands the robot (arm or suction).
  (default)    Closed-loop control.

Source of truth: examples/ur5_pick_place/robot_client.py in the openpi repo; keep in sync.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import time

import numpy as np
from openpi_client import action_chunk_broker
from openpi_client import image_tools
from openpi_client import websocket_client_policy

# Order of the dataset's 48-dim `observation.state` (meta/info.json); matches
# UR5e._motors_ft in the recording repo. The server-side UR5Inputs transform selects
# tcp_pose.{x,y,z} (0-2), tcp_force (12-17) and gripper_raw_position (18).
STATE_NAMES = (
    *(f"tcp_pose.{a}" for a in ("x", "y", "z", "rx", "ry", "rz")),
    *(f"tcp_speed.{a}" for a in ("x", "y", "z", "rx", "ry", "rz")),
    *(f"tcp_force.{a}" for a in ("x", "y", "z", "rx", "ry", "rz")),
    "gripper_raw_position",
    "gripper_raw_bin",
    "gripper_action_bin",
    *(f"joint_{i}.pos" for i in range(1, 7)),
    *(f"joint_{i}.vel" for i in range(1, 7)),
    *(f"joint_{i}.acc" for i in range(1, 7)),
    *(f"joint_{i}.force" for i in range(1, 7)),
    *(f"tcp_acc.{a}" for a in ("x", "y", "z")),
)
ACTION_NAMES = ("delta_x", "delta_y", "delta_z", "delta_rx", "delta_ry", "delta_rz", "gripper_position")

# Task strings recorded in the training data (keys of task_paraphrases.json). Other wording
# works too (the model was trained on paraphrases), but these are the safest prompts.
TRAINING_TASKS = (
    "stack the plates",
    "pick the blue plate and place it on the stack",
    "pick the yellow plate and place it on the stack",
    "pick the grey plate and place it on the stack",
    "place the yellow plate in the position marked with the blue square",
    "place the blue plate in the position marked with the blue square",
    "place the tupperware in the position marked with the blue square",
)
DEFAULT_PROMPT = TRAINING_TASKS[0]

# Per-tick caps used by the scripted demonstrations (cfg.yaml auto_pick_place.max_step_size /
# max_rot_step_size); policy actions are clipped to these as a safety limit.
MAX_DELTA_POS = 0.02
MAX_DELTA_ROT = 0.05

# action_horizon of the pi05_ur5_pick_place_* configs: the server returns (30, 7) chunks.
ACTION_HORIZON = 30

IMAGE_SIZE = 224


def preprocess_image(img: np.ndarray) -> np.ndarray:
    # Resize on the client to keep messages small; matches the training preprocessing.
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(img, IMAGE_SIZE, IMAGE_SIZE))


def fake_observation(prompt: str) -> dict:
    return {
        "observation/image": np.random.randint(256, size=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8),
        "observation/state": np.random.rand(len(STATE_NAMES)).astype(np.float32),
        "prompt": prompt,
    }


def to_policy_observation(robot_obs: dict, prompt: str) -> dict:
    """Converts a UR5e.get_observation() dict into the policy server's input format."""
    return {
        "observation/image": preprocess_image(robot_obs["exterior_image"]),
        "observation/wrist_image": preprocess_image(robot_obs["wrist_image"]),
        "observation/state": np.array([robot_obs[name] for name in STATE_NAMES], dtype=np.float32),
        "prompt": prompt,
    }


def to_robot_action(action: np.ndarray, robot_obs: dict, args: argparse.Namespace) -> dict:
    """Converts one policy action into a UR5e.send_action() dict, with safety clipping.

    Deltas are base-frame offsets from the current flange pose (reference_frame: base),
    so the workspace box is enforced on tcp_pose + delta.
    """
    action = np.asarray(action, dtype=np.float64)
    delta_pos = np.clip(action[:3], -args.max_delta_pos, args.max_delta_pos)
    delta_rot = np.clip(action[3:6], -args.max_delta_rot, args.max_delta_rot)
    if args.workspace_min is not None or args.workspace_max is not None:
        current = np.array([robot_obs["tcp_pose.x"], robot_obs["tcp_pose.y"], robot_obs["tcp_pose.z"]])
        target = np.clip(current + delta_pos, args.workspace_min, args.workspace_max)
        delta_pos = target - current
    return dict(zip(ACTION_NAMES, [*delta_pos, *delta_rot, action[6]], strict=True))


def _override(cli_value, cfg_value):
    return cfg_value if cli_value is None else cli_value


def make_robot(cfg_path: Path, args: argparse.Namespace):
    """Builds the UR5e exactly as the recording script does, from the same cfg.yaml."""
    from lerobot_robot_ur5e import UR5e
    from lerobot_robot_ur5e import UR5eConfig
    import yaml

    from scripts.core.run_record import RecordConfig
    from scripts.core.run_record import make_camera_configs

    with open(cfg_path) as f:
        record_cfg = RecordConfig(yaml.safe_load(f)["record"])

    robot_config = UR5eConfig(
        robot_ip=record_cfg.robot_ip,
        gripper_port=record_cfg.gripper_port,
        cameras=make_camera_configs(record_cfg),
        use_gripper=record_cfg.use_gripper,
        init_gripper=record_cfg.init_gripper,
        close_threshold=record_cfg.close_threshold,
        gripper_reverse=record_cfg.gripper_reverse,
        gripper_bin_threshold=record_cfg.gripper_bin_threshold,
        gripper_force=record_cfg.gripper_force,
        gripper_speed=record_cfg.gripper_speed,
        control_space=record_cfg.control_space,
        # debug=True: the robot class skips all arm motion (reset and send_action).
        debug=args.dry_run or record_cfg.debug,
        kp=record_cfg.kp,
        kd=record_cfg.kd,
        kp_rot=record_cfg.kp_rot,
        kd_rot=record_cfg.kd_rot,
        hold_current_pose_on_idle=record_cfg.hold_current_pose_on_idle,
        rtde_freq=record_cfg.rtde_freq,
        select_vector=record_cfg.select_vector,
        force_limit=record_cfg.force_limit,
        pos_delta=record_cfg.pos_delta,
        vel_delta=record_cfg.vel_delta,
        gain_scale=record_cfg.gain_scale,
        speed=record_cfg.speed,
        acceleration=record_cfg.acceleration,
        # servo_time and the speed caps come from cfg.yaml (what the training data was recorded
        # with); the CLI flags only override them when given.
        servo_time=_override(args.servo_time, record_cfg.servo_time),
        lookahead_time=record_cfg.lookahead_time,
        gain=record_cfg.gain,
        max_pos_speed=_override(args.max_pos_speed, record_cfg.max_pos_speed),
        max_rot_speed=_override(args.max_rot_speed, record_cfg.max_rot_speed),
        payload_mass=record_cfg.payload_mass,
        payload_cog=record_cfg.payload_cog,
        init_joints_deg=record_cfg.init_joints_deg,
        joint_speed=record_cfg.joint_speed,
        joint_acceleration=record_cfg.joint_acceleration,
        reference_frame=record_cfg.reference_frame,
        control_frame_euler_deg=record_cfg.control_frame_euler_deg,
    )
    if record_cfg.reference_frame != "base" or record_cfg.control_space != "position":
        logging.warning(
            "cfg.yaml has reference_frame=%s, control_space=%s; the training data used base/position.",
            record_cfg.reference_frame,
            record_cfg.control_space,
        )
    if float(record_cfg.fps) != args.hz:
        logging.warning("--hz %.1f differs from cfg.yaml fps %s (the training data rate).", args.hz, record_cfg.fps)
    if robot_config.servo_time >= 1.0 / args.hz:
        logging.warning(
            "servo_time %.3f s is not below the control period %.3f s; servoL will chase stale targets.",
            robot_config.servo_time,
            1.0 / args.hz,
        )
    logging.info(
        "servo_time=%.3f s, max_pos_speed=%.2f m/s, max_rot_speed=%.2f rad/s",
        robot_config.servo_time,
        robot_config.max_pos_speed,
        robot_config.max_rot_speed,
    )
    return UR5e(robot_config), record_cfg


class LatencyTracker:
    def __init__(self) -> None:
        self.samples: list[float] = []

    def wrap(self, policy):
        tracker = self

        class _Timed:
            def infer(self, obs):
                t0 = time.perf_counter()
                result = policy.infer(obs)
                tracker.samples.append(time.perf_counter() - t0)
                logging.info(
                    "infer %.1f ms, actions %s", 1000 * tracker.samples[-1], np.asarray(result["actions"]).shape
                )
                return result

            def reset(self):
                policy.reset()

        return _Timed()

    def summary(self) -> str:
        if not self.samples:
            return "no inference calls"
        ms = 1000 * np.asarray(self.samples)
        return f"n={len(ms)} p50={np.percentile(ms, 50):.1f}ms p95={np.percentile(ms, 95):.1f}ms max={ms.max():.1f}ms"


def main(args: argparse.Namespace) -> None:
    logging.info("Prompt: %r", args.prompt)
    if args.prompt not in TRAINING_TASKS:
        logging.warning("Prompt is not one of the recorded task strings: %s", TRAINING_TASKS)

    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    logging.info("Connected. Server metadata: %s", client.get_server_metadata())

    latency = LatencyTracker()
    policy = action_chunk_broker.ActionChunkBroker(latency.wrap(client), action_horizon=args.replan_steps)

    robot = None
    if not args.fake_obs:
        robot, record_cfg = make_robot(args.cfg, args)
        robot.connect()
        robot.reset_to_init_pose(record_cfg.init_joints_deg)
        robot.set_episode_reference_pose()

    dt = 1.0 / args.hz
    try:
        for step in range(args.num_steps):
            t_start = time.perf_counter()

            if robot is None:
                robot_obs = None
                obs = fake_observation(args.prompt)
            else:
                robot_obs = robot.get_observation()
                obs = to_policy_observation(robot_obs, args.prompt)

            action = np.asarray(policy.infer(obs)["actions"])

            if robot is not None and not args.dry_run:
                robot.send_action(to_robot_action(action, robot_obs, args))
            elif step % args.replan_steps == 0:
                logging.info("step %d action %s", step, np.array2string(action, precision=4))

            time.sleep(max(0.0, dt - (time.perf_counter() - t_start)))
    except KeyboardInterrupt:
        logging.info("Interrupted.")
    finally:
        if robot is not None:
            # Stops RTDE control and releases the suction gripper.
            robot.disconnect()
        logging.info("Inference latency: %s", latency.summary())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", required=True, help="IP of the machine running serve_policy.py")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument(
        "--cfg",
        type=Path,
        default=None,
        help="Recording repo cfg.yaml (default: scripts/config/cfg.yaml of the installed recording repo)",
    )
    p.add_argument("--fake-obs", action="store_true", help="Send random observations; no robot/cameras")
    p.add_argument("--dry-run", action="store_true", help="Read robot + cameras but never move the robot")
    p.add_argument(
        "--hz", type=float, default=30.0, help="Control rate (must match dataset fps; current datasets are 30 fps)"
    )
    p.add_argument(
        "--replan-steps",
        type=int,
        default=5,
        help="Actions executed per inference (<= action horizon, 30 for the pi05_ur5 configs)",
    )
    p.add_argument("--num-steps", type=int, default=900, help="Episode length in control steps")
    p.add_argument("--servo-time", type=float, default=None, help="servoL time (s); default: cfg.yaml")
    p.add_argument("--max-pos-speed", type=float, default=None, help="TCP linear speed cap (m/s); default: cfg.yaml")
    p.add_argument("--max-rot-speed", type=float, default=None, help="TCP rotation speed cap (rad/s); default: cfg.yaml")
    p.add_argument("--max-delta-pos", type=float, default=MAX_DELTA_POS)
    p.add_argument("--max-delta-rot", type=float, default=MAX_DELTA_ROT)
    p.add_argument("--workspace-min", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    p.add_argument("--workspace-max", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    args = p.parse_args()
    if not 1 <= args.replan_steps <= ACTION_HORIZON:
        p.error(f"--replan-steps must be in [1, {ACTION_HORIZON}]")

    if args.cfg is None and not args.fake_obs:
        import scripts

        args.cfg = Path(scripts.__file__).parent / "config" / "cfg.yaml"
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(parse_args())
