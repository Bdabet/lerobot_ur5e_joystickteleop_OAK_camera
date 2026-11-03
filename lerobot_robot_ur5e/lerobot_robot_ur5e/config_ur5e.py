from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from lerobot.robots.config import RobotConfig

@RobotConfig.register_subclass("ur5e_robot")
@dataclass
class UR5eConfig(RobotConfig):
    use_gripper: bool = True
    init_gripper: bool = True
    gripper_reverse: bool = False
    robot_ip: str = "172.17.0.3"
    gripper_port: str = "/dev/ur5e_left_gripper"
    gripper_force: int = 70
    gripper_speed: int = 60
    gripper_bin_threshold: float = 0.98
    close_threshold: float = 0.7
    debug: bool = False
    kp: int = 1200
    kd: int = 600
    kp_rot: int = 2000
    kd_rot: int = 400
    hold_current_pose_on_idle: bool = False
    rtde_freq: int = 100
    select_vector: list = field(default_factory=lambda: [1, 1, 1, 0, 0, 0])
    force_limit: list = field(default_factory=lambda: [2, 2, 2, 2, 2, 2])
    pos_delta: float = 0.02
    vel_delta: float = 4.0
    gain_scale: float = 1.5
    control_frame_euler_deg: list = field(
        default_factory=lambda: [0.0, 0.0, 0.0]
    )
    speed: float = 0.8
    acceleration: float = 0.5
    servo_time: float = 0.1
    lookahead_time: float = 0.1
    gain: int = 300
    max_pos_speed: float = 0.25
    max_rot_speed: float = 0.5
    force_stop_limit: float | None = None
    force_stop_deceleration: float = 5.0
    payload_mass: float = 1.601
    payload_cog: list = field(default_factory=lambda: [0.011, -0.002, 0.052])
    # Reset target as joint angles [base, shoulder, elbow, wrist1, wrist2, wrist3] in degrees
    # (as shown on the teach pendant); reached with moveJ at joint_speed/joint_acceleration.
    init_joints_deg: list | None = None
    joint_speed: float = 0.5  # rad/s
    joint_acceleration: float = 0.5  # rad/s^2
    reference_frame: str = "base"  # "base" or "tcp"
    control_space: str = "position"  # "position" or "force"
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
