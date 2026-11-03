"""Interactive object loading for auto_pick_place mode.

Before episodes start, the operator is asked how many objects will be used.
For each one: the arm holds at robot.init_pose with the gripper turned on for
auto_pick_place.object_load_hold_s seconds so the operator can hand it the
object, then control is handed to a manual jog teleop so the operator can
move the (suctioned) object down onto the table wherever they like; pressing
the right arrow key (the same "confirm" convention used elsewhere in this
codebase) ends the jog. The gripper is then released by a scripted segment
(not left to the operator to remember to do themselves) before the arm lifts
away. The object's actual height at that point becomes its fixed height for
the rest of the run, and where it was released becomes its starting position
in pick_region.

Uses the AutoPickPlaceController itself (already connected/set_robot()) to
drive the scripted moves (hold, retreat) through record_loop, exactly like
build_scene_reset_waypoints() does between episodes - not recorded as
demonstration data (dataset=None).
"""

import logging
import sys

from scripts.core.record_loop import record_loop

if sys.platform == "win32":
    import msvcrt
else:
    import termios

logger = logging.getLogger(__name__)


def _flush_stdin() -> None:
    if sys.platform == "win32":
        while msvcrt.kbhit():
            msvcrt.getch()
    else:
        termios.tcflush(sys.stdin, termios.TCIFLUSH)


def _ask_num_objects(question: str = "How many objects will you load?", maximum: int | None = None) -> int:
    while True:
        _flush_stdin()
        answer = input(f"====== [LOAD] {question} ").strip()
        try:
            num_objects = int(answer)
        except ValueError:
            print("Please enter a whole number.")
            continue
        if num_objects <= 0:
            print("Please enter a number greater than 0.")
            continue
        if maximum is not None and num_objects > maximum:
            print(f"Please enter a number no greater than {maximum}.")
            continue
        return num_objects


def _run_step(
    record_cfg,
    robot,
    teleop,
    events: dict,
    teleop_action_processor,
    robot_action_processor,
    robot_observation_processor,
    control_time_s: float | None = None,
) -> None:
    events["exit_early"] = False
    record_loop(
        robot=robot,
        events=events,
        fps=record_cfg.fps,
        teleop=teleop,
        teleop_action_processor=teleop_action_processor,
        robot_action_processor=robot_action_processor,
        robot_observation_processor=robot_observation_processor,
        control_time_s=control_time_s if control_time_s is not None else record_cfg.episode_time_sec,
        display_data=record_cfg.display,
    )
    if events["stop_recording"]:
        raise RuntimeError("Object loading cancelled (stop_recording).")
    events["exit_early"] = False


def load_objects_into_regions(
    record_cfg,
    robot,
    teleop,
    jog_teleop,
    events: dict,
    teleop_action_processor,
    robot_action_processor,
    robot_observation_processor,
) -> None:
    """Run the interactive per-object loading sequence described above.

    Populates the controller (teleop) with each loaded object's fixed height
    and current position via register_loaded_object(), so the first episode's
    reset_step_size() has objects to pick from pick_region.
    """
    logging.info("====== [LOAD] Starting object loading ======")
    num_objects = _ask_num_objects()

    jog_teleop.set_robot(robot)
    jog_teleop.connect()

    try:
        for i in range(1, num_objects + 1):
            logging.info(
                "\033[32m====== [LOAD] Object %d/%d: moving to init pose, gripper on for %.1fs ======\033[0m",
                i,
                num_objects,
                record_cfg.auto_pick_place.object_load_hold_s,
            )
            # Direct blocking moveJ to robot.init_joints_deg, same call used
            # everywhere else to reach the init pose - it's a joint target, so
            # it can't be streamed through the controller (which tracks
            # Cartesian waypoints in the TCP-offset-removed EE frame).
            robot.reset_to_init_pose(record_cfg.init_joints_deg)
            teleop.build_wait_for_object_waypoints(record_cfg.auto_pick_place.object_load_hold_s)
            _run_step(
                record_cfg, robot, teleop, events, teleop_action_processor, robot_action_processor, robot_observation_processor
            )

            logging.info(
                "\033[32m====== [LOAD] Object %d/%d: jog it down to where it should sit, then press the "
                "right arrow key to confirm (the gripper releases automatically) ======\033[0m",
                i,
                num_objects,
            )
            # The scripted hold step above just turned suction on; seed the
            # jog teleop's own gripper state to match (1 = CLOSE, suction on)
            # instead of its normal open-on-connect default (0), so handing
            # off control doesn't drop the object. Release itself is scripted
            # right after (build_retreat_after_manual_place_waypoints), not
            # left to the operator to remember.
            jog_teleop.gripper_action = 1
            jog_teleop.reset_step_size()
            _run_step(
                record_cfg,
                robot,
                jog_teleop,
                events,
                teleop_action_processor,
                robot_action_processor,
                robot_observation_processor,
                control_time_s=record_cfg.auto_pick_place.manual_place_time_sec,
            )

            # Keep the operator's x/y/height, but snap orientation to
            # pick_region's fixed rotation - every later pick (every
            # automatic scene reset) uses that same orientation, so the raw
            # jogged wrist angle would otherwise force a one-off, possibly
            # large reorientation the very first time this object is picked.
            placed_pose = teleop.canonicalize_pick_pose(robot.get_ee_pose())
            height = placed_pose[2]
            logging.info(
                "====== [LOAD] Object %d/%d placed at %s (height=%.4f) ======",
                i,
                num_objects,
                [round(v, 4) for v in placed_pose],
                height,
            )

            teleop.build_retreat_after_manual_place_waypoints(placed_pose)
            _run_step(
                record_cfg, robot, teleop, events, teleop_action_processor, robot_action_processor, robot_observation_processor
            )

            teleop.register_loaded_object(placed_pose, height)
    finally:
        jog_teleop.disconnect()

    logging.info("====== [LOAD] All %d objects loaded ======", num_objects)


def teach_scene(
    record_cfg,
    robot,
    teleop,
    jog_teleop,
    events: dict,
    teleop_action_processor,
    robot_action_processor,
    robot_observation_processor,
    num_objects: int | None = None,
) -> int:
    """scene_reset "manual": the operator prepares the scene by hand, then
    teaches where each object is by jogging the cup down onto it (suction
    off) and pressing the right arrow key. The TCP pose there becomes that
    object's pick pose and height, exactly as if it had been loaded by
    load_objects_into_regions(). Replaces any previously registered objects.

    Asks for the number of objects if num_objects is None; returns it so
    later calls can reuse it.
    """
    if num_objects is None:
        num_objects = _ask_num_objects()

    robot.reset_to_init_pose(record_cfg.init_joints_deg)
    teleop.clear_loaded_objects()
    jog_teleop.set_robot(robot)
    # Stays connected for the whole run (run_record disconnects it): pynput's
    # listener thread doesn't stop right away, so a disconnect here followed by
    # a connect before the next episode fails with "already connected".
    if not jog_teleop.is_connected:
        jog_teleop.connect()
    # The jog teleop is already live here, so the joystick can move the
    # arm out of the way while the operator arranges the objects.
    logging.info(
        "\033[32m====== [SCENE] Arrange the %d object(s) by hand for the next episode (joystick "
        "moves the arm), then press the right arrow key ======\033[0m",
        num_objects,
    )
    jog_teleop.gripper_action = 0
    jog_teleop.reset_step_size()
    _run_step(
        record_cfg,
        robot,
        jog_teleop,
        events,
        teleop_action_processor,
        robot_action_processor,
        robot_observation_processor,
        control_time_s=record_cfg.auto_pick_place.manual_place_time_sec,
    )

    for i in range(1, num_objects + 1):
        logging.info(
            "\033[32m====== [SCENE] Object %d/%d: jog the cup down onto it, then press the "
            "right arrow key to confirm ======\033[0m",
            i,
            num_objects,
        )
        jog_teleop.gripper_action = 0
        jog_teleop.reset_step_size()
        _run_step(
            record_cfg,
            robot,
            jog_teleop,
            events,
            teleop_action_processor,
            robot_action_processor,
            robot_observation_processor,
            control_time_s=record_cfg.auto_pick_place.manual_place_time_sec,
        )

        pick_pose = teleop.canonicalize_pick_pose(robot.get_ee_pose())
        logging.info(
            "====== [SCENE] Object %d/%d at %s ======", i, num_objects, [round(v, 4) for v in pick_pose]
        )
        teleop.register_loaded_object(pick_pose, pick_pose[2])

        # Suction off (should already be) and lift clear before the next one.
        teleop.build_retreat_after_manual_place_waypoints(pick_pose)
        _run_step(
            record_cfg, robot, teleop, events, teleop_action_processor, robot_action_processor, robot_observation_processor
        )

    logging.info("====== [SCENE] All %d objects taught ======", num_objects)
    return num_objects


def load_objects_from_stack(
    record_cfg,
    robot,
    teleop,
    jog_teleop,
    events: dict,
    teleop_action_processor,
    robot_action_processor,
    robot_observation_processor,
    num_objects: int | None = None,
    stack_size: int | None = None,
) -> tuple[int, int]:
    """load_mode "stack": the operator builds a stack of stack_size objects
    anywhere and jogs the cup down onto the top one (right arrow key to
    confirm). The robot then unstacks the top num_objects of them into fresh
    spots in pick_region, which is where the next episode picks them from;
    the rest stay as the base the episodes stack them back onto. Replaces
    any previously registered objects.

    Asks for both counts if not given; returns (num_objects, stack_size) so
    later calls can reuse them.
    """
    if stack_size is None:
        stack_size = _ask_num_objects("How many objects are in the stack?")
    if num_objects is None:
        num_objects = _ask_num_objects(
            f"How many of them should be picked off the top (1-{stack_size})?", maximum=stack_size
        )

    robot.reset_to_init_pose(record_cfg.init_joints_deg)
    jog_teleop.set_robot(robot)
    # Stays connected for the whole run, see teach_scene().
    if not jog_teleop.is_connected:
        jog_teleop.connect()
    logging.info(
        "\033[32m====== [STACK] Stack all %d object(s) anywhere in reach, jog the cup down onto the top "
        "one, then press the right arrow key ======\033[0m",
        stack_size,
    )
    jog_teleop.gripper_action = 0
    jog_teleop.reset_step_size()
    _run_step(
        record_cfg,
        robot,
        jog_teleop,
        events,
        teleop_action_processor,
        robot_action_processor,
        robot_observation_processor,
        control_time_s=record_cfg.auto_pick_place.manual_place_time_sec,
    )

    top_pose = teleop.canonicalize_pick_pose(robot.get_ee_pose())
    logging.info("====== [STACK] Stack top at %s ======", [round(v, 4) for v in top_pose])
    teleop.build_retreat_after_manual_place_waypoints(top_pose)
    _run_step(
        record_cfg, robot, teleop, events, teleop_action_processor, robot_action_processor, robot_observation_processor
    )

    top_z = teleop.register_loaded_stack(top_pose, num_objects, stack_size)
    logging.info(
        "====== [STACK] Unstacking the top %d of %d object(s) into the pick region ======", num_objects, stack_size
    )
    teleop.build_scene_reset_waypoints(stack_top_z=top_z)
    _run_step(
        record_cfg, robot, teleop, events, teleop_action_processor, robot_action_processor, robot_observation_processor
    )
    return num_objects, stack_size
