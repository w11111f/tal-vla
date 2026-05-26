from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

from isaacsim import SimulationApp

parser = argparse.ArgumentParser(description="Isaac Sim automatic OpenPI H5 collector")
parser.add_argument("--usd-path", type=str, default="/root/Desktop/Collected_exp3/expff.usd")
parser.add_argument("--dataset-dir", type=str, default="/root/gpufree-data/h5_dataset_auto")
parser.add_argument("--prompt", type=str, default="pick the cube and place in the pallet")
parser.add_argument("--num-episodes", type=int, default=1)
parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument("--image-width", type=int, default=224)
parser.add_argument("--image-height", type=int, default=224)
parser.add_argument("--robot-prim-path", type=str, default="/World/Mobie_grasper2")
parser.add_argument("--cube-prim-path", type=str, default="/World/Cube")
parser.add_argument("--pallet-prim-path", type=str, default="/World/SmallPallet")
parser.add_argument("--camera-high-path", type=str, default="/World/high")
parser.add_argument("--camera-wrist-path", type=str, default="/World/Mobie_grasper2/firefighter/joint6/wrist")
parser.add_argument("--ee-prim-path", type=str, default="/World/Mobie_grasper2/firefighter/joint6")
parser.add_argument("--ee-frame-name", type=str, default="joint6")
parser.add_argument("--robot-description-path", type=str, required=True)
parser.add_argument("--robot-urdf-path", type=str, required=True)
parser.add_argument("--finger-open", type=float, default=0.13824108)
parser.add_argument("--finger-closed", type=float, default=0.0)
parser.add_argument("--approach-height", type=float, default=0.12)
parser.add_argument("--grasp-height-offset", type=float, default=0.01)
parser.add_argument("--lift-height", type=float, default=0.16)
parser.add_argument("--place-height-offset", type=float, default=0.04)
parser.add_argument("--move-steps", type=int, default=45)
parser.add_argument("--gripper-steps", type=int, default=20)
parser.add_argument("--settle-steps", type=int, default=10)
parser.add_argument("--warmup-steps", type=int, default=240)
parser.add_argument("--tool-offset-x", type=float, default=0.0)
parser.add_argument("--tool-offset-y", type=float, default=0.0)
parser.add_argument("--tool-offset-z", type=float, default=0.0)
args, unknown_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + unknown_args

simulation_app = SimulationApp({"headless": args.headless})

import h5py
import numpy as np
import omni.replicator.core as rep
import omni.usd
from pxr import Gf
from pxr import Usd
from pxr import UsdGeom

CONTROL_JOINT_NAMES = [
    "joint1_to_base",
    "joint2_to_joint1",
    "joint3_to_joint2",
    "joint4_to_joint3",
    "joint5_to_joint4",
    "joint6_to_joint5",
    "finger_joint",
]

TRAIN_INIT_STATE = np.array(
    [-0.12466581, -0.15327631, 1.2, -0.1757595, 1.5070096, -0.320009, 0.13824108],
    dtype=np.float32,
)


@dataclass(frozen=True)
class StageTarget:
    name: str
    position: np.ndarray
    finger_joint_target: np.float32
    num_steps: int


@dataclass
class SceneSession:
    world: Any
    robot: Any
    target_indices: np.ndarray
    arm_indices: np.ndarray
    render_product_high: Any
    render_product_wrist: Any
    rgb_high_annotator: Any
    rgb_wrist_annotator: Any
    ik_solver: Any
    lula_solver: Any


class DataRecorder:
    def __init__(self, save_dir: str | Path):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.episode_idx = self._get_next_episode_index()
        self.buffer: list[dict[str, Any]] = []

    def _get_next_episode_index(self) -> int:
        max_index = -1
        for path in self.save_dir.glob("episode_*.h5"):
            stem = path.stem
            parts = stem.split("_")
            if len(parts) != 2:
                continue
            try:
                max_index = max(max_index, int(parts[1]))
            except ValueError:
                continue
        return max_index + 1

    def start_episode(self) -> None:
        self.buffer = []

    def record_frame(
        self,
        *,
        img_high: np.ndarray,
        img_wrist: np.ndarray,
        obs_state: np.ndarray,
        action: np.ndarray,
        prompt: str,
    ) -> None:
        self.buffer.append(
            {
                "cam_high": np.asarray(img_high, dtype=np.uint8),
                "cam_wrist": np.asarray(img_wrist, dtype=np.uint8),
                "state": np.asarray(obs_state, dtype=np.float32),
                "action": np.asarray(action, dtype=np.float32),
                "prompt": prompt,
            }
        )

    def save_episode(self, prompt_text: str) -> Path:
        filename = self.save_dir / f"episode_{self.episode_idx:06d}.h5"
        with h5py.File(filename, "w") as f:
            f.attrs["prompt"] = prompt_text
            f.attrs["num_frames"] = len(self.buffer)
            f.create_dataset("cam_high", data=np.array([x["cam_high"] for x in self.buffer]), compression="gzip")
            f.create_dataset("cam_wrist", data=np.array([x["cam_wrist"] for x in self.buffer]), compression="gzip")
            f.create_dataset("state", data=np.array([x["state"] for x in self.buffer], dtype=np.float32))
            f.create_dataset("action", data=np.array([x["action"] for x in self.buffer], dtype=np.float32))
            dt = h5py.special_dtype(vlen=str)
            prompt_dset = f.create_dataset("prompt", (len(self.buffer),), dtype=dt)
            prompt_dset[:] = [x["prompt"] for x in self.buffer]
        self.episode_idx += 1
        return filename


def import_world_class() -> Any:
    try:
        from isaacsim.core.api import World
    except ModuleNotFoundError:
        from omni.isaac.core import World
    return World


def import_articulation_class() -> Any:
    try:
        from isaacsim.core.prims import SingleArticulation as Articulation
    except ModuleNotFoundError:
        try:
            from omni.isaac.core.articulations import Articulation
        except ModuleNotFoundError:
            from isaacsim.core.experimental.prims import Articulation
    return Articulation


def import_articulation_action() -> Any:
    try:
        from isaacsim.core.utils.types import ArticulationAction
    except ModuleNotFoundError:
        from omni.isaac.core.utils.types import ArticulationAction
    return ArticulationAction


def open_stage_path(usd_path: str) -> None:
    try:
        from isaacsim.core.utils.stage import open_stage
    except ModuleNotFoundError:
        from omni.isaac.core.utils.stage import open_stage
    open_stage(usd_path)


def create_ik_solver(robot: Any, robot_description_path: str, robot_urdf_path: str, ee_frame_name: str) -> tuple[Any, Any]:
    try:
        from omni.isaac.motion_generation import ArticulationKinematicsSolver
        from omni.isaac.motion_generation import LulaKinematicsSolver
    except ModuleNotFoundError:
        from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver
        from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver

    lula_solver = LulaKinematicsSolver(
        robot_description_path=robot_description_path,
        urdf_path=robot_urdf_path,
    )
    ik_solver = ArticulationKinematicsSolver(robot, lula_solver, ee_frame_name)
    return ik_solver, lula_solver


def create_scene_session() -> SceneSession:
    World = import_world_class()
    Articulation = import_articulation_class()

    open_stage_path(args.usd_path)
    try:
        world = World(stage_units_in_meters=1.0)
    except TypeError:
        world = World()

    robot = world.scene.add(Articulation(prim_path=args.robot_prim_path, name="firefighter"))
    world.reset()

    target_indices = []
    for joint_name in CONTROL_JOINT_NAMES:
        joint_index = robot.get_dof_index(joint_name)
        if joint_index is None:
            raise RuntimeError(f"Could not resolve joint {joint_name!r} in {args.robot_prim_path}")
        target_indices.append(int(joint_index))

    target_indices_np = np.array(target_indices, dtype=np.int32)
    arm_indices_np = target_indices_np[:6]

    render_product_high = rep.create.render_product(args.camera_high_path, (args.image_width, args.image_height))
    render_product_wrist = rep.create.render_product(args.camera_wrist_path, (args.image_width, args.image_height))
    rgb_high_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_high_annotator.attach(render_product_high)
    rgb_wrist_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_wrist_annotator.attach(render_product_wrist)

    ik_solver, lula_solver = create_ik_solver(
        robot,
        robot_description_path=args.robot_description_path,
        robot_urdf_path=args.robot_urdf_path,
        ee_frame_name=args.ee_frame_name,
    )

    return SceneSession(
        world=world,
        robot=robot,
        target_indices=target_indices_np,
        arm_indices=arm_indices_np,
        render_product_high=render_product_high,
        render_product_wrist=render_product_wrist,
        rgb_high_annotator=rgb_high_annotator,
        rgb_wrist_annotator=rgb_wrist_annotator,
        ik_solver=ik_solver,
        lula_solver=lula_solver,
    )


def destroy_scene_session(session: SceneSession) -> None:
    try:
        high_path = session.render_product_high.path if hasattr(session.render_product_high, "path") else str(session.render_product_high)
        wrist_path = session.render_product_wrist.path if hasattr(session.render_product_wrist, "path") else str(session.render_product_wrist)
        session.rgb_high_annotator.detach([high_path])
        session.rgb_wrist_annotator.detach([wrist_path])
        rep.orchestrator.step()
    except Exception as exc:
        print(f"[collector][WARN] failed to detach annotators cleanly: {exc}")


def get_stage() -> Usd.Stage:
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No USD stage is currently loaded")
    return stage


def get_prim_world_pose(prim_path: str) -> tuple[np.ndarray, np.ndarray]:
    stage = get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Invalid prim path: {prim_path}")

    xformable = UsdGeom.Xformable(prim)
    matrix = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    translation = np.array(matrix.ExtractTranslation(), dtype=np.float32)

    rotation = Gf.Transform(matrix).GetRotation().GetQuat()
    quat_xyzw = np.array(
        [
            float(rotation.GetImaginary()[0]),
            float(rotation.GetImaginary()[1]),
            float(rotation.GetImaginary()[2]),
            float(rotation.GetReal()),
        ],
        dtype=np.float32,
    )
    return translation, quat_xyzw


def capture_rgb(rgb_annotator: Any) -> np.ndarray:
    image = rgb_annotator.get_data()
    if image.shape[-1] == 4:
        image = image[..., :3]
    return np.asarray(image, dtype=np.uint8)


def interpolate_joint_positions(start: np.ndarray, goal: np.ndarray, num_steps: int) -> np.ndarray:
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    start = np.asarray(start, dtype=np.float32)
    goal = np.asarray(goal, dtype=np.float32)
    alphas = np.linspace(1.0 / num_steps, 1.0, num_steps, dtype=np.float32)[:, None]
    return start[None, :] + alphas * (goal - start)[None, :]


def build_pick_place_stage_targets(
    cube_position: np.ndarray,
    pallet_position: np.ndarray,
) -> list[StageTarget]:
    cube_above = cube_position.copy()
    cube_above[2] += np.float32(args.approach_height)

    grasp_position = cube_position.copy()
    grasp_position[2] += np.float32(args.grasp_height_offset)

    lifted_position = cube_position.copy()
    lifted_position[2] += np.float32(args.lift_height)

    pallet_above = pallet_position.copy()
    pallet_above[2] += np.float32(args.lift_height)

    place_position = pallet_position.copy()
    place_position[2] += np.float32(args.place_height_offset)

    retreat_position = pallet_position.copy()
    retreat_position[2] += np.float32(args.lift_height + 0.03)

    return [
        StageTarget("move_above_cube", cube_above, np.float32(args.finger_open), args.move_steps),
        StageTarget("move_to_grasp", grasp_position, np.float32(args.finger_open), args.move_steps),
        StageTarget("close_gripper", grasp_position, np.float32(args.finger_closed), args.gripper_steps),
        StageTarget("lift_cube", lifted_position, np.float32(args.finger_closed), args.move_steps),
        StageTarget("move_above_pallet", pallet_above, np.float32(args.finger_closed), args.move_steps),
        StageTarget("move_to_place", place_position, np.float32(args.finger_closed), args.move_steps),
        StageTarget("open_gripper", place_position, np.float32(args.finger_open), args.gripper_steps),
        StageTarget("retreat", retreat_position, np.float32(args.finger_open), args.move_steps),
    ]


def get_tool_offset_vector() -> np.ndarray:
    return np.array([args.tool_offset_x, args.tool_offset_y, args.tool_offset_z], dtype=np.float32)


def apply_tool_offset(target_position: np.ndarray, ee_orientation_xyzw: np.ndarray) -> np.ndarray:
    offset = get_tool_offset_vector()
    if not np.any(offset):
        return np.asarray(target_position, dtype=np.float32)

    quat = Gf.Quatf(
        float(ee_orientation_xyzw[3]),
        Gf.Vec3f(
            float(ee_orientation_xyzw[0]),
            float(ee_orientation_xyzw[1]),
            float(ee_orientation_xyzw[2]),
        ),
    )
    rotation = Gf.Rotation(quat)
    rotated_offset = rotation.TransformDir(
        Gf.Vec3d(float(offset[0]), float(offset[1]), float(offset[2]))
    )

    return np.asarray(target_position, dtype=np.float32) - np.array(
        [rotated_offset[0], rotated_offset[1], rotated_offset[2]],
        dtype=np.float32,
    )



def warm_up_robot(session: SceneSession) -> None:
    ArticulationAction = import_articulation_action()
    start_positions = session.robot.get_joint_positions()[session.target_indices].astype(np.float32)
    trajectory = interpolate_joint_positions(start_positions, TRAIN_INIT_STATE, args.warmup_steps)
    for target in trajectory:
        session.robot.apply_action(ArticulationAction(joint_positions=target, joint_indices=session.target_indices))
        session.world.step(render=True)
    for _ in range(30):
        session.robot.apply_action(ArticulationAction(joint_positions=TRAIN_INIT_STATE, joint_indices=session.target_indices))
        session.world.step(render=False)


def read_state(session: SceneSession) -> np.ndarray:
    return session.robot.get_joint_positions()[session.target_indices].astype(np.float32)


def solve_arm_target(session: SceneSession, target_position: np.ndarray, target_orientation: np.ndarray) -> np.ndarray:
    robot_world_pos, robot_world_quat = session.robot.get_world_pose()
    session.lula_solver.set_robot_base_pose(
        np.asarray(robot_world_pos, dtype=np.float32),
        np.asarray(robot_world_quat, dtype=np.float32),
    )

    ik_action, success = session.ik_solver.compute_inverse_kinematics(
        target_position=np.asarray(target_position, dtype=np.float32),
        target_orientation=np.asarray(target_orientation, dtype=np.float32),
    )
    if not success:
        raise RuntimeError(f"IK failed for target position {target_position.tolist()}")

    joint_positions = np.asarray(ik_action.joint_positions, dtype=np.float32)
    if joint_positions.shape[0] == session.robot.num_dof:
        return joint_positions[session.arm_indices]
    if joint_positions.shape[0] == len(session.arm_indices):
        return joint_positions
    raise RuntimeError(
        f"Unexpected IK output size {joint_positions.shape[0]}, expected {session.robot.num_dof} or {len(session.arm_indices)}"
    )


def apply_and_record(
    session: SceneSession,
    recorder: DataRecorder,
    commanded_target: np.ndarray,
    prompt: str,
) -> None:
    ArticulationAction = import_articulation_action()
    session.robot.apply_action(
        ArticulationAction(
            joint_positions=commanded_target.astype(np.float32),
            joint_indices=session.target_indices,
        )
    )
    session.world.step(render=True)
    rep.orchestrator.step()

    recorder.record_frame(
        img_high=capture_rgb(session.rgb_high_annotator),
        img_wrist=capture_rgb(session.rgb_wrist_annotator),
        obs_state=read_state(session),
        action=commanded_target,
        prompt=prompt,
    )


def run_episode(session: SceneSession, recorder: DataRecorder, prompt: str) -> Path:
    recorder.start_episode()
    warm_up_robot(session)

    cube_position, _ = get_prim_world_pose(args.cube_prim_path)
    pallet_position, _ = get_prim_world_pose(args.pallet_prim_path)
    _, ee_orientation = get_prim_world_pose(args.ee_prim_path)
    robot_world_pos, robot_world_quat = session.robot.get_world_pose()

    print(f"[collector][DEBUG] robot world pos: {np.asarray(robot_world_pos, dtype=np.float32).tolist()}")
    print(f"[collector][DEBUG] robot world quat: {np.asarray(robot_world_quat, dtype=np.float32).tolist()}")
    print(f"[collector][DEBUG] cube world pos: {cube_position.tolist()}")
    print(f"[collector][DEBUG] pallet world pos: {pallet_position.tolist()}")
    print(f"[collector][DEBUG] ee orientation xyzw: {ee_orientation.tolist()}")
    print(f"[collector][DEBUG] tool offset xyz: {get_tool_offset_vector().tolist()}")

    stage_targets = build_pick_place_stage_targets(cube_position, pallet_position)
    current_command = read_state(session)

    for stage_target in stage_targets:
        ik_target_position = apply_tool_offset(stage_target.position, ee_orientation)
        print(
            f"[collector][DEBUG] stage={stage_target.name} "
            f"raw_target={stage_target.position.tolist()} "
            f"ik_target={ik_target_position.tolist()} "
            f"finger={float(stage_target.finger_joint_target):.6f}"
        )

        arm_goal = solve_arm_target(session, ik_target_position, ee_orientation)
        joint_goal = current_command.copy()
        joint_goal[:6] = arm_goal
        joint_goal[6] = stage_target.finger_joint_target

        trajectory = interpolate_joint_positions(current_command, joint_goal, stage_target.num_steps)
        for target in trajectory:
            apply_and_record(session, recorder, target.astype(np.float32), prompt)

        current_command = joint_goal
        for _ in range(args.settle_steps):
            apply_and_record(session, recorder, current_command.astype(np.float32), prompt)

    return recorder.save_episode(prompt)


def main() -> None:
    recorder = DataRecorder(args.dataset_dir)
    print(f"[collector] saving to: {args.dataset_dir}")
    print(f"[collector] next episode index: {recorder.episode_idx:06d}")

    for episode_idx in range(args.num_episodes):
        print(f"[collector] starting episode {episode_idx + 1}/{args.num_episodes}")
        session = create_scene_session()
        try:
            saved_path = run_episode(session, recorder, args.prompt)
            print(f"[collector] saved episode to: {saved_path}")
        finally:
            destroy_scene_session(session)


if __name__ == "__main__":
    import traceback

    try:
        main()
    except Exception as exc:
        print(f"[collector][ERROR] {exc}")
        traceback.print_exc()
        raise
    finally:
        if simulation_app.is_running():
            simulation_app.close()
