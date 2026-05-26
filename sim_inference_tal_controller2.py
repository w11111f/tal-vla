from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping
import contextlib
import dataclasses
import json
import math
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import threading
import time
import traceback
from typing import Any

import cv2
import numpy as np
from PIL import Image


parser = argparse.ArgumentParser(description="Isaac Sim TAL + OpenPI closed-loop controller")
parser.add_argument("--prompt", type=str, default="pick up the block", help="The language instruction for the robot")
parser.add_argument("--server-host", type=str, default="127.0.0.1", help="OpenPI policy server host")
parser.add_argument("--server-port", type=int, default=8000, help="OpenPI policy server port")
parser.add_argument("--tal-root", type=str, required=True, help="Path to TAL2 repo root")
parser.add_argument("--qwen-model", type=str, default="qwen3-max", help="DashScope model name used by TAL")
parser.add_argument("--qwen-api-key-env", type=str, default="DASHSCOPE_API_KEY", help="Env var storing DashScope key")
parser.add_argument("--manual-scene-graph-json", type=str, default="", help="Optional JSON file path for scene graph")
parser.add_argument("--replan-every-n-steps", type=int, default=3000, help="Replan every N control steps")
parser.add_argument("--max-steps", type=int, default=-1, help="Maximum control loop steps; -1 means unlimited")
parser.add_argument(
    "--tal-world-state-name",
    type=str,
    default="Initialize",
    help='Initial TAL scene graph state token for debug, for example "Initialize"',
)
parser.add_argument("--nav-control-dt", type=float, default=0.05, help="Navigation bridge/control period in seconds")
parser.add_argument("--nav-goal-timeout-sec", type=float, default=120.0, help="Timeout for a single Nav2 goal")
parser.add_argument(
    "--nav-warmup-sec",
    type=float,
    default=4.0,
    help="Seconds to advance simulation after starting the Nav2 bridge so /clock, /odom, and /tf are live",
)
parser.add_argument(
    "--wheel-self-test",
    action="store_true",
    default=False,
    help="Run a direct wheel-velocity self test and exit before TAL/OpenPI/Nav2 closed-loop control",
)
parser.add_argument(
    "--wheel-self-test-duration-sec",
    type=float,
    default=5.0,
    help="Duration for the direct wheel-velocity self test",
)
parser.add_argument(
    "--wheel-self-test-left-rad-s",
    type=float,
    default=2.0,
    help="Left wheel angular velocity used in wheel self test",
)
parser.add_argument(
    "--wheel-self-test-right-rad-s",
    type=float,
    default=2.0,
    help="Right wheel angular velocity used in wheel self test",
)
parser.add_argument("--headless", action="store_true", default=False, help="Run Isaac Sim in headless mode")
args, unknown_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + unknown_args


CAMERA_HIGH_PATH = "/World/high"
CAMERA_WRIST_PATH = "/World/Mobie_grasper2/firefighter/joint6/wrist"
TRAIN_INIT_STATE = np.array(
    [-0.12466581, -0.15327631, 1.2, -0.1757595, 1.5070096, -0.320009, 0.13824108],
    dtype=np.float32,
)
JOINT_NAMES_IN_ORDER = [
    "joint1_to_base",
    "joint2_to_joint1",
    "joint3_to_joint2",
    "joint4_to_joint3",
    "joint5_to_joint4",
    "joint6_to_joint5",
    "finger_joint",
]
NAV_MAP_YAML_PATH = Path(
    os.environ.get(
        "TAL_NAV_MAP_YAML",
        "/root/gpufree-data/code/robot_ws/src/robot_navigation/maps/expff_map.yaml",
    )
)


_TRACE_T0 = time.monotonic()


def trace_phase(message: str) -> None:
    elapsed = time.monotonic() - _TRACE_T0
    print(f"[TRACE +{elapsed:7.3f}s] {message}", flush=True)


@dataclasses.dataclass
class TALPlanResult:
    status: str
    first_action_text: str | None
    predicted_actions: list[Any]
    current_scene_graph_json: dict[str, Any] | None = None
    goal_scene_graph_json: dict[str, Any] | None = None
    error: str | None = None


@dataclasses.dataclass
class NavigationGoal:
    x: float
    y: float
    yaw: float = 0.0
    frame_id: str = "map"


@dataclasses.dataclass
class NavOccupancyMap:
    resolution: float
    origin_x: float
    origin_y: float
    image: np.ndarray

    def world_to_grid(self, x: float, y: float) -> tuple[int, int] | None:
        height, width = self.image.shape
        ix = int((x - self.origin_x) / self.resolution)
        iy = height - 1 - int((y - self.origin_y) / self.resolution)
        if ix < 0 or iy < 0 or ix >= width or iy >= height:
            return None
        return ix, iy

    def is_free(self, x: float, y: float) -> bool:
        grid = self.world_to_grid(x, y)
        if grid is None:
            return False
        ix, iy = grid
        return int(self.image[iy, ix]) >= 250

    def has_clearance(self, x: float, y: float, radius_m: float) -> bool:
        grid = self.world_to_grid(x, y)
        if grid is None:
            return False
        ix, iy = grid
        radius_cells = max(int(math.ceil(radius_m / self.resolution)), 0)
        height, width = self.image.shape
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                x2 = ix + dx
                y2 = iy + dy
                if x2 < 0 or y2 < 0 or x2 >= width or y2 >= height:
                    return False
                if int(self.image[y2, x2]) < 250:
                    return False
        return True


@dataclasses.dataclass
class PendingNavigation:
    goal: NavigationGoal
    accepted: bool = False
    success: bool = False
    status: int | None = None
    error: str | None = None
    goal_handle: Any | None = None
    done_event: threading.Event = dataclasses.field(default_factory=threading.Event)


@dataclasses.dataclass
class ParsedTALSubtask:
    name: str
    args: list[str]
    text: str | None = None
    raw: Any | None = None

    @property
    def is_navigation(self) -> bool:
        return self.name.lower() == "moveto" and len(self.args) >= 1


@dataclasses.dataclass
class TALControllerConfig:
    tal_root: str
    qwen_model: str = "qwen3-max"
    qwen_api_key_env: str = "DASHSCOPE_API_KEY"
    candidate_action_num: int = 20
    select_from_candidate: int = 10
    max_planning_steps: int = 60
    headless: bool = False


@dataclasses.dataclass
class TALRuntimeContext:
    tal_root: Path
    sim_env_config: Any
    planner_env_config: Any
    approx: Any
    isaac_env: Any
    scene_graph_translator: Any
    plan_with_natural_language_instruction: Any
    scene_graph_json_to_dgl: Any
    model_action: Any
    model_action_effect: Any
    action_effect_features: Any
    simulation_app: Any
    qwen_model: str
    qwen_api_key_env: str
    candidate_action_num: int
    select_from_candidate: int
    max_planning_steps: int

    def close(self) -> None:
        try:
            self.approx.close_backend()
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed to close TAL planner backend cleanly: {exc}")
        try:
            self.isaac_env.destroy()
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed to close TAL Isaac backend cleanly: {exc}")


@contextlib.contextmanager
def pushd(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def build_fused_prompt(original_instruction: str, tal_first_action: str | None) -> str:
    if not tal_first_action:
        return original_instruction
    return f"User task: {original_instruction.strip()}.\nCurrent subtask: {tal_first_action.strip()}."


def format_tal_action(action: Any) -> str | None:
    if action is None:
        return None
    if isinstance(action, str):
        return action.strip()
    if isinstance(action, Mapping):
        name = str(action.get("name", "")).strip()
        args = action.get("args", [])
        if not isinstance(args, list):
            args = [args]
        args_text = ", ".join(str(arg) for arg in args if str(arg).strip())
        if name and args_text:
            return f"{name}({args_text})"
        if name:
            return name
        return json.dumps(action, ensure_ascii=False)
    if isinstance(action, (list, tuple)):
        return ", ".join(str(item) for item in action)
    return str(action).strip()


def _normalize_tal_arg(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().strip("\"'")


def parse_tal_subtask(action: Any) -> ParsedTALSubtask | None:
    if action is None:
        return None
    if isinstance(action, Mapping):
        name = str(action.get("name", "")).strip()
        args_value = action.get("args", [])
        if not isinstance(args_value, list):
            args_value = [args_value]
        args = [_normalize_tal_arg(arg) for arg in args_value if _normalize_tal_arg(arg)]
        if not name:
            return None
        return ParsedTALSubtask(name=name, args=args, text=format_tal_action(action), raw=action)

    action_text = format_tal_action(action)
    if not action_text:
        return None

    match = re.fullmatch(r"\s*([A-Za-z_][A-Za-z0-9_]*)\((.*)\)\s*", action_text)
    if match:
        name = match.group(1)
        args_blob = match.group(2).strip()
        args = [_normalize_tal_arg(item) for item in args_blob.split(",") if _normalize_tal_arg(item)]
        return ParsedTALSubtask(name=name, args=args, text=action_text, raw=action)

    return ParsedTALSubtask(name=action_text.strip(), args=[], text=action_text, raw=action)


def derive_executable_subtask(parsed_subtask: ParsedTALSubtask | None) -> ParsedTALSubtask | None:
    if parsed_subtask is None:
        return None

    name = parsed_subtask.name.lower()
    if name in {"picknplaceaonb", "pushto"} and parsed_subtask.args:
        nav_target = parsed_subtask.args[0]
        return ParsedTALSubtask(
            name="moveTo",
            args=[nav_target],
            text=f"moveTo({nav_target})",
            raw={
                "name": "moveTo",
                "args": [nav_target],
                "derived_from": parsed_subtask.raw if parsed_subtask.raw is not None else parsed_subtask.text,
            },
        )

    return parsed_subtask


def yaw_to_quaternion(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)], dtype=np.float32)


def quaternion_to_yaw(quaternion: np.ndarray | list[float] | tuple[float, ...] | None) -> float:
    if quaternion is None:
        return 0.0
    q = np.asarray(quaternion, dtype=np.float32).reshape(-1)
    if q.size != 4:
        return 0.0
    w, x, y, z = [float(v) for v in q]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _to_abs_repo_path(repo_root: Path, maybe_relative: str) -> str:
    path = Path(maybe_relative)
    if path.is_absolute():
        return str(path)
    return str((repo_root / path).resolve())


def _build_env_config(tal_root: Path, init_args: Any, EnvironmentConfig: Any, *, policy_backend: str, qwen_model: str, qwen_api_key_env: str) -> Any:
    with pushd(tal_root):
        tal_args = init_args()
        tal_args.exec_type = "policy"
        tal_args.policy_backend = policy_backend
        tal_args.qwen_model = qwen_model

        import torch

        tal_args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tal_args.qwen_api_key = os.getenv(qwen_api_key_env) if qwen_api_key_env else None
        if getattr(tal_args, "data_dir", None):
            tal_args.data_dir = _to_abs_repo_path(tal_root, tal_args.data_dir)

        env_config = EnvironmentConfig(tal_args)

    env_config.MODEL_SAVE_PATH = _to_abs_repo_path(tal_root, env_config.MODEL_SAVE_PATH)
    env_config.Aall_path = _to_abs_repo_path(tal_root, env_config.Aall_path)
    env_config.all_possible_actions_path = _to_abs_repo_path(tal_root, env_config.all_possible_actions_path)
    return env_config


def _load_required_model(env_config: Any, get_model: Any, load_model: Any, model_name: str) -> Any:
    model = get_model(env_config, model_name, env_config.features_dim, env_config.num_objects)
    seq_prefix = "Seq_" if env_config.training == "gcn_seq" else ""
    stable_ckpt = Path(env_config.MODEL_SAVE_PATH) / f"{seq_prefix}{model.name}_Trained.ckpt"
    ckpt_path = stable_ckpt if stable_ckpt.exists() else None
    if ckpt_path is None:
        model_dir = Path(env_config.MODEL_SAVE_PATH)
        best_epoch = -1
        for filename in model_dir.iterdir():
            if not filename.name.startswith(seq_prefix + model.name + "_") or filename.suffix != ".ckpt":
                continue
            try:
                epoch = int(filename.stem.rsplit("_", 1)[-1])
            except ValueError:
                continue
            if epoch > best_epoch:
                best_epoch = epoch
                ckpt_path = filename
    if ckpt_path is None:
        raise FileNotFoundError(f"Could not find checkpoint for TAL model {model.name}")
    model, _, _, _ = load_model(env_config, seq_prefix + model.name + "_Trained", model, file_path=str(ckpt_path))
    return model.to(env_config.device)


def initialize_tal_runtime(config: TALControllerConfig) -> TALRuntimeContext:
    trace_phase(f"initialize_tal_runtime: start (tal_root={config.tal_root}, headless={config.headless})")
    tal_root = Path(config.tal_root).resolve()
    if not (tal_root / "src").exists():
        raise FileNotFoundError(f"Invalid TAL root: {tal_root}")

    os.environ["TAL_ISAAC_HEADLESS"] = "1" if config.headless else "0"

    if str(tal_root) not in sys.path:
        sys.path.insert(0, str(tal_root))

    trace_phase("initialize_tal_runtime: importing TAL modules")
    tal_config_module = __import__("src.config.config", fromlist=["init_args"])
    env_constants_module = __import__("src.envs.CONSTANTS", fromlist=["EnvironmentConfig"])
    planning_module = __import__("src.tal.utils_planning", fromlist=["plan_with_natural_language_instruction"])
    training_module = __import__("src.tal.utils_training", fromlist=["get_model", "load_model"])
    translator_module = __import__(
        "src.tal.scene_graph_translator",
        fromlist=["scene_graph_json_to_dgl", "datapoint_to_scene_graph_json"],
    )
    approx_module = __import__("src.envs.approx", fromlist=["initPolicy", "close_backend"])
    isaac_env_module = __import__("src.envs.isaac_env", fromlist=["start", "getDatapoint", "simulation_app"])

    init_args = tal_config_module.init_args
    EnvironmentConfig = env_constants_module.EnvironmentConfig
    plan_with_natural_language_instruction = planning_module.plan_with_natural_language_instruction
    scene_graph_json_to_dgl = translator_module.scene_graph_json_to_dgl
    get_model = training_module.get_model
    load_model = training_module.load_model

    trace_phase("initialize_tal_runtime: building env configs")
    sim_env_config = _build_env_config(
        tal_root,
        init_args,
        EnvironmentConfig,
        policy_backend="isaaclab",
        qwen_model=config.qwen_model,
        qwen_api_key_env=config.qwen_api_key_env,
    )
    planner_env_config = _build_env_config(
        tal_root,
        init_args,
        EnvironmentConfig,
        policy_backend="symbolic",
        qwen_model=config.qwen_model,
        qwen_api_key_env=config.qwen_api_key_env,
    )

    import pickle

    trace_phase("initialize_tal_runtime: loading TAL models")
    model_action_effect = _load_required_model(planner_env_config, get_model, load_model, "AFE")
    model_action = _load_required_model(planner_env_config, get_model, load_model, "APN")
    features_save_path = Path(planner_env_config.MODEL_SAVE_PATH) / "action_effect_features_avg.pkl"
    with features_save_path.open("rb") as file_obj:
        action_effect_features = pickle.load(file_obj)

    world_num = 0
    graph_world_name = getattr(sim_env_config, "graph_world_name", "")
    digits = "".join(ch for ch in str(graph_world_name) if ch.isdigit())
    if digits:
        world_num = int(digits)

    trace_phase(f"initialize_tal_runtime: approx.initPolicy(world_num={world_num})")
    approx_module.initPolicy(
        sim_env_config,
        sim_env_config.domain,
        goal_json=None,
        world_num=world_num,
        SET_GAOL_JSON=False,
    )

    trace_phase("initialize_tal_runtime: complete")
    return TALRuntimeContext(
        tal_root=tal_root,
        sim_env_config=sim_env_config,
        planner_env_config=planner_env_config,
        approx=approx_module,
        isaac_env=isaac_env_module,
        scene_graph_translator=translator_module,
        plan_with_natural_language_instruction=plan_with_natural_language_instruction,
        scene_graph_json_to_dgl=scene_graph_json_to_dgl,
        model_action=model_action,
        model_action_effect=model_action_effect,
        action_effect_features=action_effect_features,
        simulation_app=isaac_env_module.simulation_app,
        qwen_model=config.qwen_model,
        qwen_api_key_env=config.qwen_api_key_env,
        candidate_action_num=config.candidate_action_num,
        select_from_candidate=config.select_from_candidate,
        max_planning_steps=config.max_planning_steps,
    )


class TALSceneGraphProvider:
    def __init__(self, runtime_ctx: TALRuntimeContext):
        self._runtime = runtime_ctx

    def _refresh_live_datapoint(self) -> Any:
        isaac_env = self._runtime.isaac_env
        isaac_env.update_metrics()
        isaac_env.resetDatapoint(self._runtime.sim_env_config)
        isaac_env.initRootNode()
        return isaac_env.getDatapoint(self._runtime.sim_env_config)

    def get_current_scene_graph(
        self,
        *,
        state_name: str | None = None,
        manual_scene_graph: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], Any | None]:
        if manual_scene_graph is not None:
            return manual_scene_graph, None

        if state_name is None:
            datapoint = self._refresh_live_datapoint()
        else:
            self._runtime.isaac_env.update_metrics()
            datapoint = self._runtime.isaac_env.getDatapoint(self._runtime.sim_env_config)

        scene_graph = self._runtime.scene_graph_translator.datapoint_to_scene_graph_json(
            self._runtime.sim_env_config,
            datapoint,
            state_name=state_name,
        )
        return scene_graph, datapoint


class LazyTALPlanner:
    def __init__(self, runtime_ctx: TALRuntimeContext):
        self._runtime = runtime_ctx
        self._plan_lock = threading.Lock()

    @staticmethod
    def _summarize_scene_graph(scene_graph: Mapping[str, Any]) -> str:
        node_count = len(scene_graph.get("nodes", [])) if isinstance(scene_graph, Mapping) else -1
        edge_count = len(scene_graph.get("edges", [])) if isinstance(scene_graph, Mapping) else -1
        return f"nodes={node_count}, edges={edge_count}"

    def plan_first_action(
        self,
        user_instruction: str,
        current_scene_graph_json: Mapping[str, Any],
        start_node: Any | None = None,
    ) -> TALPlanResult:
        with self._plan_lock:
            planner_config = self._runtime.planner_env_config
            plan_begin = time.perf_counter()
            print(
                "[TALTrace] plan_first_action start | "
                f"instruction={user_instruction!r} | "
                f"{self._summarize_scene_graph(current_scene_graph_json)} | "
                f"start_node_type={type(start_node).__name__ if start_node is not None else 'None'}"
            )

            graph_build_begin = time.perf_counter()
            current_state_graph = self._runtime.scene_graph_json_to_dgl(planner_config, dict(current_scene_graph_json))
            current_state_graph = current_state_graph.to(planner_config.device)
            graph_build_elapsed = time.perf_counter() - graph_build_begin
            print(
                "[TALTrace] scene_graph_json_to_dgl done | "
                f"elapsed={graph_build_elapsed:.3f}s | device={planner_config.device}"
            )

            world_num = 0
            graph_world_name = getattr(planner_config, "graph_world_name", "")
            digits = "".join(ch for ch in str(graph_world_name) if ch.isdigit())
            if digits:
                world_num = int(digits)

            planner_call_begin = time.perf_counter()
            qwen_api_key = os.getenv(self._runtime.qwen_api_key_env) if self._runtime.qwen_api_key_env else None
            print(
                "[TALTrace] planner call begin | "
                f"world_num={world_num} | qwen_model={self._runtime.qwen_model} | "
                f"candidate_action_num={self._runtime.candidate_action_num} | "
                f"trajectory_length={self._runtime.max_planning_steps} | "
                f"select_from_candidate={self._runtime.select_from_candidate} | "
                f"api_key_present={bool(qwen_api_key)}"
            )
            try:
                result = self._runtime.plan_with_natural_language_instruction(
                    planner_config,
                    model_action=self._runtime.model_action,
                    model_extract_feature=self._runtime.model_action_effect,
                    action_effect_features=self._runtime.action_effect_features,
                    instruction=user_instruction,
                    world_num=world_num,
                    start_node=start_node,
                    current_state_graph=current_state_graph,
                    current_scene_graph_json=dict(current_scene_graph_json),
                    qwen_model_name=self._runtime.qwen_model,
                    qwen_api_key=qwen_api_key,
                    candidate_action_num=self._runtime.candidate_action_num,
                    select_from_candidate=self._runtime.select_from_candidate,
                    trajectory_length=self._runtime.max_planning_steps,
                    with_pca=True,
                )
            except Exception as exc:
                planner_call_elapsed = time.perf_counter() - planner_call_begin
                print(
                    "[TALTrace] planner call raised | "
                    f"elapsed={planner_call_elapsed:.3f}s | "
                    f"exc_type={type(exc).__name__} | exc={exc}"
                )
                raise
            planner_call_elapsed = time.perf_counter() - planner_call_begin
            print(
                "[TALTrace] planner call returned | "
                f"elapsed={planner_call_elapsed:.3f}s | "
                f"result_type={type(result).__name__}"
            )

        if not isinstance(result, Mapping):
            raise TypeError(f"TAL planner returned unsupported type: {type(result).__name__}")

        print(
            "[TALTrace] planner result summary | "
            f"keys={sorted(result.keys())} | "
            f"status={result.get('status')} | "
            f"predicted_actions_len={len(result.get('predicted_actions', [])) if isinstance(result.get('predicted_actions', []), list) else 'n/a'} | "
            f"error={result.get('error')!r}"
        )
        goal_scene_graph_json = result.get("goal_scene_graph_json")
        if isinstance(goal_scene_graph_json, Mapping):
            print(
                "[TALTrace] planner goal scene graph summary | "
                f"{self._summarize_scene_graph(goal_scene_graph_json)}"
            )
            print(
                "[TALTrace] planner goal scene graph json: "
                f"{json.dumps(goal_scene_graph_json, ensure_ascii=False)}"
            )
        print(f"[TALTrace] plan_first_action total_elapsed={time.perf_counter() - plan_begin:.3f}s")

        predicted_actions = list(result.get("predicted_actions", []))
        first_action = format_tal_action(predicted_actions[0]) if predicted_actions else None
        return TALPlanResult(
            status=result.get("status", "Unknown"),
            first_action_text=first_action,
            predicted_actions=predicted_actions,
            current_scene_graph_json=result.get("current_scene_graph_json"),
            goal_scene_graph_json=result.get("goal_scene_graph_json"),
            error=result.get("error"),
        )


class IsaacNavBridge:
    def __init__(
        self,
        robot: Any,
        robot_root_controller: RobotRootPoseController,
        runtime_ctx: TALRuntimeContext,
        ArticulationAction: Any,
    ):
        self._robot = robot
        self._robot_root_controller = robot_root_controller
        self._runtime_ctx = runtime_ctx
        self._ArticulationAction = ArticulationAction
        self._cmd_vx = 0.0
        self._cmd_vw = 0.0
        self._applied_vx = 0.0
        self._applied_vw = 0.0
        self._sim_time_s = 0.0
        self._bridge_runner: Any | None = None
        self._bridge_thread: threading.Thread | None = None
        self._bridge_thread_error: BaseException | None = None
        self._bridge_initialized_rclpy = False
        self._scan_angle_min = -math.pi / 2.0
        self._scan_angle_max = math.pi / 2.0
        self._scan_angle_increment = math.pi / 180.0
        self._scan_range_min = 0.12
        self._scan_range_max = 3.5
        self._max_linear_accel = 0.6
        self._max_angular_accel = 1.5
        self._wheel_radius_m = 0.165
        self._track_width_m = 0.55
        self._use_kinematic_base = os.environ.get("TAL_NAV_KINEMATIC_BASE", "1").lower() not in {
            "0",
            "false",
            "no",
        }
        self._left_wheel_joint_sign = float(os.environ.get("TAL_NAV_LEFT_WHEEL_SIGN", "1.0"))
        self._right_wheel_joint_sign = float(os.environ.get("TAL_NAV_RIGHT_WHEEL_SIGN", "1.0"))
        self._root_yaw_offset = float(
            os.environ.get(
                "TAL_ROBOT_ROOT_YAW_OFFSET",
                str(getattr(runtime_ctx.sim_env_config, "robot_root_yaw_offset", 0.0)),
            )
        )
        self._active_goal: NavigationGoal | None = None
        self._diag_last_time_s = 0.0
        self._diag_last_pose = None
        self._diag_last_articulation_pose = None
        self._diag_last_goal_distance_m = None
        self._state_lock = threading.Lock()
        self._heartbeat_dt = max(float(os.environ.get("TAL_NAV_HEARTBEAT_DT", "0.05")), 0.01)
        self._closed_event = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._last_scan_ranges: list[float] = []

        self._cmd_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._cmd_socket.bind(("127.0.0.1", 0))
        self._cmd_socket.setblocking(False)
        self._cmd_port = int(self._cmd_socket.getsockname()[1])

        state_probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        state_probe.bind(("127.0.0.1", 0))
        self._state_port = int(state_probe.getsockname()[1])
        state_probe.close()
        self._state_addr = ("127.0.0.1", self._state_port)

        self._start_bridge_process()

        self._wheel_indices = self._detect_wheel_indices()
        print(f"[NavBridge] kinematic base mode={self._use_kinematic_base}")
        print(f"[NavBridge] root visual yaw offset={self._root_yaw_offset:.3f} rad")
        print(
            "[NavBridge] wheel joint signs "
            f"left={self._left_wheel_joint_sign:.1f} right={self._right_wheel_joint_sign:.1f}"
        )

        position, orientation = self._robot_root_controller.get_world_pose()
        if position is None:
            raise RuntimeError("Failed to read initial robot world pose for navigation bridge.")
        initial_root_yaw = quaternion_to_yaw(orientation)
        if abs(self._root_yaw_offset) > 1e-6:
            corrected_root_yaw = normalize_angle(initial_root_yaw + self._root_yaw_offset)
            self._robot_root_controller.set_world_pose(
                position=np.asarray(position, dtype=np.float32),
                orientation=yaw_to_quaternion(corrected_root_yaw),
            )
            print(
                "[NavBridge] corrected robot root orientation for visual front alignment: "
                f"base_yaw={initial_root_yaw:.3f} root_yaw={corrected_root_yaw:.3f}"
            )
            position, orientation = self._robot_root_controller.get_world_pose()

        self._xform_x = float(position[0])
        self._xform_y = float(position[1])
        self._xform_z = float(position[2])
        self._xform_yaw = normalize_angle(quaternion_to_yaw(orientation) - self._root_yaw_offset)
        self._diag_last_pose = (self._xform_x, self._xform_y, self._xform_yaw)

        articulation_position, articulation_orientation = self._robot.get_world_pose()
        if articulation_position is None:
            raise RuntimeError("Failed to read initial articulation world pose for navigation bridge.")
        articulation_position = np.asarray(articulation_position, dtype=np.float32)
        articulation_yaw = (
            normalize_angle(
                quaternion_to_yaw(np.asarray(articulation_orientation, dtype=np.float32)) - self._root_yaw_offset
            )
            if articulation_orientation is not None
            else 0.0
        )
        if self._use_kinematic_base:
            self._x = self._xform_x
            self._y = self._xform_y
            self._z = self._xform_z
            self._yaw = self._xform_yaw
        else:
            self._x = float(articulation_position[0])
            self._y = float(articulation_position[1])
            self._z = float(articulation_position[2])
            self._yaw = articulation_yaw
        self._diag_last_articulation_pose = (
            float(articulation_position[0]),
            float(articulation_position[1]),
            articulation_yaw,
        )
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name="isaac-nav-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        return normalize_angle(angle)

    @staticmethod
    def _step_towards(current: float, target: float, max_delta: float) -> float:
        if target > current:
            return min(current + max_delta, target)
        return max(current - max_delta, target)

    def _detect_wheel_indices(self) -> dict[str, int]:
        dof_names = list(getattr(self._robot, "dof_names", []))
        print(f"[NavBridge] robot.dof_names={dof_names}")

        if not dof_names:
            raise RuntimeError("Robot articulation exposed no DOF names; cannot configure wheel control.")

        normalized_names = [(idx, name, name.lower().replace("-", "_")) for idx, name in enumerate(dof_names)]

        def _find_index(*token_options: tuple[str, ...]) -> int:
            for tokens in token_options:
                candidates: list[tuple[int, str]] = []
                for idx, name, normalized in normalized_names:
                    if "wheel" not in normalized and "joint_" not in normalized:
                        continue
                    if all(token in normalized for token in tokens):
                        candidates.append((idx, name))
                if candidates:
                    idx, resolved_name = candidates[0]
                    print(f"[NavBridge] matched wheel {'+'.join(tokens)} -> {resolved_name} (index={idx})")
                    return idx
            raise RuntimeError(
                f"Failed to locate wheel joint with token options {token_options!r} in DOFs: {dof_names}"
            )

        wheel_indices = {
            "front_left": _find_index(("front", "left"), ("left", "front")),
            "front_right": _find_index(("front", "right"), ("right", "front")),
            "rear_left": _find_index(("rear", "left"), ("back", "left"), ("left", "back")),
            "rear_right": _find_index(("rear", "right"), ("back", "right"), ("right", "back")),
        }
        print(f"[NavBridge] wheel indices={wheel_indices}")
        return wheel_indices

    def _apply_wheel_velocities(self, left_rad_s: float, right_rad_s: float) -> None:
        left_joint_rad_s = self._left_wheel_joint_sign * float(left_rad_s)
        right_joint_rad_s = self._right_wheel_joint_sign * float(right_rad_s)
        joint_indices = np.array(
            [
                self._wheel_indices["front_left"],
                self._wheel_indices["front_right"],
                self._wheel_indices["rear_left"],
                self._wheel_indices["rear_right"],
            ],
            dtype=np.int32,
        )
        joint_velocities = np.array(
            [left_joint_rad_s, right_joint_rad_s, left_joint_rad_s, right_joint_rad_s],
            dtype=np.float32,
        )
        action_cmd = self._ArticulationAction(
            joint_velocities=joint_velocities,
            joint_indices=joint_indices,
        )
        self._robot.apply_action(action_cmd)

    def set_active_goal(self, goal: NavigationGoal | None) -> None:
        self._active_goal = goal
        self._diag_last_goal_distance_m = None

    def settle_to_goal_pose(self, goal: NavigationGoal) -> None:
        """Place the simulated base exactly on the accepted Nav2 goal pose."""
        yaw = self._normalize_angle(goal.yaw)
        self._cmd_vx = 0.0
        self._cmd_vw = 0.0
        self._applied_vx = 0.0
        self._applied_vw = 0.0
        self._x = float(goal.x)
        self._y = float(goal.y)
        self._yaw = yaw
        self._xform_x = self._x
        self._xform_y = self._y
        self._xform_yaw = yaw
        self._robot_root_controller.set_world_pose(
            position=np.array([self._x, self._y, self._xform_z], dtype=np.float32),
            orientation=yaw_to_quaternion(normalize_angle(yaw + self._root_yaw_offset)),
        )
        self._apply_wheel_velocities(0.0, 0.0)
        scan_ranges = self._compute_scan_ranges()
        with self._state_lock:
            self._last_scan_ranges = scan_ranges
            self._publish_locked(scan_ranges)
        print(
            "[NavBridge] settled final base pose to goal: "
            f"x={self._x:.3f} y={self._y:.3f} yaw={self._yaw:.3f} "
            f"root_yaw={normalize_angle(self._yaw + self._root_yaw_offset):.3f}",
            flush=True,
        )

    def _maybe_log_diagnostics(self, left_rad_s: float, right_rad_s: float) -> None:
        if self._diag_last_pose is None:
            self._diag_last_pose = (self._x, self._y, self._yaw)
            self._diag_last_time_s = self._sim_time_s
            return
        elapsed = self._sim_time_s - self._diag_last_time_s
        if elapsed < 1.0:
            return

        prev_x, prev_y, prev_yaw = self._diag_last_pose
        dx = self._xform_x - prev_x
        dy = self._xform_y - prev_y
        dyaw = self._normalize_angle(self._xform_yaw - prev_yaw)
        speed_m_s = math.hypot(dx, dy) / max(elapsed, 1e-6)
        yaw_rate_rad_s = dyaw / max(elapsed, 1e-6)

        articulation_dx = 0.0
        articulation_dy = 0.0
        articulation_dyaw = 0.0
        articulation_speed_m_s = 0.0
        articulation_yaw_rate_rad_s = 0.0
        articulation_position, articulation_orientation = self._robot.get_world_pose()
        if articulation_position is not None:
            articulation_position = np.asarray(articulation_position, dtype=np.float32)
            articulation_yaw = (
                normalize_angle(
                    quaternion_to_yaw(np.asarray(articulation_orientation, dtype=np.float32)) - self._root_yaw_offset
                )
                if articulation_orientation is not None
                else 0.0
            )
            if self._diag_last_articulation_pose is not None:
                prev_ax, prev_ay, prev_ayaw = self._diag_last_articulation_pose
                articulation_dx = float(articulation_position[0]) - prev_ax
                articulation_dy = float(articulation_position[1]) - prev_ay
                articulation_dyaw = self._normalize_angle(articulation_yaw - prev_ayaw)
                articulation_speed_m_s = math.hypot(articulation_dx, articulation_dy) / max(elapsed, 1e-6)
                articulation_yaw_rate_rad_s = articulation_dyaw / max(elapsed, 1e-6)
            self._diag_last_articulation_pose = (
                float(articulation_position[0]),
                float(articulation_position[1]),
                articulation_yaw,
            )

        goal_distance_str = "n/a"
        goal_delta_str = "n/a"
        if self._active_goal is not None:
            goal_distance = math.hypot(self._active_goal.x - self._x, self._active_goal.y - self._y)
            goal_distance_str = f"{goal_distance:.3f}"
            if self._diag_last_goal_distance_m is not None:
                goal_delta_str = f"{goal_distance - self._diag_last_goal_distance_m:+.3f}"
            self._diag_last_goal_distance_m = goal_distance

        print(
            "[NavDiag] "
            f"cmd_vx={self._cmd_vx:.3f} cmd_vw={self._cmd_vw:.3f} "
            f"applied_vx={self._applied_vx:.3f} applied_vw={self._applied_vw:.3f} "
            f"wheel_left={left_rad_s:.3f} wheel_right={right_rad_s:.3f} "
            f"xform_dx={dx:.3f} xform_dy={dy:.3f} xform_dyaw={dyaw:.3f} "
            f"xform_speed={speed_m_s:.3f} xform_yaw_rate={yaw_rate_rad_s:.3f} "
            f"art_dx={articulation_dx:.3f} art_dy={articulation_dy:.3f} art_dyaw={articulation_dyaw:.3f} "
            f"art_speed={articulation_speed_m_s:.3f} art_yaw_rate={articulation_yaw_rate_rad_s:.3f} "
            f"goal_dist={goal_distance_str} goal_delta={goal_delta_str}"
        )

        self._diag_last_pose = (self._xform_x, self._xform_y, self._xform_yaw)
        self._diag_last_time_s = self._sim_time_s

    def _object_scan_footprint(self, object_name: str) -> tuple[float, float]:
        config = self._runtime_ctx.sim_env_config
        overrides = getattr(config, "lidar_footprint_overrides", {})
        if object_name in overrides:
            width, depth = overrides[object_name]
            return max(float(width), 0.04), max(float(depth), 0.04)

        obj_entry = config.get_object_entry(object_name)
        if obj_entry is None:
            return 0.10, 0.10

        size = obj_entry.get("size", [0.10, 0.10, 0.10])
        width = min(max(float(size[0]), 0.04), 1.20)
        depth = min(max(float(size[1]), 0.04), 1.20)
        return width, depth

    def _compute_scan_ranges(self) -> list[float]:
        isaac_env = self._runtime_ctx.isaac_env
        config = self._runtime_ctx.sim_env_config
        isaac_env.update_metrics()
        metrics = isaac_env.metrics

        robot_xy = np.array([self._x, self._y], dtype=np.float32)
        beam_count = max(
            1,
            int(round((self._scan_angle_max - self._scan_angle_min) / self._scan_angle_increment)) + 1,
        )
        ranges = [self._scan_range_max] * beam_count

        for object_name in config.all_objects:
            if object_name == "husky" or object_name not in metrics:
                continue

            center = np.asarray(metrics[object_name][0][:2], dtype=np.float32)
            rel = center - robot_xy
            width, depth = self._object_scan_footprint(object_name)
            radius = 0.5 * math.hypot(width, depth)
            if radius <= 1e-6:
                continue

            center_dist = float(np.linalg.norm(rel))
            if center_dist > self._scan_range_max + radius:
                continue

            center_angle = math.atan2(float(rel[1]), float(rel[0])) - self._yaw
            center_angle = math.atan2(math.sin(center_angle), math.cos(center_angle))
            angular_padding = math.asin(min(radius / max(center_dist, radius), 1.0))

            beam_start = max(
                0,
                int(math.floor((center_angle - angular_padding - self._scan_angle_min) / self._scan_angle_increment)),
            )
            beam_end = min(
                beam_count - 1,
                int(math.ceil((center_angle + angular_padding - self._scan_angle_min) / self._scan_angle_increment)),
            )

            for beam_idx in range(beam_start, beam_end + 1):
                beam_angle = self._scan_angle_min + beam_idx * self._scan_angle_increment
                direction = np.array(
                    [math.cos(self._yaw + beam_angle), math.sin(self._yaw + beam_angle)],
                    dtype=np.float32,
                )
                proj = float(np.dot(rel, direction))
                if proj <= 0.0:
                    continue
                closest_sq = float(np.dot(rel, rel) - proj * proj)
                radius_sq = radius * radius
                if closest_sq >= radius_sq:
                    continue
                hit = proj - math.sqrt(max(radius_sq - closest_sq, 0.0))
                if self._scan_range_min <= hit < ranges[beam_idx]:
                    ranges[beam_idx] = hit

        return ranges

    @property
    def node(self) -> Any | None:
        return None

    def _start_bridge_process(self) -> None:
        import rclpy
        from isaac_nav_bridge_runner import IsaacNavBridgeRunner

        if not rclpy.ok():
            rclpy.init(args=None)
            self._bridge_initialized_rclpy = True

        self._bridge_runner = IsaacNavBridgeRunner("127.0.0.1", self._state_port, self._cmd_port)

        def _spin_bridge() -> None:
            try:
                assert self._bridge_runner is not None
                self._bridge_runner.spin()
            except BaseException as exc:  # noqa: BLE001
                self._bridge_thread_error = exc

        self._bridge_thread = threading.Thread(target=_spin_bridge, name="isaac-nav-bridge", daemon=True)
        self._bridge_thread.start()

    def _poll_cmd_vel(self) -> None:
        while True:
            try:
                packet, _ = self._cmd_socket.recvfrom(65535)
            except BlockingIOError:
                break
            except OSError:
                return
            payload = json.loads(packet.decode("utf-8"))
            self._cmd_vx = float(payload.get("vx", 0.0))
            self._cmd_vw = float(payload.get("vw", 0.0))

    def _heartbeat_loop(self) -> None:
        while not self._closed_event.wait(self._heartbeat_dt):
            with self._state_lock:
                self._sim_time_s += self._heartbeat_dt
                self._publish_locked(self._last_scan_ranges)

    def _check_bridge_process(self) -> None:
        if self._bridge_thread_error is not None:
            raise RuntimeError(f"In-process Isaac nav bridge failed: {self._bridge_thread_error}")
        if self._bridge_thread is None:
            return
        if self._bridge_thread.is_alive():
            return
        raise RuntimeError("In-process Isaac nav bridge thread exited unexpectedly.")

    def advance(self, dt: float) -> None:
        dt = max(float(dt), 1e-3)
        self._check_bridge_process()
        self._poll_cmd_vel()

        position, orientation = self._robot_root_controller.get_world_pose()
        if position is not None:
            self._xform_x = float(position[0])
            self._xform_y = float(position[1])
            self._xform_z = float(position[2])
        if orientation is not None:
            self._xform_yaw = normalize_angle(quaternion_to_yaw(orientation) - self._root_yaw_offset)

        articulation_position, articulation_orientation = self._robot.get_world_pose()
        if not self._use_kinematic_base:
            if articulation_position is not None:
                self._x = float(articulation_position[0])
                self._y = float(articulation_position[1])
                self._z = float(articulation_position[2])
            if articulation_orientation is not None:
                self._yaw = normalize_angle(quaternion_to_yaw(articulation_orientation) - self._root_yaw_offset)

        target_vx = float(self._cmd_vx)
        target_vw = float(self._cmd_vw)

        self._applied_vx = self._step_towards(
            self._applied_vx,
            target_vx,
            self._max_linear_accel * dt,
        )
        self._applied_vw = self._step_towards(
            self._applied_vw,
            target_vw,
            self._max_angular_accel * dt,
        )

        left_linear_m_s = self._applied_vx - 0.5 * self._track_width_m * self._applied_vw
        right_linear_m_s = self._applied_vx + 0.5 * self._track_width_m * self._applied_vw
        left_rad_s = left_linear_m_s / self._wheel_radius_m
        right_rad_s = right_linear_m_s / self._wheel_radius_m
        if self._use_kinematic_base:
            self._yaw = self._normalize_angle(self._yaw + self._applied_vw * dt)
            self._x += self._applied_vx * math.cos(self._yaw) * dt
            self._y += self._applied_vx * math.sin(self._yaw) * dt
            self._xform_x = self._x
            self._xform_y = self._y
            self._xform_yaw = self._yaw
            self._robot_root_controller.set_world_pose(
                position=np.array([self._x, self._y, self._xform_z], dtype=np.float32),
                orientation=yaw_to_quaternion(normalize_angle(self._yaw + self._root_yaw_offset)),
            )
            self._apply_wheel_velocities(0.0, 0.0)
        else:
            self._apply_wheel_velocities(left_rad_s, right_rad_s)
        scan_ranges = self._compute_scan_ranges()
        with self._state_lock:
            self._sim_time_s += dt
            self._last_scan_ranges = scan_ranges
            self._maybe_log_diagnostics(left_rad_s, right_rad_s)
            self._publish_locked(scan_ranges)

    def _publish_locked(self, scan_ranges: list[float]) -> None:
        payload = {
            "sim_time_s": self._sim_time_s,
            "x": self._x,
            "y": self._y,
            "z": self._z,
            "yaw": self._yaw,
            "vx": self._applied_vx,
            "vw": self._applied_vw,
            "scan_angle_min": self._scan_angle_min,
            "scan_angle_max": self._scan_angle_max,
            "scan_angle_increment": self._scan_angle_increment,
            "scan_range_min": self._scan_range_min,
            "scan_range_max": self._scan_range_max,
            "scan_ranges": scan_ranges,
        }
        packet = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        try:
            self._cmd_socket.sendto(packet, self._state_addr)
        except OSError:
            pass

    def publish(self) -> None:
        with self._state_lock:
            self._publish_locked(self._last_scan_ranges)

    def close(self) -> None:
        import rclpy

        self._closed_event.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=1.0)
        try:
            self._apply_wheel_velocities(0.0, 0.0)
        except Exception:
            pass
        if self._bridge_runner is not None:
            try:
                self._bridge_runner.close()
            except Exception:
                pass
            self._bridge_runner = None
        if self._bridge_thread is not None and self._bridge_thread.is_alive():
            self._bridge_thread.join(timeout=2.0)
        self._bridge_thread = None
        try:
            self._cmd_socket.close()
        except OSError:
            pass
        if self._bridge_initialized_rclpy and rclpy.ok():
            rclpy.shutdown()


class RobotRootPoseController:
    def __init__(self, prim: Any):
        self._prim = prim

    def get_world_pose(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        positions, orientations = self._prim.get_world_poses()
        if positions is None or len(positions) == 0:
            return None, None
        position = np.asarray(positions[0], dtype=np.float32)
        orientation = None
        if orientations is not None and len(orientations) > 0:
            orientation = np.asarray(orientations[0], dtype=np.float32)
        return position, orientation

    def set_world_pose(
        self,
        *,
        position: np.ndarray | None = None,
        orientation: np.ndarray | None = None,
    ) -> None:
        current_position, current_orientation = self.get_world_pose()
        if current_position is None:
            raise RuntimeError("Failed to read root prim world pose.")

        target_position = current_position if position is None else np.asarray(position, dtype=np.float32)
        target_orientation = current_orientation if orientation is None else np.asarray(orientation, dtype=np.float32)

        positions = target_position.reshape(1, 3)
        orientations = None if target_orientation is None else target_orientation.reshape(1, 4)
        self._prim.set_world_poses(positions=positions, orientations=orientations)

    def set_linear_velocity(self, _velocity: np.ndarray) -> None:
        return

    def set_angular_velocity(self, _velocity: np.ndarray) -> None:
        return


class AsyncNav2GoalClient:
    def __init__(self):
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from nav2_msgs.action import NavigateToPose
        from rclpy.action import ActionClient
        from rclpy.node import Node

        if not rclpy.ok():
            rclpy.init(args=None)
        self._PoseStamped = PoseStamped
        self._NavigateToPose = NavigateToPose
        self._node = Node("tal_nav2_goal_client")
        self._client = ActionClient(self._node, NavigateToPose, "navigate_to_pose")

    @property
    def node(self) -> Any:
        return self._node

    def wait_for_server(self, timeout_sec: float = 10.0) -> bool:
        return self._client.wait_for_server(timeout_sec=timeout_sec)

    def _build_goal_msg(self, goal: NavigationGoal) -> Any:
        goal_msg = self._NavigateToPose.Goal()
        pose = self._PoseStamped()
        pose.header.frame_id = goal.frame_id
        pose.pose.position.x = float(goal.x)
        pose.pose.position.y = float(goal.y)
        pose.pose.position.z = 0.0
        pose.pose.orientation.z = math.sin(goal.yaw / 2.0)
        pose.pose.orientation.w = math.cos(goal.yaw / 2.0)
        goal_msg.pose = pose
        return goal_msg

    def send_goal(self, goal: NavigationGoal) -> PendingNavigation:
        request = PendingNavigation(goal=goal)
        if not self.wait_for_server():
            request.error = "NavigateToPose action server is not available."
            request.done_event.set()
            return request

        goal_future = self._client.send_goal_async(self._build_goal_msg(goal))
        goal_future.add_done_callback(lambda future: self._on_goal_response(future, request))
        return request

    def _on_goal_response(self, future: Any, request: PendingNavigation) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:  # noqa: BLE001
            request.error = f"Failed to send Nav2 goal: {exc}"
            request.done_event.set()
            return

        if goal_handle is None or not goal_handle.accepted:
            request.error = "Nav2 goal was rejected."
            request.done_event.set()
            return

        request.accepted = True
        request.goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(lambda result: self._on_result(result, request))

    def _on_result(self, future: Any, request: PendingNavigation) -> None:
        try:
            result = future.result()
        except Exception as exc:  # noqa: BLE001
            request.error = f"Failed to get Nav2 result: {exc}"
            request.done_event.set()
            return

        if result is None:
            request.error = "Nav2 returned no result."
            request.done_event.set()
            return

        request.status = int(result.status)
        request.success = request.status == 4
        if not request.success and request.error is None:
            request.error = f"Nav2 goal finished with status={request.status}."
        request.done_event.set()

    def cancel(self, request: PendingNavigation) -> None:
        if request.goal_handle is None:
            return
        try:
            request.goal_handle.cancel_goal_async()
        except Exception:
            pass

    def close(self) -> None:
        self._node.destroy_node()


class SubprocessNav2GoalClient:
    def __init__(self, workspace_setup_bash: str):
        self._workspace_setup_bash = workspace_setup_bash
        self._runner_script = Path(__file__).with_name("nav2_goal_runner.py")

    @property
    def node(self) -> Any | None:
        return None

    @staticmethod
    def _clean_env_prefix() -> str:
        env_parts = ["env", "-i", "HOME=/root", "PATH=/usr/bin:/bin:/usr/sbin:/sbin"]
        for name in (
            "ROS_DOMAIN_ID",
            "ROS_LOCALHOST_ONLY",
            "RMW_IMPLEMENTATION",
            "CYCLONEDDS_URI",
            "FASTRTPS_DEFAULT_PROFILES_FILE",
        ):
            value = os.environ.get(name)
            if value:
                env_parts.append(f"{name}={shlex.quote(value)}")
        return " ".join(env_parts)

    def _build_command(self, goal: NavigationGoal, result_timeout: float) -> str:
        clean_shell = f"{self._clean_env_prefix()} bash --noprofile --norc -lc "
        server_timeout = max(60.0, min(float(result_timeout), 120.0))
        command = (
            f"source /opt/ros/humble/setup.bash && "
            f"source {self._workspace_setup_bash} && "
            f"python3 {self._runner_script} "
            f"--x {goal.x} --y {goal.y} --yaw {goal.yaw} "
            f"--frame-id {goal.frame_id} "
            f"--server-timeout {server_timeout} "
            f"--result-timeout {result_timeout}"
        )
        return clean_shell + repr(command)

    def send_goal(self, goal: NavigationGoal, *, result_timeout: float) -> PendingNavigation:
        request = PendingNavigation(goal=goal)
        command = self._build_command(goal, result_timeout)

        try:
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception as exc:  # noqa: BLE001
            request.error = f"Failed to launch Nav2 subprocess: {exc}"
            request.done_event.set()
            return request

        request.goal_handle = process

        def _wait_for_result() -> None:
            try:
                stdout, stderr = process.communicate()
                payload = None
                for line in reversed(stdout.splitlines()):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue

                if payload is None:
                    request.error = (
                        "Nav2 subprocess produced no JSON result. "
                        f"stdout={stdout[-500:]!r} stderr={stderr[-500:]!r}"
                    )
                else:
                    request.success = bool(payload.get("success", False))
                    request.status = int(payload["status"]) if payload.get("status") is not None else None
                    request.error = payload.get("error")
                    request.accepted = request.success or request.error != "Nav2 goal was rejected."
            except Exception as exc:  # noqa: BLE001
                request.error = f"Failed to collect Nav2 subprocess result: {exc}"
            finally:
                request.done_event.set()

        threading.Thread(target=_wait_for_result, name="nav2-subprocess-waiter", daemon=True).start()
        return request

    def cancel(self, request: PendingNavigation) -> None:
        process = request.goal_handle
        if process is None:
            return
        try:
            process.terminate()
        except Exception:
            pass

    def close(self) -> None:
        return


def start_ros_executor(*nodes: Any) -> tuple[Any, threading.Thread]:
    import rclpy
    from rclpy.executors import MultiThreadedExecutor

    if not rclpy.ok():
        rclpy.init(args=None)

    executor = MultiThreadedExecutor()
    for node in nodes:
        executor.add_node(node)

    thread = threading.Thread(target=executor.spin, name="ros2-executor", daemon=True)
    thread.start()
    return executor, thread


def stop_ros_executor(executor: Any, thread: threading.Thread | None) -> None:
    import rclpy

    try:
        executor.shutdown()
    except Exception:
        pass
    if thread is not None and thread.is_alive():
        thread.join(timeout=2.0)
    if rclpy.ok():
        rclpy.shutdown()


def load_manual_scene_graph(path_str: str) -> dict[str, Any] | None:
    if not path_str:
        return None
    path = Path(path_str)
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _parse_nav_map_yaml(path: Path) -> dict[str, Any] | None:
    try:
        import yaml

        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else None
    except Exception:
        pass

    parsed: dict[str, Any] = {}
    origin_values: list[float] | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("- ") and origin_values is not None:
            origin_values.append(float(line[2:].strip()))
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key == "origin":
            if value.startswith("[") and value.endswith("]"):
                origin_values = [float(item.strip()) for item in value[1:-1].split(",") if item.strip()]
                parsed[key] = origin_values
            else:
                origin_values = []
                parsed[key] = origin_values
        elif value:
            parsed[key] = value
    return parsed


def load_nav_occupancy_map(path: Path = NAV_MAP_YAML_PATH) -> NavOccupancyMap | None:
    if not path.is_file():
        return None

    map_config = _parse_nav_map_yaml(path)
    if map_config is None:
        return None

    image_name = map_config.get("image")
    resolution = map_config.get("resolution")
    origin = map_config.get("origin")
    if image_name is None or resolution is None or origin is None or len(origin) < 2:
        return None

    image_path = (path.parent / str(image_name)).resolve()
    image = np.asarray(Image.open(image_path).convert("L"), dtype=np.uint8)
    return NavOccupancyMap(
        resolution=float(resolution),
        origin_x=float(origin[0]),
        origin_y=float(origin[1]),
        image=image,
    )


def project_nav_goal_to_free_space(
    occupancy_map: NavOccupancyMap | None,
    robot_xy: np.ndarray,
    target_xy: np.ndarray,
    proposed_xy: np.ndarray,
) -> np.ndarray:
    clearance_radius_m = 0.10
    if occupancy_map is None or occupancy_map.has_clearance(
        float(proposed_xy[0]), float(proposed_xy[1]), clearance_radius_m
    ):
        return proposed_xy

    ray = proposed_xy - target_xy
    ray_norm = float(np.linalg.norm(ray))
    if ray_norm < 1e-6:
        ray = robot_xy - target_xy
        ray_norm = float(np.linalg.norm(ray))
    if ray_norm < 1e-6:
        return proposed_xy

    direction = ray / ray_norm
    max_backoff = max(float(np.linalg.norm(robot_xy - target_xy)), 0.5)
    step = max(occupancy_map.resolution, 0.05)
    distance = 0.0
    while distance <= max_backoff:
        candidate = proposed_xy + direction * distance
        if occupancy_map.has_clearance(float(candidate[0]), float(candidate[1]), clearance_radius_m):
            print(
                f"[NavGoal] Adjusted occupied goal from {proposed_xy.tolist()} "
                f"to clear cell {candidate.tolist()} (backoff={distance:.2f}m, clearance={clearance_radius_m:.2f}m)"
            )
            return candidate
        distance += step
    return proposed_xy


def resolve_tal_object_name(config: Any, object_name: str) -> str:
    normalized = _normalize_tal_arg(object_name)
    if normalized in config.object2idx:
        return normalized
    if normalized in getattr(config, "usd_to_tal", {}):
        return config.usd_to_tal[normalized]
    lowered = normalized.lower()
    for candidate in config.all_objects:
        if candidate.lower() == lowered:
            return candidate
    raise KeyError(f"Unknown TAL navigation target: {object_name}")


def infer_navigation_approach_distance(config: Any, target_name: str, source_action_name: str | None) -> float:
    resolved_target = resolve_tal_object_name(config, target_name)
    override_map = getattr(config, "nav_approach_distance_overrides", {})
    if resolved_target in override_map:
        return max(float(override_map[resolved_target]), 0.0)

    normalized_action = (source_action_name or "").strip().lower()
    if normalized_action == "picknplaceaonb":
        return max(float(getattr(config, "pick_approach_distance", 0.5)), 0.0)
    if normalized_action == "pushto":
        return max(float(getattr(config, "push_approach_distance", 0.55)), 0.0)
    return max(float(getattr(config, "base_approach_distance", 0.5)), 0.0)


def infer_navigation_approach_direction(config: Any, target_name: str) -> np.ndarray | None:
    resolved_target = resolve_tal_object_name(config, target_name)
    override_map = getattr(config, "nav_approach_direction_overrides", {})
    override = override_map.get(resolved_target)
    if override is None:
        return None

    direction = np.asarray(override, dtype=np.float32)[:2]
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return None
    return direction / norm


def build_navigation_goal(
    runtime_ctx: TALRuntimeContext,
    target_name: str,
    *,
    source_action_name: str | None = None,
) -> NavigationGoal:
    config = runtime_ctx.sim_env_config
    isaac_env = runtime_ctx.isaac_env
    isaac_env.update_metrics()
    metrics = isaac_env.metrics

    resolved_target = resolve_tal_object_name(config, target_name)
    robot_pos = np.asarray(metrics["husky"][0], dtype=np.float32)
    target_pos = np.asarray(metrics[resolved_target][0], dtype=np.float32)
    delta = target_pos[:2] - robot_pos[:2]
    dist = float(np.linalg.norm(delta))
    approach_distance = infer_navigation_approach_distance(config, resolved_target, source_action_name)
    configured_direction = infer_navigation_approach_direction(config, resolved_target)

    if configured_direction is not None:
        approach_mode = "configured_front"
        approach_direction = configured_direction
        proposed_goal_xy = target_pos[:2] + approach_direction * approach_distance
    else:
        approach_mode = "robot_vector"
        if dist < 1e-6:
            direction = np.array([1.0, 0.0], dtype=np.float32)
        else:
            direction = delta / dist
        approach_direction = -direction
        proposed_goal_xy = target_pos[:2] + approach_direction * approach_distance

    projected_goal_xy = project_nav_goal_to_free_space(
        load_nav_occupancy_map(),
        robot_pos[:2],
        target_pos[:2],
        proposed_goal_xy.astype(np.float32, copy=False),
    )
    goal_xy = projected_goal_xy

    normalized_action = (source_action_name or "").strip().lower()
    projection_shift = float(np.linalg.norm(projected_goal_xy - proposed_goal_xy))
    if normalized_action == "picknplaceaonb" and projection_shift > 0.25:
        goal_xy = proposed_goal_xy
        print(
            f"[NavGoal] projection override for grasp target={resolved_target}: "
            f"using proposed goal {proposed_goal_xy.tolist()} instead of projected "
            f"{projected_goal_xy.tolist()} (shift={projection_shift:.3f}m)"
        )

    target_distance = float(np.linalg.norm(target_pos[:2] - goal_xy))
    arm_reach_m = float(getattr(config, "arm_effective_reach_m", 0.40))
    yaw = math.atan2(float(target_pos[1] - goal_xy[1]), float(target_pos[0] - goal_xy[0]))
    print(
        f"[NavGoal] target={resolved_target} source_action={source_action_name or 'unknown'} "
        f"robot_xy={robot_pos[:2].tolist()} target_xy={target_pos[:2].tolist()} "
        f"approach_mode={approach_mode} approach_direction={approach_direction.tolist()} "
        f"approach_distance={approach_distance:.3f} proposed_goal_xy={proposed_goal_xy.tolist()} "
        f"goal_xy={goal_xy.tolist()} target_distance={target_distance:.3f} "
        f"arm_reach_limit={arm_reach_m:.3f}"
    )
    if target_distance > arm_reach_m:
        print(
            f"[NavGoal][WARN] target distance {target_distance:.3f}m exceeds "
            f"configured arm reach {arm_reach_m:.3f}m"
        )
    return NavigationGoal(x=float(goal_xy[0]), y=float(goal_xy[1]), yaw=yaw, frame_id="map")


def _camera_rgb_ready(camera: Any) -> bool:
    rgba = camera.get_rgba()
    return isinstance(rgba, np.ndarray) and rgba.ndim == 3 and rgba.shape[2] >= 3


def warm_up_cameras(
    world: Any,
    nav_bridge: IsaacNavBridge | None,
    cameras: Mapping[str, Any],
    *,
    dt: float,
    max_steps: int = 60,
) -> None:
    print(f"Warming up cameras for up to {max_steps} rendered simulation steps...")
    for step in range(max_steps):
        advance_simulation(world, nav_bridge, dt, render=True)
        ready_labels = [label for label, camera in cameras.items() if _camera_rgb_ready(camera)]
        if len(ready_labels) == len(cameras):
            print(f"Camera warmup complete after {step + 1} rendered steps.")
            return
    missing = [label for label, camera in cameras.items() if not _camera_rgb_ready(camera)]
    print(f"[Warn] Camera warmup timed out; cameras without valid RGBA buffers: {missing}")


def capture_rgb_images(cam_high: Any, cam_wrist: Any) -> dict[str, np.ndarray]:
    def _read_rgb(camera: Any, label: str) -> np.ndarray:
        fallback = np.zeros((224, 224, 3), dtype=np.uint8)
        for attempt in range(5):
            rgba = camera.get_rgba()
            if isinstance(rgba, np.ndarray) and rgba.ndim == 3 and rgba.shape[2] >= 3:
                rgb = rgba[:, :, :3]
                if rgb.dtype == np.float32:
                    return (rgb * 255).astype(np.uint8)
                return rgb.astype(np.uint8, copy=False)
            time.sleep(0.1)
        print(f"[Warn] Camera {label} returned invalid RGBA buffer; falling back to a black frame.")
        return fallback

    img_high_rgb = _read_rgb(cam_high, "cam_high")
    img_wrist_rgb = _read_rgb(cam_wrist, "cam_wrist")

    return {
        "cam_high": cv2.cvtColor(img_high_rgb, cv2.COLOR_RGB2BGR),
        "cam_wrist": cv2.cvtColor(img_wrist_rgb, cv2.COLOR_RGB2BGR),
    }


def destroy_camera(camera: Any, label: str) -> None:
    if camera is None:
        return
    destroy_fn = getattr(camera, "destroy", None)
    if callable(destroy_fn):
        try:
            destroy_fn()
            print(f"[Shutdown] Destroyed camera {label}.")
            return
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed to destroy camera {label} cleanly: {exc}")
    cleanup_fn = getattr(camera, "cleanup", None)
    if callable(cleanup_fn):
        try:
            cleanup_fn()
            print(f"[Shutdown] Cleaned up camera {label}.")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed to cleanup camera {label} cleanly: {exc}")


def read_robot_state(robot: Any, joint_names: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    all_joint_pos = robot.get_joint_positions()
    all_dof_names = robot.dof_names
    ordered_state = []
    for name in joint_names:
        if name not in all_dof_names:
            raise ValueError(f"Joint {name} not found in simulation DOF names: {all_dof_names}")
        idx = all_dof_names.index(name)
        ordered_state.append(all_joint_pos[idx])
    return np.array(ordered_state, dtype=np.float32), all_joint_pos, all_dof_names


def should_replan(step_idx: int, replan_every_n_steps: int) -> bool:
    if replan_every_n_steps <= 1:
        return True
    return step_idx % replan_every_n_steps == 0


def infer_action(policy_client: Any, images: dict[str, np.ndarray], state: np.ndarray, fused_prompt: str) -> np.ndarray:
    obs = {
        "observation/images/cam_high": images["cam_high"],
        "observation/images/cam_wrist": images["cam_wrist"],
        "observation/state": state,
        "prompt": fused_prompt,
    }
    result = policy_client.infer(obs)
    return result["actions"][0]


def apply_robot_action(robot: Any, target_action: np.ndarray, target_indices: np.ndarray, ArticulationAction: Any) -> None:
    target_action = np.asarray(target_action, dtype=np.float32)
    action_cmd = ArticulationAction(
        joint_positions=target_action,
        joint_indices=target_indices.astype(np.int32),
    )
    robot.apply_action(action_cmd)


def advance_simulation(world: Any, nav_bridge: IsaacNavBridge | None, dt: float, *, render: bool) -> None:
    if nav_bridge is not None:
        nav_bridge.advance(dt)
    world.step(render=render)


def stabilize_robot_root(
    robot_root: Any,
    world: Any,
    *,
    articulation: Any | None = None,
    nav_bridge: IsaacNavBridge | None = None,
    dt: float = 0.05,
    render: bool = True,
    settle_steps: int = 60,
) -> None:
    pose_source = articulation if articulation is not None else robot_root
    current_position, _ = pose_source.get_world_pose()
    if current_position is None:
        raise RuntimeError("Failed to read world pose for robot")

    print(f"[InitPose] Keeping robot at current world position: {np.asarray(current_position, dtype=np.float32).tolist()}")

    for step in range(settle_steps):
        if articulation is not None:
            articulation.set_linear_velocity(np.zeros(3, dtype=np.float32))
            articulation.set_angular_velocity(np.zeros(3, dtype=np.float32))
            joint_velocities = articulation.get_joint_velocities()
            if joint_velocities is not None:
                zero_velocities = np.zeros_like(np.asarray(joint_velocities, dtype=np.float32))
                try:
                    articulation.set_joint_velocities(zero_velocities)
                except AttributeError:
                    pass
        else:
            robot_root.set_linear_velocity(np.zeros(3))
            robot_root.set_angular_velocity(np.zeros(3))
        advance_simulation(world, nav_bridge, dt, render=render if step == 0 else False)

    final_position, _ = pose_source.get_world_pose()
    if final_position is None:
        raise RuntimeError("Failed to verify final world pose for robot")
    print(f"[InitPose] Final stabilized world position: {np.asarray(final_position, dtype=np.float32).tolist()}")


def warm_up_robot(
    robot: Any,
    world: Any,
    target_indices: np.ndarray,
    ArticulationAction: Any,
    *,
    nav_bridge: IsaacNavBridge | None = None,
    dt: float = 0.05,
    render: bool = False,
) -> None:
    start_positions = robot.get_joint_positions()[target_indices]
    num_steps = 240
    for i in range(num_steps):
        alpha = (i + 1) / float(num_steps)
        interpolated_positions = start_positions + alpha * (TRAIN_INIT_STATE - start_positions)
        step_action = ArticulationAction(
            joint_positions=interpolated_positions,
            joint_indices=target_indices.astype(np.int32),
        )
        robot.apply_action(step_action)
        advance_simulation(world, nav_bridge, dt, render=render)

    final_action = ArticulationAction(
        joint_positions=TRAIN_INIT_STATE,
        joint_indices=target_indices.astype(np.int32),
    )
    for _ in range(60):
        robot.apply_action(final_action)
        advance_simulation(world, nav_bridge, dt, render=False)


def run_wheel_self_test(
    robot: Any,
    robot_root_controller: RobotRootPoseController,
    world: Any,
    nav_bridge: IsaacNavBridge,
    *,
    dt: float,
    duration_sec: float,
    left_rad_s: float,
    right_rad_s: float,
    render: bool,
) -> None:
    print("[WheelSelfTest] Starting direct wheel velocity self test.")
    print(
        f"[WheelSelfTest] command left_rad_s={left_rad_s:.3f} "
        f"right_rad_s={right_rad_s:.3f} duration_sec={duration_sec:.2f} dt={dt:.3f}"
    )

    start_position, start_orientation = robot_root_controller.get_world_pose()
    if start_position is None:
        raise RuntimeError("Failed to read initial robot pose for wheel self test.")
    start_position = np.asarray(start_position, dtype=np.float32)
    start_yaw = quaternion_to_yaw(start_orientation)
    articulation_start_position, articulation_start_orientation = robot.get_world_pose()
    if articulation_start_position is None:
        raise RuntimeError("Failed to read initial articulation pose for wheel self test.")
    articulation_start_position = np.asarray(articulation_start_position, dtype=np.float32)
    articulation_start_yaw = quaternion_to_yaw(articulation_start_orientation)
    print(
        f"[WheelSelfTest] start xform_pose x={float(start_position[0]):.3f} "
        f"y={float(start_position[1]):.3f} yaw={start_yaw:.3f}"
    )
    print(
        f"[WheelSelfTest] start articulation_pose x={float(articulation_start_position[0]):.3f} "
        f"y={float(articulation_start_position[1]):.3f} yaw={articulation_start_yaw:.3f}"
    )

    steps = max(int(round(duration_sec / max(dt, 1e-3))), 1)
    wheel_joint_indices = np.array(
        [
            nav_bridge._wheel_indices["front_left"],
            nav_bridge._wheel_indices["front_right"],
            nav_bridge._wheel_indices["rear_left"],
            nav_bridge._wheel_indices["rear_right"],
        ],
        dtype=np.int32,
    )

    for step in range(steps):
        nav_bridge._apply_wheel_velocities(left_rad_s, right_rad_s)
        world.step(render=render)

        if (step + 1) % max(int(round(1.0 / max(dt, 1e-3))), 1) == 0 or step == steps - 1:
            position, orientation = robot_root_controller.get_world_pose()
            articulation_position, articulation_orientation = robot.get_world_pose()
            if position is None or articulation_position is None:
                continue
            position = np.asarray(position, dtype=np.float32)
            yaw = quaternion_to_yaw(orientation)
            articulation_position = np.asarray(articulation_position, dtype=np.float32)
            articulation_yaw = quaternion_to_yaw(articulation_orientation)
            joint_velocities = robot.get_joint_velocities()
            measured_joint_velocities = (
                np.asarray(joint_velocities, dtype=np.float32)[wheel_joint_indices].tolist()
                if joint_velocities is not None
                else None
            )
            dx = float(position[0] - start_position[0])
            dy = float(position[1] - start_position[1])
            dyaw = float(math.atan2(math.sin(yaw - start_yaw), math.cos(yaw - start_yaw)))
            distance = float(math.hypot(dx, dy))
            art_dx = float(articulation_position[0] - articulation_start_position[0])
            art_dy = float(articulation_position[1] - articulation_start_position[1])
            art_dyaw = float(
                math.atan2(
                    math.sin(articulation_yaw - articulation_start_yaw),
                    math.cos(articulation_yaw - articulation_start_yaw),
                )
            )
            art_distance = float(math.hypot(art_dx, art_dy))
            print(
                f"[WheelSelfTest] t={(step + 1) * dt:.2f}s "
                f"xform_dx={dx:.3f} xform_dy={dy:.3f} xform_dyaw={dyaw:.3f} xform_dist={distance:.3f} "
                f"art_dx={art_dx:.3f} art_dy={art_dy:.3f} art_dyaw={art_dyaw:.3f} art_dist={art_distance:.3f} "
                f"measured_joint_velocities={measured_joint_velocities}"
            )

    nav_bridge._apply_wheel_velocities(0.0, 0.0)
    for _ in range(max(int(round(1.0 / max(dt, 1e-3))), 1)):
        world.step(render=render)

    final_position, final_orientation = robot_root_controller.get_world_pose()
    articulation_final_position, articulation_final_orientation = robot.get_world_pose()
    if final_position is None or articulation_final_position is None:
        raise RuntimeError("Failed to read final robot pose for wheel self test.")
    final_position = np.asarray(final_position, dtype=np.float32)
    final_yaw = quaternion_to_yaw(final_orientation)
    articulation_final_position = np.asarray(articulation_final_position, dtype=np.float32)
    articulation_final_yaw = quaternion_to_yaw(articulation_final_orientation)
    final_dx = float(final_position[0] - start_position[0])
    final_dy = float(final_position[1] - start_position[1])
    final_dyaw = float(math.atan2(math.sin(final_yaw - start_yaw), math.cos(final_yaw - start_yaw)))
    final_distance = float(math.hypot(final_dx, final_dy))
    articulation_final_dx = float(articulation_final_position[0] - articulation_start_position[0])
    articulation_final_dy = float(articulation_final_position[1] - articulation_start_position[1])
    articulation_final_dyaw = float(
        math.atan2(
            math.sin(articulation_final_yaw - articulation_start_yaw),
            math.cos(articulation_final_yaw - articulation_start_yaw),
        )
    )
    articulation_final_distance = float(math.hypot(articulation_final_dx, articulation_final_dy))
    print(
        f"[WheelSelfTest] final xform_dx={final_dx:.3f} xform_dy={final_dy:.3f} "
        f"xform_dyaw={final_dyaw:.3f} xform_dist={final_distance:.3f} "
        f"art_dx={articulation_final_dx:.3f} art_dy={articulation_final_dy:.3f} "
        f"art_dyaw={articulation_final_dyaw:.3f} art_dist={articulation_final_distance:.3f}"
    )


def main() -> None:
    trace_phase(f"main: start (prompt={args.prompt!r})")
    nav_bridge = None
    nav_client = None
    cam_high = None
    cam_wrist = None
    world = None
    robot = None
    trace_phase("main: before initialize_tal_runtime")
    runtime_ctx = initialize_tal_runtime(
        TALControllerConfig(
            tal_root=args.tal_root,
            qwen_model=args.qwen_model,
            qwen_api_key_env=args.qwen_api_key_env,
            headless=args.headless,
        )
    )
    trace_phase("main: initialize_tal_runtime returned")
    simulation_app = runtime_ctx.simulation_app

    trace_phase("main: importing Isaac world/articulation/camera types")
    try:
        from isaacsim.core.api import World
    except ModuleNotFoundError:
        from omni.isaac.core import World

    try:
        from isaacsim.core.prims import SingleArticulation as Articulation
    except ModuleNotFoundError:
        try:
            from omni.isaac.core.articulations import Articulation
        except ModuleNotFoundError:
            from isaacsim.core.experimental.prims import Articulation

    try:
        from isaacsim.core.prims import XFormPrim
    except ModuleNotFoundError:
        try:
            from omni.isaac.core.prims import XFormPrim
        except ModuleNotFoundError:
            from isaacsim.core.experimental.prims import XformPrim as XFormPrim

    try:
        from isaacsim.core.utils.types import ArticulationAction
    except ModuleNotFoundError:
        from omni.isaac.core.utils.types import ArticulationAction

    try:
        from isaacsim.sensors.camera import Camera
    except ModuleNotFoundError:
        from omni.isaac.sensor import Camera

    robot_usd_name = runtime_ctx.sim_env_config.tal_to_usd["husky"]
    robot_prim_path = f"/World/{robot_usd_name}"
    trace_phase(f"main: loaded TAL scene path {runtime_ctx.sim_env_config.scene_usd_path}")
    trace_phase(f"main: robot prim resolved to {robot_prim_path}")

    trace_phase("main: constructing World")
    try:
        world = World(stage_units_in_meters=1.0)
    except TypeError:
        world = World()
    trace_phase("main: World constructed")
    trace_phase("main: adding robot articulation to world")
    robot = world.scene.add(Articulation(prim_path=robot_prim_path, name="firefighter"))
    trace_phase("main: robot articulation added")
    robot_root_prim = XFormPrim(robot_prim_path)
    if not robot_root_prim.is_valid():
        raise RuntimeError(f"Robot root prim is invalid: {robot_prim_path}")
    robot_root_controller = RobotRootPoseController(robot_root_prim)

    trace_phase("main: calling world.reset()")
    world.reset()
    trace_phase("main: world.reset() complete")

    sim_dof_names = robot.dof_names
    target_indices = []
    for name in JOINT_NAMES_IN_ORDER:
        if name in sim_dof_names:
            target_indices.append(sim_dof_names.index(name))
        else:
            print(f"Warning: joint {name} was not found in simulation.")
    target_indices = np.array(target_indices, dtype=np.int32)

    trace_phase("main: creating IsaacNavBridge")
    nav_bridge = IsaacNavBridge(robot, robot_root_controller, runtime_ctx, ArticulationAction)
    trace_phase("main: IsaacNavBridge created")
    trace_phase("main: stabilize_robot_root start")
    stabilize_robot_root(
        robot_root_controller,
        world,
        articulation=robot,
        nav_bridge=nav_bridge,
        dt=args.nav_control_dt,
        render=not args.headless,
    )
    trace_phase("main: stabilize_robot_root complete")

    if args.wheel_self_test:
        run_wheel_self_test(
            robot,
            robot_root_controller,
            world,
            nav_bridge,
            dt=args.nav_control_dt,
            duration_sec=args.wheel_self_test_duration_sec,
            left_rad_s=args.wheel_self_test_left_rad_s,
            right_rad_s=args.wheel_self_test_right_rad_s,
            render=not args.headless,
        )
        return

    trace_phase("main: warm_up_robot start")
    warm_up_robot(
        robot,
        world,
        target_indices,
        ArticulationAction,
        nav_bridge=nav_bridge,
        dt=args.nav_control_dt,
        render=not args.headless,
    )
    trace_phase("main: warm_up_robot complete")

    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    policy = None

    tal_planner = LazyTALPlanner(runtime_ctx)
    scene_graph_provider = TALSceneGraphProvider(runtime_ctx)
    manual_scene_graph = load_manual_scene_graph(args.manual_scene_graph_json)

    nav_client = SubprocessNav2GoalClient("/root/gpufree-data/code/robot_ws/install/local_setup.bash")
    warmup_steps = max(int(args.nav_warmup_sec / max(args.nav_control_dt, 1e-3)), 1)
    trace_phase(
        f"main: warming up Nav2 bridge for {args.nav_warmup_sec:.2f}s "
        f"({warmup_steps} simulation steps)"
    )
    for _ in range(warmup_steps):
        advance_simulation(world, nav_bridge, args.nav_control_dt, render=not args.headless)
    trace_phase("main: Nav2 bridge warmup complete")
    trace_phase("main: creating cameras")
    cam_high = Camera(prim_path=CAMERA_HIGH_PATH, resolution=(224, 224))
    cam_wrist = Camera(prim_path=CAMERA_WRIST_PATH, resolution=(224, 224))
    cam_high.initialize()
    cam_wrist.initialize()
    trace_phase("main: cameras initialized")
    trace_phase("main: warm_up_cameras start")
    warm_up_cameras(
        world,
        nav_bridge,
        {"cam_high": cam_high, "cam_wrist": cam_wrist},
        dt=args.nav_control_dt,
    )
    trace_phase("main: warm_up_cameras complete")
    trace_phase("main: entering TAL + Nav2 + OpenPI closed-loop inference")

    latest_subtask = None
    latest_fused_prompt = args.prompt
    latest_parsed_subtask: ParsedTALSubtask | None = None
    latest_raw_subtask: str | None = None
    latest_raw_parsed_subtask: ParsedTALSubtask | None = None
    completed_navigation_subtasks: set[tuple[str, tuple[str, ...]]] = set()
    skip_replan_once = False
    force_replan = True
    step_idx = 0
    control_error: BaseException | None = None

    try:
        while True:
            print(f"[Loop] entering step {step_idx}")
            if args.max_steps >= 0 and step_idx >= args.max_steps:
                print("Reached max steps, exiting.")
                break

            advance_simulation(world, nav_bridge, args.nav_control_dt, render=not args.headless)

            print(f"[Step {step_idx}] Capturing RGB images...")
            images = capture_rgb_images(cam_high, cam_wrist)
            print(f"[Step {step_idx}] Reading robot state...")
            current_state, _, _ = read_robot_state(robot, JOINT_NAMES_IN_ORDER)
            print(f"[Step {step_idx}] Robot state: {current_state.tolist()}")

            if skip_replan_once:
                skip_replan_once = False
            elif force_replan or should_replan(step_idx, args.replan_every_n_steps):
                scene_graph_state_name = args.tal_world_state_name if step_idx == 0 else None
                print(
                    f"[Step {step_idx}] Replanning triggered. "
                    f"scene_graph_state_name={scene_graph_state_name!r}, "
                    f"manual_scene_graph={'yes' if manual_scene_graph is not None else 'no'}"
                )
                print(f"[Step {step_idx}] Building current scene graph from TAL runtime...")
                current_scene_graph, current_datapoint = scene_graph_provider.get_current_scene_graph(
                    state_name=scene_graph_state_name,
                    manual_scene_graph=manual_scene_graph,
                )
                if current_datapoint is not None:
                    print(f"[Step {step_idx}] TAL datapoint actions: {list(getattr(current_datapoint, 'actions', []))}")
                print(f"[Step {step_idx}] Calling TAL planner...")
                try:
                    tal_result = tal_planner.plan_first_action(
                        args.prompt,
                        current_scene_graph,
                        start_node=current_datapoint,
                    )
                    latest_raw_subtask = tal_result.first_action_text
                    latest_raw_parsed_subtask = parse_tal_subtask(
                        tal_result.predicted_actions[0] if tal_result.predicted_actions else latest_raw_subtask
                    )
                    latest_parsed_subtask = derive_executable_subtask(latest_raw_parsed_subtask)
                    if latest_parsed_subtask is not None and latest_parsed_subtask.is_navigation:
                        nav_key = (latest_parsed_subtask.name.lower(), tuple(latest_parsed_subtask.args))
                        if nav_key in completed_navigation_subtasks:
                            print(
                                f"[Step {step_idx}] Skipping already completed navigation subtask: "
                                f"{latest_parsed_subtask.text}"
                            )
                            latest_parsed_subtask = None
                            latest_subtask = latest_raw_subtask
                        else:
                            latest_subtask = latest_parsed_subtask.text
                    else:
                        latest_subtask = latest_parsed_subtask.text if latest_parsed_subtask is not None else latest_raw_subtask
                    latest_fused_prompt = build_fused_prompt(args.prompt, latest_subtask)
                except Exception as exc:  # noqa: BLE001
                    tal_result = TALPlanResult(
                        status="Error",
                        first_action_text=None,
                        predicted_actions=[],
                        current_scene_graph_json=current_scene_graph,
                        goal_scene_graph_json=None,
                        error=str(exc),
                    )
                    latest_subtask = None
                    latest_parsed_subtask = None
                    latest_fused_prompt = args.prompt
                    latest_raw_subtask = None
                    latest_raw_parsed_subtask = None
                force_replan = False
                print("=" * 80)
                print(f"[Step {step_idx}] user prompt: {args.prompt}")
                print(f"[Step {step_idx}] current scene graph: {json.dumps(current_scene_graph, ensure_ascii=False)}")
                if tal_result.goal_scene_graph_json is not None:
                    print(
                        f"[Step {step_idx}] TAL goal scene graph: "
                        f"{json.dumps(tal_result.goal_scene_graph_json, ensure_ascii=False)}"
                    )
                print(f"[Step {step_idx}] TAL status: {tal_result.status}")
                print(f"[Step {step_idx}] TAL predicted actions(raw): {tal_result.predicted_actions}")
                print(f"[Step {step_idx}] TAL first action(raw text): {latest_raw_subtask}")
                print(f"[Step {step_idx}] TAL parsed subtask(raw): {latest_raw_parsed_subtask}")
                print(f"[Step {step_idx}] TAL execution subtask: {latest_subtask}")
                print(f"[Step {step_idx}] TAL parsed subtask(exec): {latest_parsed_subtask}")
                print(f"[Step {step_idx}] fused prompt: {latest_fused_prompt}")
                if tal_result.error:
                    print(f"[Step {step_idx}] TAL error: {tal_result.error}")

            if latest_parsed_subtask is not None and latest_parsed_subtask.is_navigation:
                derived_from = latest_parsed_subtask.raw.get("derived_from") if isinstance(latest_parsed_subtask.raw, Mapping) else None
                source_action_name = derived_from.get("name") if isinstance(derived_from, Mapping) else None
                nav_goal = build_navigation_goal(
                    runtime_ctx,
                    latest_parsed_subtask.args[0],
                    source_action_name=source_action_name,
                )
                print(
                    f"[Step {step_idx}] Routing TAL subtask to Nav2: "
                    f"{latest_parsed_subtask.text} -> goal(x={nav_goal.x:.3f}, y={nav_goal.y:.3f}, yaw={nav_goal.yaw:.3f})"
                )
                nav_bridge.set_active_goal(nav_goal)
                pending_nav = nav_client.send_goal(nav_goal, result_timeout=args.nav_goal_timeout_sec)
                deadline = time.monotonic() + args.nav_goal_timeout_sec

                while not pending_nav.done_event.wait(timeout=0.0):
                    if time.monotonic() >= deadline:
                        nav_client.cancel(pending_nav)
                        raise TimeoutError(
                            f"Timed out waiting for Nav2 goal after {args.nav_goal_timeout_sec:.1f}s: {nav_goal}"
                        )
                    advance_simulation(world, nav_bridge, args.nav_control_dt, render=not args.headless)
                    time.sleep(min(max(args.nav_control_dt * 0.25, 0.005), 0.02))

                if not pending_nav.success:
                    nav_bridge.set_active_goal(None)
                    raise RuntimeError(
                        pending_nav.error or f"Nav2 navigation failed with status={pending_nav.status}"
                    )

                print(f"[Step {step_idx}] Nav2 goal reached successfully.", flush=True)
                nav_bridge.settle_to_goal_pose(nav_goal)
                nav_bridge.set_active_goal(None)
                completed_navigation_subtasks.add((latest_parsed_subtask.name.lower(), tuple(latest_parsed_subtask.args)))
                if isinstance(derived_from, Mapping):
                    latest_subtask = None
                    latest_fused_prompt = args.prompt
                    skip_replan_once = True
                    force_replan = False
                else:
                    force_replan = True
                latest_parsed_subtask = None
                step_idx += 1
                continue

            if policy is None:
                trace_phase("main: connecting to OpenPI Policy Server")
                policy = WebsocketClientPolicy(host=args.server_host, port=args.server_port)
                trace_phase("main: connected to OpenPI Policy Server")

            print(f"[Step {step_idx}] Sending fused prompt to OpenPI...", flush=True)
            target_action = infer_action(policy, images, current_state, latest_fused_prompt)
            print(f"[Step {step_idx}] OpenPI first action: {target_action}", flush=True)
            print(f"[Step {step_idx}] Applying action to robot...", flush=True)
            apply_robot_action(robot, target_action, target_indices, ArticulationAction)
            step_idx += 1
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as exc:  # noqa: BLE001
        control_error = exc
        print(f"[ERROR] Control loop failed: {exc}")
        traceback.print_exc()
    finally:
        print("[Shutdown] Closing ROS2 bridge/executor...")
        if nav_client is not None:
            try:
                nav_client.close()
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] Failed to close Nav2 client cleanly: {exc}")
        if nav_bridge is not None:
            try:
                nav_bridge.close()
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] Failed to close Isaac nav bridge cleanly: {exc}")
        print("[Shutdown] Releasing camera resources...")
        destroy_camera(cam_high, "cam_high")
        destroy_camera(cam_wrist, "cam_wrist")
        if world is not None:
            try:
                for _ in range(3):
                    world.step(render=False)
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] Failed to flush final world steps cleanly: {exc}")
        print("[Shutdown] Closing TAL runtime and SimulationApp...")
        runtime_ctx.close()

    if control_error is not None:
        raise SystemExit(1) from control_error


if __name__ == "__main__":
    main()
