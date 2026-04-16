from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import importlib
import logging
import os
from pathlib import Path
import pickle
import sys
import threading
import time
from typing import Any, Protocol

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TALPlanResult:
    status: str
    first_action_text: str | None
    predicted_actions: list[str]
    current_scene_graph_json: dict[str, Any] | None = None
    goal_scene_graph_json: dict[str, Any] | None = None
    error: str | None = None


class SceneGraphProvider(Protocol):
    def extract(self, images: Mapping[str, Any], state: Any) -> dict[str, Any]:
        """Build the current scene graph from RGB images and robot state."""


class TALPlanner(Protocol):
    def plan_first_action(self, user_instruction: str, current_scene_graph_json: Mapping[str, Any]) -> TALPlanResult:
        """Return the first action from a TAL plan."""


class MissingSceneGraphProvider:
    def extract(self, images: Mapping[str, Any], state: Any) -> dict[str, Any]:
        raise NotImplementedError(
            "Scene graph extraction is not implemented yet. Provide `current_scene_graph_json` in the inference "
            "payload or register a scene graph provider."
        )


@dataclasses.dataclass
class TALRuntimeConfig:
    tal_enabled: bool = False
    tal_repo_root: str | None = None
    tal_config_path: str | None = None
    tal_qwen_model: str | None = None
    tal_qwen_api_key_env: str | None = None
    scene_graph_provider_cls: str | None = None
    replan_every_n_steps: int = 1
    replan_timeout_s: float | None = None
    prompt_fusion_mode: str = "original_plus_tal_first_action"
    fallback_to_raw_prompt: bool = True
    enable_tal_debug: bool = False
    candidate_action_num: int = 20
    select_from_candidate: int = 10
    max_planning_steps: int = 60


def build_fused_prompt(original_instruction: str, tal_first_action: str | None, mode: str) -> str:
    if mode != "original_plus_tal_first_action":
        raise ValueError(f"Unsupported prompt fusion mode: {mode}")
    if not tal_first_action:
        return original_instruction
    return f"User task: {original_instruction.strip()}.\nCurrent subtask: {tal_first_action.strip()}."


def load_object(dotted_path: str) -> Any:
    module_name, _, attr_name = dotted_path.rpartition(".")
    if not module_name:
        raise ValueError(f"Expected a dotted import path, got: {dotted_path}")
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def create_scene_graph_provider(class_path: str | None) -> SceneGraphProvider:
    if not class_path:
        return MissingSceneGraphProvider()
    provider_or_type = load_object(class_path)
    return provider_or_type() if isinstance(provider_or_type, type) else provider_or_type


class LazyTALPlanner:
    def __init__(self, runtime_config: TALRuntimeConfig):
        self._runtime_config = runtime_config
        self._runtime_lock = threading.Lock()
        self._runtime: dict[str, Any] | None = None

    def _resolve_tal_root(self) -> Path:
        if self._runtime_config.tal_repo_root is None:
            raise ValueError("TAL is enabled but `tal_repo_root` is not configured.")

        root = Path(self._runtime_config.tal_repo_root).resolve()
        candidates = (
            root,
            root / "TAL2",
            root.parent / "TAL2",
        )
        for candidate in candidates:
            if (candidate / "src").exists() and (candidate / "settings").exists():
                return candidate
        raise FileNotFoundError(f"Could not locate TAL repo root under: {root}")

    def _ensure_runtime(self) -> dict[str, Any]:
        if self._runtime is not None:
            return self._runtime

        with self._runtime_lock:
            if self._runtime is not None:
                return self._runtime

            tal_root = self._resolve_tal_root()
            if str(tal_root) not in sys.path:
                sys.path.insert(0, str(tal_root))

            tal_config_module = importlib.import_module("src.config.config")
            env_constants_module = importlib.import_module("src.envs.CONSTANTS")
            planning_module = importlib.import_module("src.tal.utils_planning")
            training_module = importlib.import_module("src.tal.utils_training")
            translator_module = importlib.import_module("src.tal.scene_graph_translator")
            approx_module = importlib.import_module("src.envs.approx")

            init_args = tal_config_module.init_args
            EnvironmentConfig = env_constants_module.EnvironmentConfig
            plan_with_natural_language_instruction = planning_module.plan_with_natural_language_instruction
            scene_graph_json_to_dgl = translator_module.scene_graph_json_to_dgl
            get_model = training_module.get_model
            load_model = training_module.load_model

            args = init_args()
            args.exec_type = "policy"
            try:
                import torch

                args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            except Exception:
                args.device = None

            if self._runtime_config.tal_qwen_model:
                args.qwen_model = self._runtime_config.tal_qwen_model

            api_key_env = self._runtime_config.tal_qwen_api_key_env
            if api_key_env:
                args.qwen_api_key = os.getenv(api_key_env, args.qwen_api_key)

            config = EnvironmentConfig(args)

            def load_required_model(model_name: str):
                model = get_model(config, model_name, config.features_dim, config.num_objects)
                seq_prefix = "Seq_" if config.training == "gcn_seq" else ""
                stable_ckpt = Path(config.MODEL_SAVE_PATH) / f"{seq_prefix}{model.name}_Trained.ckpt"
                ckpt_path = stable_ckpt if stable_ckpt.exists() else None
                if ckpt_path is None:
                    model_dir = Path(config.MODEL_SAVE_PATH)
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
                model, _, _, _ = load_model(config, seq_prefix + model.name + "_Trained", model, file_path=str(ckpt_path))
                if config.device is not None:
                    model = model.to(config.device)
                return model

            model_action_effect = load_required_model("AFE")
            model_action = load_required_model("APN")

            features_save_path = Path(config.MODEL_SAVE_PATH) / "action_effect_features_avg.pkl"
            with features_save_path.open("rb") as file_obj:
                action_effect_features = pickle.load(file_obj)

            self._runtime = {
                "config": config,
                "plan_with_natural_language_instruction": plan_with_natural_language_instruction,
                "scene_graph_json_to_dgl": scene_graph_json_to_dgl,
                "model_action": model_action,
                "model_action_effect": model_action_effect,
                "action_effect_features": action_effect_features,
                "approx": approx_module,
            }
            return self._runtime

    def plan_first_action(self, user_instruction: str, current_scene_graph_json: Mapping[str, Any]) -> TALPlanResult:
        runtime = self._ensure_runtime()
        config = runtime["config"]
        scene_graph_json_to_dgl = runtime["scene_graph_json_to_dgl"]
        plan_with_natural_language_instruction = runtime["plan_with_natural_language_instruction"]

        current_state_graph = scene_graph_json_to_dgl(config, dict(current_scene_graph_json))
        if config.device is not None:
            current_state_graph = current_state_graph.to(config.device)

        world_num = 0
        graph_world_name = getattr(config, "graph_world_name", "")
        digits = "".join(ch for ch in str(graph_world_name) if ch.isdigit())
        if digits:
            world_num = int(digits)

        result = plan_with_natural_language_instruction(
            config,
            model_action=runtime["model_action"],
            model_extract_feature=runtime["model_action_effect"],
            action_effect_features=runtime["action_effect_features"],
            instruction=user_instruction,
            world_num=world_num,
            current_state_graph=current_state_graph,
            current_scene_graph_json=dict(current_scene_graph_json),
            qwen_model_name=self._runtime_config.tal_qwen_model or getattr(config, "qwen_model", "qwen3-max"),
            qwen_api_key=os.getenv(self._runtime_config.tal_qwen_api_key_env, None)
            if self._runtime_config.tal_qwen_api_key_env
            else None,
            candidate_action_num=self._runtime_config.candidate_action_num,
            select_from_candidate=self._runtime_config.select_from_candidate,
            trajectory_length=self._runtime_config.max_planning_steps,
            with_pca=True,
        )

        predicted_actions = list(result.get("predicted_actions", []))
        first_action = predicted_actions[0] if predicted_actions else None
        return TALPlanResult(
            status=result.get("status", "Unknown"),
            first_action_text=first_action,
            predicted_actions=predicted_actions,
            current_scene_graph_json=result.get("current_scene_graph_json"),
            goal_scene_graph_json=result.get("goal_scene_graph_json"),
            error=result.get("error"),
        )

    def close(self) -> None:
        runtime = self._runtime
        if runtime is None:
            return
        approx = runtime.get("approx")
        if approx is None:
            return
        try:
            approx.close_backend()
        except Exception:
            logger.exception("Failed to close TAL backend cleanly.")


class ClosedLoopTALManager:
    def __init__(
        self,
        runtime_config: TALRuntimeConfig,
        *,
        scene_graph_provider: SceneGraphProvider | None = None,
        tal_planner: TALPlanner | None = None,
    ):
        self._runtime_config = runtime_config
        self._scene_graph_provider = scene_graph_provider or create_scene_graph_provider(runtime_config.scene_graph_provider_cls)
        self._tal_planner = tal_planner or LazyTALPlanner(runtime_config)
        self._task_instruction: str | None = None
        self._latest_plan: TALPlanResult | None = None
        self._effective_prompt: str | None = None
        self._step_count = 0
        self._last_replan_monotonic: float | None = None
        self._last_debug_payload: dict[str, Any] | None = None

    @property
    def last_debug_payload(self) -> dict[str, Any] | None:
        return self._last_debug_payload

    def reset(self) -> None:
        self._task_instruction = None
        self._latest_plan = None
        self._effective_prompt = None
        self._step_count = 0
        self._last_replan_monotonic = None
        self._last_debug_payload = None

    def reset_step_counter(self) -> None:
        self._step_count = 0
        self._last_replan_monotonic = time.monotonic()

    def set_task(self, prompt: str) -> None:
        normalized_prompt = prompt.strip()
        if self._task_instruction == normalized_prompt:
            return
        self._task_instruction = normalized_prompt
        self._latest_plan = None
        self._effective_prompt = normalized_prompt
        self._step_count = 0
        self._last_replan_monotonic = None
        self._last_debug_payload = None

    def get_effective_prompt(self) -> str | None:
        if self._effective_prompt is not None:
            return self._effective_prompt
        return self._task_instruction

    def _timeout_reached(self) -> bool:
        if self._runtime_config.replan_timeout_s is None or self._last_replan_monotonic is None:
            return False
        return (time.monotonic() - self._last_replan_monotonic) >= self._runtime_config.replan_timeout_s

    def _should_replan(self) -> bool:
        if self._task_instruction is None:
            return False
        if self._latest_plan is None:
            return True
        if self._step_count >= max(1, self._runtime_config.replan_every_n_steps):
            return True
        return self._timeout_reached()

    def _fallback_prompt(self) -> str:
        if self._task_instruction is None:
            raise ValueError("Cannot build a fallback prompt before a task is set.")
        return self._task_instruction

    def maybe_replan(
        self,
        *,
        images: Mapping[str, Any],
        state: Any,
        injected_scene_graph: Mapping[str, Any] | None = None,
    ) -> str:
        if self._task_instruction is None:
            raise ValueError("Task instruction must be set before TAL replanning.")

        effective_prompt = self.get_effective_prompt() or self._task_instruction
        if not self._runtime_config.tal_enabled:
            self._step_count += 1
            return effective_prompt

        if not self._should_replan():
            self._step_count += 1
            return effective_prompt

        try:
            if injected_scene_graph is not None:
                scene_graph = dict(injected_scene_graph)
            else:
                scene_graph = self._scene_graph_provider.extract(images, state)

            plan = self._tal_planner.plan_first_action(self._task_instruction, scene_graph)
            self._latest_plan = plan
            self._effective_prompt = build_fused_prompt(
                self._task_instruction,
                plan.first_action_text,
                self._runtime_config.prompt_fusion_mode,
            )
            self.reset_step_counter()
            if self._runtime_config.enable_tal_debug:
                self._last_debug_payload = dataclasses.asdict(plan)
            effective_prompt = self._effective_prompt
        except Exception as exc:
            logger.exception("Falling back to raw prompt because TAL replanning failed.")
            if not self._runtime_config.fallback_to_raw_prompt:
                raise
            self._latest_plan = None
            self._effective_prompt = self._fallback_prompt()
            if self._runtime_config.enable_tal_debug:
                self._last_debug_payload = {"status": "Error", "error": str(exc)}
            effective_prompt = self._effective_prompt

        self._step_count += 1
        return effective_prompt
