from __future__ import annotations

import dataclasses
import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi import tal_runtime


def make_custom_example() -> dict:
    """Creates a random input example for the pro630 policy."""
    return {
        "cam_high": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "cam_wrist": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "state": np.random.rand(7),
        "prompt": "pick up the object",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _lookup(data: dict, *keys: str, default=None):
    for key in keys:
        if key in data:
            return data[key]
    return default


@dataclasses.dataclass
class Pro630InputAdapter(transforms.DataTransformFn):
    """Normalizes both training-style and runtime-style payloads for the pro630 policy."""

    def __call__(self, data: dict) -> dict:
        normalized = dict(data)

        cam_high = _lookup(normalized, "observation/images/cam_high", "cam_high")
        cam_wrist = _lookup(normalized, "observation/images/cam_wrist", "cam_wrist")
        state = _lookup(normalized, "observation/state", "state")
        actions = _lookup(normalized, "actions", "action")
        prompt = _lookup(normalized, "prompt", "task")
        current_scene_graph_json = _lookup(normalized, "current_scene_graph_json")
        reset_tal_context = bool(_lookup(normalized, "reset_tal_context", default=False))

        result = {
            "observation/images/cam_high": cam_high,
            "observation/images/cam_wrist": cam_wrist,
            "observation/state": state,
        }
        if actions is not None:
            result["actions"] = actions
        if prompt is not None:
            result["prompt"] = prompt
        if current_scene_graph_json is not None:
            result["current_scene_graph_json"] = current_scene_graph_json
        if reset_tal_context:
            result["reset_tal_context"] = True
        return result


@dataclasses.dataclass
class CustomInputs(transforms.DataTransformFn):
    """Builds OpenPI model inputs for the pro630 robot, with optional TAL closed-loop prompting."""

    model_type: _model.ModelType
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
    tal_manager: tal_runtime.ClosedLoopTALManager | None = None
    scene_graph_provider: tal_runtime.SceneGraphProvider | None = None
    tal_planner: tal_runtime.TALPlanner | None = None

    def __post_init__(self):
        if self.tal_manager is not None or not (
            self.tal_enabled or self.scene_graph_provider is not None or self.tal_planner is not None
        ):
            return
        runtime_config = tal_runtime.TALRuntimeConfig(
            tal_enabled=self.tal_enabled,
            tal_repo_root=self.tal_repo_root,
            tal_config_path=self.tal_config_path,
            tal_qwen_model=self.tal_qwen_model,
            tal_qwen_api_key_env=self.tal_qwen_api_key_env,
            scene_graph_provider_cls=self.scene_graph_provider_cls,
            replan_every_n_steps=self.replan_every_n_steps,
            replan_timeout_s=self.replan_timeout_s,
            prompt_fusion_mode=self.prompt_fusion_mode,
            fallback_to_raw_prompt=self.fallback_to_raw_prompt,
            enable_tal_debug=self.enable_tal_debug,
        )
        self.tal_manager = tal_runtime.ClosedLoopTALManager(
            runtime_config,
            scene_graph_provider=self.scene_graph_provider,
            tal_planner=self.tal_planner,
        )

    def _compute_prompt(self, data: dict, base_image: np.ndarray, wrist_image: np.ndarray, state: np.ndarray) -> str | None:
        prompt = data.get("prompt")
        if prompt is None:
            return None
        if isinstance(prompt, bytes):
            prompt = prompt.decode("utf-8")
        else:
            prompt = str(prompt)

        if self.tal_manager is None or not self.tal_enabled:
            return prompt

        if data.get("reset_tal_context", False):
            self.tal_manager.reset()

        self.tal_manager.set_task(prompt)

        # Do not invoke TAL during supervised training batches, which include action targets.
        if "actions" in data:
            return prompt

        images = {
            "cam_high": base_image,
            "cam_wrist": wrist_image,
        }
        injected_scene_graph = data.get("current_scene_graph_json")
        return self.tal_manager.maybe_replan(
            images=images,
            state=state,
            injected_scene_graph=injected_scene_graph,
        )

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/images/cam_high"])
        wrist_image = _parse_image(data["observation/images/cam_wrist"])
        state = np.asarray(data["observation/state"])

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        effective_prompt = self._compute_prompt(data, base_image, wrist_image, state)
        if effective_prompt is not None:
            inputs["prompt"] = effective_prompt

        return inputs


@dataclasses.dataclass
class CustomOutputs(transforms.DataTransformFn):
    """Projects the model action chunk back into the pro630 action space."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])}
