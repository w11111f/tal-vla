import numpy as np

from openpi.models import model as _model
from openpi.policies import pro630demo_policy
from openpi import tal_runtime


class FakeSceneGraphProvider:
    def __init__(self):
        self.calls = 0

    def extract(self, images, state):
        self.calls += 1
        return {"nodes": [{"name": "cube_gray"}], "edges": []}


class FakePlanner:
    def __init__(self):
        self.calls = 0
        self.scene_graphs = []

    def plan_first_action(self, user_instruction, current_scene_graph_json):
        self.calls += 1
        self.scene_graphs.append(current_scene_graph_json)
        return tal_runtime.TALPlanResult(
            status="Correct",
            first_action_text=f"pick cube step {self.calls}",
            predicted_actions=[f"pick cube step {self.calls}", "move"],
            current_scene_graph_json=current_scene_graph_json,
            goal_scene_graph_json={"nodes": [], "edges": []},
        )


def _make_data(prompt="pick up the cube"):
    return {
        "observation/images/cam_high": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/images/cam_wrist": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/state": np.zeros((7,), dtype=np.float32),
        "prompt": prompt,
    }


def test_input_adapter_supports_runtime_fields():
    adapter = pro630demo_policy.Pro630InputAdapter()

    transformed = adapter(
        {
            "cam_high": np.zeros((224, 224, 3), dtype=np.uint8),
            "cam_wrist": np.zeros((224, 224, 3), dtype=np.uint8),
            "state": np.zeros((7,), dtype=np.float32),
            "prompt": "pick up the cube",
            "current_scene_graph_json": {"nodes": [], "edges": []},
            "reset_tal_context": True,
        }
    )

    assert "observation/images/cam_high" in transformed
    assert transformed["prompt"] == "pick up the cube"
    assert transformed["current_scene_graph_json"] == {"nodes": [], "edges": []}
    assert transformed["reset_tal_context"] is True


def test_custom_inputs_fuses_prompt_with_tal_action():
    provider = FakeSceneGraphProvider()
    planner = FakePlanner()
    transform = pro630demo_policy.CustomInputs(
        model_type=_model.ModelType.PI05,
        tal_enabled=True,
        scene_graph_provider=provider,
        tal_planner=planner,
        replan_every_n_steps=3,
    )

    result = transform(_make_data())

    assert result["prompt"] == "User task: pick up the cube.\nCurrent subtask: pick cube step 1."
    assert provider.calls == 1
    assert planner.calls == 1
    assert result["image"]["base_0_rgb"].shape == (224, 224, 3)
    assert result["image_mask"]["right_wrist_0_rgb"] == np.False_


def test_custom_inputs_uses_injected_scene_graph_before_provider():
    provider = FakeSceneGraphProvider()
    planner = FakePlanner()
    transform = pro630demo_policy.CustomInputs(
        model_type=_model.ModelType.PI05,
        tal_enabled=True,
        scene_graph_provider=provider,
        tal_planner=planner,
        replan_every_n_steps=1,
    )

    data = _make_data()
    data["current_scene_graph_json"] = {"nodes": [{"name": "tray"}], "edges": []}
    result = transform(data)

    assert result["prompt"].endswith("pick cube step 1.")
    assert provider.calls == 0
    assert planner.scene_graphs[0] == {"nodes": [{"name": "tray"}], "edges": []}


def test_custom_inputs_replans_after_threshold():
    provider = FakeSceneGraphProvider()
    planner = FakePlanner()
    transform = pro630demo_policy.CustomInputs(
        model_type=_model.ModelType.PI05,
        tal_enabled=True,
        scene_graph_provider=provider,
        tal_planner=planner,
        replan_every_n_steps=2,
    )

    first = transform(_make_data())
    second = transform(_make_data())
    third = transform(_make_data())

    assert first["prompt"].endswith("pick cube step 1.")
    assert second["prompt"].endswith("pick cube step 1.")
    assert third["prompt"].endswith("pick cube step 2.")
    assert planner.calls == 2


def test_custom_inputs_falls_back_to_raw_prompt_on_tal_failure():
    class FailingProvider:
        def extract(self, images, state):
            raise NotImplementedError("not ready")

    transform = pro630demo_policy.CustomInputs(
        model_type=_model.ModelType.PI05,
        tal_enabled=True,
        scene_graph_provider=FailingProvider(),
        replan_every_n_steps=1,
        fallback_to_raw_prompt=True,
    )

    result = transform(_make_data())

    assert result["prompt"] == "pick up the cube"


def test_training_batches_skip_tal_replanning():
    provider = FakeSceneGraphProvider()
    planner = FakePlanner()
    transform = pro630demo_policy.CustomInputs(
        model_type=_model.ModelType.PI05,
        tal_enabled=True,
        scene_graph_provider=provider,
        tal_planner=planner,
    )

    data = _make_data()
    data["actions"] = np.zeros((10, 7), dtype=np.float32)
    result = transform(data)

    assert result["prompt"] == "pick up the cube"
    assert provider.calls == 0
    assert planner.calls == 0


def test_reset_tal_context_forces_replanning():
    provider = FakeSceneGraphProvider()
    planner = FakePlanner()
    transform = pro630demo_policy.CustomInputs(
        model_type=_model.ModelType.PI05,
        tal_enabled=True,
        scene_graph_provider=provider,
        tal_planner=planner,
        replan_every_n_steps=10,
    )

    transform(_make_data())
    reset_data = _make_data()
    reset_data["reset_tal_context"] = True
    result = transform(reset_data)

    assert result["prompt"].endswith("pick cube step 2.")
    assert planner.calls == 2


def test_tal_manager_replans_on_timeout():
    provider = FakeSceneGraphProvider()
    planner = FakePlanner()
    manager = tal_runtime.ClosedLoopTALManager(
        tal_runtime.TALRuntimeConfig(
            tal_enabled=True,
            replan_every_n_steps=100,
            replan_timeout_s=0.01,
        ),
        scene_graph_provider=provider,
        tal_planner=planner,
    )
    manager.set_task("pick up the cube")

    first = manager.maybe_replan(
        images={"cam_high": np.zeros((1, 1, 3), dtype=np.uint8)},
        state=np.zeros((7,), dtype=np.float32),
    )
    manager._last_replan_monotonic -= 1.0
    second = manager.maybe_replan(
        images={"cam_high": np.zeros((1, 1, 3), dtype=np.uint8)},
        state=np.zeros((7,), dtype=np.float32),
    )

    assert first.endswith("pick cube step 1.")
    assert second.endswith("pick cube step 2.")
    assert planner.calls == 2
