# Closed-Loop TAL + OpenPI Integration For `pi05_pro630_lora`

## Summary
Implement a closed-loop inference pipeline for your `pro630` robot policy so runtime behavior becomes:

`RGB image + robot state + user instruction`
-> `scene_graph_provider` extracts current scene graph from RGB
-> TAL converts `user instruction + current scene graph` into a multi-step plan
-> select the first planned action
-> build fused VLA prompt from `original instruction + TAL first action`
-> existing OpenPI `pi05_pro630_lora` policy predicts robot action
-> robot executes action
-> after a fixed number of inference/control steps or timeout, reacquire RGB/state, regenerate scene graph, rerun TAL, and continue until task stop condition is handled externally

This should be implemented as a runtime integration layer around the existing pro630 policy path, without changing the base OpenPI model interface.

## Key Changes
- Add a TAL runtime bridge module in `openpi/src/openpi/` that:
  - loads TAL models once at startup
  - exposes a callable like `plan_first_action(user_instruction, current_scene_graph_json) -> TALPlanResult`
  - internally reuses TAL model loading and planning logic from the current TAL codebase
  - returns at least:
    - `first_action_text`
    - `predicted_actions`
    - `status`
    - optional debug payloads such as current/goal scene graph JSON

- Add a scene graph extraction interface for the future RGB perception module:
  - define `SceneGraphProvider` interface with `extract(images, state) -> scene_graph_json`
  - add a default placeholder implementation that raises a clear not-implemented error
  - support optional direct injection of `current_scene_graph_json` in inference input so you can test without the detector being finished
  - scene graph source precedence:
    1. request-provided `current_scene_graph_json`
    2. `scene_graph_provider.extract(...)`
    3. fallback path to raw prompt if TAL is enabled but no scene graph is available

- Extend [`pro630demo_policy.py`](c:\Users\w'y'f\OneDrive\Desktop\tal-vla\openpi\src\openpi\policies\pro630demo_policy.py) to make the pro630 input transform TAL-aware:
  - preserve current image parsing and state handling
  - keep mapping:
    - `cam_high -> base_0_rgb`
    - `cam_wrist -> left_wrist_0_rgb`
    - zero-filled `right_wrist_0_rgb`
  - accept raw user prompt as the task-level instruction
  - before returning model inputs:
    - obtain current scene graph
    - ask TAL for a plan
    - take the first TAL action
    - build fused prompt from original instruction + TAL first action
  - if TAL fails, or scene graph is unavailable, fall back to original prompt
  - keep `CustomOutputs` unchanged except for any needed comments/cleanup

- Add a closed-loop TAL context/state manager owned by the pro630 runtime path:
  - store task-level instruction
  - store latest TAL result
  - store current replan step counter
  - store timestamp / timeout bookkeeping
  - expose methods like:
    - `set_task(prompt)`
    - `maybe_replan(images, state, injected_scene_graph=None)`
    - `reset()`
    - `get_effective_prompt()`
  - replan policy:
    - TAL runs once when a task starts
    - TAL reruns after a fixed number of policy inference calls or timeout
    - no internal subtask-success classifier for v1
    - no “plan once forever” caching for v1 closed loop

- Update the pro630 data config in [`config.py`](c:\Users\w'y'f\OneDrive\Desktop\tal-vla\openpi\src\openpi\training\config.py):
  - keep `pi05_pro630_lora` model config unchanged unless prompt length proves insufficient
  - extend `LeRobotpro630DataConfig` with runtime TAL fields:
    - `tal_enabled: bool`
    - `tal_repo_root: str | None`
    - `tal_config_path: str | None`
    - `tal_qwen_model: str | None`
    - `tal_qwen_api_key_env: str | None`
    - `scene_graph_provider_cls: str | None`
    - `replan_every_n_steps: int`
    - `replan_timeout_s: float | None`
    - `prompt_fusion_mode: Literal["original_plus_tal_first_action"]`
    - `fallback_to_raw_prompt: bool`
    - `enable_tal_debug: bool`
  - pass those settings into the TAL-aware pro630 transform/runtime

- Implement prompt fusion behavior as a deterministic formatter:
  - default fused prompt should preserve both the user’s goal and TAL’s current subtask
  - recommended template:
    - `User task: <original instruction>.`
    - `Current subtask: <tal first action>.`
  - keep formatting centralized in one helper so you can revise wording later without touching policy logic

- Add runtime hooks for closed-loop replanning:
  - each inference call increments a local step counter
  - when `step_count >= replan_every_n_steps`, refresh TAL prompt context
  - if timeout is configured and exceeded, force replan on next inference
  - reset counters when:
    - prompt changes
    - explicit reset is called
    - TAL replans successfully

- Keep the base OpenPI server and model contract stable:
  - websocket server remains transport-only
  - model still receives normal OpenPI `Observation`
  - no changes to `Policy.infer()` signature are required beyond allowing extra inference fields consumed by the pro630 transform/runtime

## Public Interfaces / Runtime Contract
- Existing pro630 inference payload should continue to support current fields.
- Add optional runtime fields for the closed-loop integration:
  - `prompt`: user instruction
  - `current_scene_graph_json`: optional externally supplied scene graph
  - optional `reset_tal_context`: bool to clear the TAL closed-loop state
- The effective prompt sent into tokenization/model transforms becomes:
  - raw prompt if TAL is unavailable
  - fused prompt if TAL succeeds
- Closed-loop replanning is driven by:
  - fixed inference-step count
  - optional timeout
- Subtask completion is not judged inside OpenPI in v1; replanning is periodic.

## Test Plan
- Unit tests for pro630 TAL-aware input transform:
  - raw prompt without TAL -> unchanged prompt path
  - TAL success -> fused prompt contains original instruction and TAL first action
  - missing scene graph provider -> fallback to raw prompt
  - injected `current_scene_graph_json` -> TAL runs without provider
  - prompt change -> TAL context resets

- Closed-loop state tests:
  - initial infer triggers TAL plan
  - repeated infer before threshold does not replan
  - infer at `replan_every_n_steps` triggers replan
  - timeout forces replan
  - explicit reset clears counters and cached TAL result

- Data compatibility tests:
  - image parsing still produces valid OpenPI image keys
  - state shape remains valid for `pi05_pro630_lora`
  - fused prompt is still tokenized by existing model transforms
  - output actions remain sliced to robot action dimensions as before

- TAL bridge tests with mocks:
  - mocked scene graph provider + mocked TAL planner produce stable first-action prompt fusion
  - TAL exception path falls back correctly
  - no mutation to model-facing interface beyond prompt content

## Assumptions And Defaults
- RGB-to-scene-graph extraction is not implemented yet in this repo, so the code will provide an interface plus placeholder and support externally injected scene graph JSON for testing.
- TAL output mode is fixed to “generate multi-step plan, use only first action.”
- Prompt fusion mode is fixed to “original instruction + TAL first action.”
- Replanning is closed-loop and periodic, not one-shot.
- Replan trigger for v1 is fixed-step-count with optional timeout; no OpenPI-side subtask success checker is added in v1.
- TAL/scene-graph failures fall back to the raw user instruction so the robot can still run.
- External task termination / final goal completion is assumed to be handled by your robot runtime or future scene-graph-based success logic, not by this first integration pass.
