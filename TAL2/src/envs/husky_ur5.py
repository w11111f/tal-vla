"""Compatibility wrapper that routes the legacy husky_ur5 API to Isaac Lab."""

import pickle

from src.envs import isaac_env


start = isaac_env.start
saveState = isaac_env.saveState
restoreState = isaac_env.restoreState
execute_collect_data = isaac_env.execute_collect_data
initRootNode = isaac_env.initRootNode
getDatapoint = isaac_env.getDatapoint
resetDatapoint = isaac_env.resetDatapoint
destroy = isaac_env.destroy


def save_bullet(file_path):
    state_id, previous_constraints = isaac_env.saveState()
    with open(file_path, "wb") as handle:
        pickle.dump(
            {
                "state_id": state_id,
                "previous_constraints": previous_constraints,
            },
            handle,
        )


def restore_bullet(file_path):
    with open(file_path, "rb") as handle:
        payload = pickle.load(handle)
    isaac_env.restoreState(
        payload["state_id"],
        payload.get("previous_constraints", {"husky": []}),
        None,
    )
