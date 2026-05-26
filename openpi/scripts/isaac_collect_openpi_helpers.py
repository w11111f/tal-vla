from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


@dataclass(frozen=True)
class StageTarget:
    name: str
    position: np.ndarray
    finger_joint_target: np.float32
    num_steps: int


def get_next_episode_index(dataset_dir: str | Path) -> int:
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        return 0

    max_index = -1
    for path in dataset_path.glob("episode_*.h5"):
        match = re.fullmatch(r"episode_(\d+)\.h5", path.name)
        if match is None:
            continue
        max_index = max(max_index, int(match.group(1)))
    return max_index + 1


def interpolate_joint_positions(start: np.ndarray, goal: np.ndarray, *, num_steps: int) -> np.ndarray:
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    start = np.asarray(start, dtype=np.float32)
    goal = np.asarray(goal, dtype=np.float32)
    alphas = np.linspace(1.0 / num_steps, 1.0, num_steps, dtype=np.float32)[:, None]
    return start[None, :] + alphas * (goal - start)[None, :]


def build_pick_place_stage_targets(
    cube_position: np.ndarray,
    pallet_position: np.ndarray,
    *,
    approach_height: float,
    grasp_height_offset: float,
    lift_height: float,
    place_height_offset: float,
    finger_open: float,
    finger_closed: float,
    move_steps: int = 45,
    gripper_steps: int = 20,
) -> list[StageTarget]:
    cube_position = np.asarray(cube_position, dtype=np.float32)
    pallet_position = np.asarray(pallet_position, dtype=np.float32)

    cube_above = cube_position.copy()
    cube_above[2] += np.float32(approach_height)

    grasp_position = cube_position.copy()
    grasp_position[2] += np.float32(grasp_height_offset)

    lifted_position = cube_position.copy()
    lifted_position[2] += np.float32(lift_height)

    pallet_above = pallet_position.copy()
    pallet_above[2] += np.float32(lift_height)

    place_position = pallet_position.copy()
    place_position[2] += np.float32(place_height_offset)

    retreat_position = pallet_position.copy()
    retreat_position[2] += np.float32(lift_height + 0.03)

    return [
        StageTarget("move_above_cube", cube_above, np.float32(finger_open), move_steps),
        StageTarget("move_to_grasp", grasp_position, np.float32(finger_open), move_steps),
        StageTarget("close_gripper", grasp_position, np.float32(finger_closed), gripper_steps),
        StageTarget("lift_cube", lifted_position, np.float32(finger_closed), move_steps),
        StageTarget("move_above_pallet", pallet_above, np.float32(finger_closed), move_steps),
        StageTarget("move_to_place", place_position, np.float32(finger_closed), move_steps),
        StageTarget("open_gripper", place_position, np.float32(finger_open), gripper_steps),
        StageTarget("retreat", retreat_position, np.float32(finger_open), move_steps),
    ]
