from pathlib import Path

import numpy as np

from isaac_collect_openpi_helpers import StageTarget
from isaac_collect_openpi_helpers import build_pick_place_stage_targets
from isaac_collect_openpi_helpers import get_next_episode_index
from isaac_collect_openpi_helpers import interpolate_joint_positions


def test_get_next_episode_index_ignores_unrelated_files(tmp_path: Path) -> None:
    (tmp_path / "episode_000000.h5").write_text("", encoding="utf-8")
    (tmp_path / "episode_000014.h5").write_text("", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("", encoding="utf-8")
    (tmp_path / "episode_bad.h5").write_text("", encoding="utf-8")

    assert get_next_episode_index(tmp_path) == 15


def test_interpolate_joint_positions_hits_target_at_last_step() -> None:
    start = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    goal = np.array([1.0, 1.5, 2.0], dtype=np.float32)

    trajectory = interpolate_joint_positions(start, goal, num_steps=4)

    assert trajectory.shape == (4, 3)
    np.testing.assert_allclose(trajectory[-1], goal)
    assert np.all(trajectory[0] > start)


def test_build_pick_place_stage_targets_uses_expected_sequence() -> None:
    cube_position = np.array([0.4, 0.2, 0.1], dtype=np.float32)
    pallet_position = np.array([0.8, -0.1, 0.05], dtype=np.float32)

    stages = build_pick_place_stage_targets(
        cube_position,
        pallet_position,
        grasp_height_offset=0.01,
        approach_height=0.12,
        lift_height=0.16,
        place_height_offset=0.03,
        finger_open=0.14,
        finger_closed=0.02,
    )

    assert [stage.name for stage in stages] == [
        "move_above_cube",
        "move_to_grasp",
        "close_gripper",
        "lift_cube",
        "move_above_pallet",
        "move_to_place",
        "open_gripper",
        "retreat",
    ]
    assert all(isinstance(stage, StageTarget) for stage in stages)
    np.testing.assert_allclose(stages[0].position, np.array([0.4, 0.2, 0.22], dtype=np.float32))
    np.testing.assert_allclose(stages[1].position, np.array([0.4, 0.2, 0.11], dtype=np.float32))
    assert stages[2].finger_joint_target == np.float32(0.02)
    np.testing.assert_allclose(stages[-1].position, np.array([0.8, -0.1, 0.24], dtype=np.float32))
