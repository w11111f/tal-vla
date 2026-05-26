"""
将自定义 HDF5 数据集转换为 LeRobot 格式的脚本。
修正版：移除了不兼容的 consolidate()，增加了路径检查。
"""

import shutil
from pathlib import Path
import h5py
import numpy as np
import tyro
import os
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME

# 你的数据集名称
REPO_NAME = "sim/pickdata2"

def main(
    data_dir: str = "/root/gpufree-data/h5_dataset",
    fps: int = 20,
    robot_type: str = "custom_7dof",
    push_to_hub: bool = False,
):
    data_path = Path(data_dir)
    h5_files = sorted(list(data_path.glob("*.h5")) + list(data_path.glob("*.hdf5")))
    
    if not h5_files:
        print(f"错误: 在 {data_dir} 下未找到 .h5 或 .hdf5 文件")
        return

    print(f"找到 {len(h5_files)} 个数据文件，准备开始转换...")

    # --- 1. 读取第一个文件获取维度信息 ---
    with h5py.File(h5_files[0], "r") as f:
        img_high_shape = f["cam_high"].shape[1:] 
        img_wrist_shape = f["cam_wrist"].shape[1:]
        state_shape = f["state"].shape[1:]
        action_shape = f["action"].shape[1:]
        
        print(f"检测到数据维度:")
        print(f"  - cam_high: {img_high_shape}")
        print(f"  - cam_wrist: {img_wrist_shape}")
        print(f"  - state: {state_shape}")
        print(f"  - action: {action_shape}")

    # --- 2. 清理旧目录 ---
    output_path = HF_LEROBOT_HOME / REPO_NAME
    # if output_path.exists():
    #     print(f"清理旧数据目录: {output_path}")
    #     shutil.rmtree(output_path)

    # --- 3. 创建 LeRobot 数据集实例 ---
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type=robot_type,
        fps=fps,
        features={
            "cam_high": {
                "dtype": "image",
                "shape": img_high_shape,
                "names": ["height", "width", "channels"],
            },
            "cam_wrist": {
                "dtype": "image",
                "shape": img_wrist_shape,
                "names": ["height", "width", "channels"],
            },
            "state": {
                "dtype": "float32",
                "shape": state_shape,
                "names": ["state"], 
            },
            "action": {
                "dtype": "float32",
                "shape": action_shape,
                "names": ["action"], 
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # --- 4. 遍历文件并写入数据 ---
    try:
        for i, file_path in enumerate(h5_files):
            print(f"[{i+1}/{len(h5_files)}] 处理文件: {file_path.name}")
            
            with h5py.File(file_path, "r") as f:
                length = f["action"].shape[0]
                
                # 获取 Prompt
                if "prompt" in f:
                    raw_prompt = f["prompt"]
                else:
                    raw_prompt = np.array([b"pick up object"] * length)

                for t in range(length):
                    # Prompt 处理逻辑
                    if raw_prompt.shape == () or raw_prompt.shape == (1,):
                        current_prompt = raw_prompt[()] if raw_prompt.shape == () else raw_prompt[0]
                    else:
                        current_prompt = raw_prompt[t]
                    
                    if isinstance(current_prompt, bytes):
                        current_prompt = current_prompt.decode("utf-8")

                    state_data = f["state"][t].astype(np.float32)
                    action_data = f["action"][t].astype(np.float32)
                    
                    frame = {
                        "cam_high": f["cam_high"][t],
                        "cam_wrist": f["cam_wrist"][t],
                        "state": state_data,
                        "action": action_data,
                        "task": current_prompt,
                    }
                    
                    dataset.add_frame(frame)
                
                # 保存这一集 (这会自动更新 episodes.jsonl)
                dataset.save_episode()

        # [修改] 移除了 dataset.consolidate()

    except Exception as e:
        print(f"转换过程中发生错误: {e}")
        return

    # --- 5. 验证文件生成 ---
    meta_dir = output_path / "meta"
    print("\n转换完成！")
    print(f"数据集路径: {output_path}")
    
    if meta_dir.exists():
        files = list(meta_dir.glob("*"))
        print(f"Meta 文件夹内容: {[f.name for f in files]}")
        
        # 强制检查 episodes.jsonl
        episodes_path = meta_dir / "episodes.jsonl"
        if episodes_path.exists():
            print(f"SUCCESS: {episodes_path} 已生成。")
        else:
            print(f"WARNING: {episodes_path} 未找到！可能是路径权限问题。")
    else:
        print(f"ERROR: Meta 文件夹未生成: {meta_dir}")

    # 打印给用户后续的操作指南
    print("\n下一步操作指南:")
    print(f"1. 确认环境变量: export HF_LEROBOT_HOME={str(HF_LEROBOT_HOME.parent)}")
    print(f"2. 建立软链(如果需要): ln -s {output_path} /root/.cache/huggingface/lerobot/{REPO_NAME}")
    print(f"3. 计算统计量: uv run scripts/compute_norm_stats.py --config-name pi05_pro630_lora")

if __name__ == "__main__":
    tyro.cli(main)