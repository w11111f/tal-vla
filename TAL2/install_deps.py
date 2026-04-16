import yaml
import subprocess
import os

# 1. 你的路径配置
YAML_PATH = "/root/gpufree-data/code/TAL2/settings/tal.yaml"
ISAAC_LAB_SH = "/root/isaaclab/isaaclab.sh" # 请确保这是你真实的 IsaacLab 路径

with open(YAML_PATH, 'r') as f:
    data = yaml.safe_load(f)

deps = data.get('dependencies', [])
pip_deps = []
conda_deps = []

for dep in deps:
    if isinstance(dep, str):
        # 过滤掉 python 和可能冲突的底层包
        if not dep.startswith('python') and not dep.startswith('cuda'):
            conda_deps.append(dep.split('=')[0])
    elif isinstance(dep, dict) and 'pip' in dep:
        pip_deps.extend([p.split('=')[0] for p in dep['pip']])

all_packages = conda_deps + pip_deps
print(f"准备安装包: {all_packages}")

# 2. 调用 Isaac Lab 的 pip
for pkg in all_packages:
    print(f"正在安装 {pkg}...")
    subprocess.run([ISAAC_LAB_SH, "-p", "-m", "pip", "install", pkg])