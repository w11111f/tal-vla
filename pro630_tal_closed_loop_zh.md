# pro630 闭环 TAL + OpenPI 联调说明

本文档说明如何在服务器上测试当前已经接入的闭环推理链路，并说明后续如何把当前的 `RGB -> scene graph` 占位接口替换成你自己的视觉感知模块。

适用对象：
- 你已经在服务器上准备好了 `openpi` 运行环境
- 你已经准备好了 `TAL2` 代码和相关模型权重
- 你希望测试 `pi05_pro630_lora` 的闭环推理流程

本文档对应的代码改动主要在：
- [src/openpi/tal_runtime.py](tal-vla/openpi/src/openpi/tal_runtime.py)
- [src/openpi/policies/pro630demo_policy.py](tal-vla/openpi/src/openpi/policies/pro630demo_policy.py)
- [src/openpi/training/config.py](tal-vla/openpi/src/openpi/training/config.py)
- [src/openpi/policies/pro630demo_policy_test.py](tal-vla/openpi/src/openpi/policies/pro630demo_policy_test.py)

## 1. 当前已经实现的闭环流程

当前 `pi05_pro630_lora` 的推理流程已经按下面这条链路接好了：

1. 输入 `RGB 图像 + 机器人状态 + 用户自然语言指令`
2. 优先尝试获取当前场景图
3. TAL 根据 `用户指令 + 当前场景图` 做多步规划
4. 取 TAL 规划出的首个动作
5. 将提示词融合成：
   - `User task: <原始指令>.`
   - `Current subtask: <TAL 首个动作>.`
6. OpenPI `pi05_pro630_lora` 接收：
   - `RGB 图像`
   - `state`
   - 融合后的 prompt
7. OpenPI 输出动作
8. 外部机器人执行动作
9. 下一轮推理时，如果达到重规划条件，则重新获取场景图并再次调用 TAL

当前版本的重规划策略是：
- 默认每次推理都重规划一次
- 也支持按固定步数重规划
- 也支持按超时重规划
- 当前版本不在 OpenPI 内部做“子任务完成判定”

## 2. 当前代码中各模块职责

### 2.1 `tal_runtime.py`

文件：
- [src/openpi/tal_runtime.py](/c:/Users/w'y'f/OneDrive/Desktop/tal-vla/openpi/src/openpi/tal_runtime.py)

这个文件负责闭环运行时逻辑，主要有四部分：

1. `TALRuntimeConfig`
   - 保存 TAL 运行时配置
   - 包括 TAL 开关、TAL 根目录、Qwen 配置、重规划步数、超时等

2. `SceneGraphProvider`
   - 这是一个接口协议
   - 你未来的视觉模块最终要实现这里定义的 `extract(images, state)` 方法

3. `LazyTALPlanner`
   - 懒加载 TAL 模型
   - 首次调用时才会导入 TAL2 模块、加载 AFE/APN 和特征文件
   - 后续重复使用

4. `ClosedLoopTALManager`
   - 管理闭环状态
   - 记录当前任务指令
   - 记录上一次 TAL 结果
   - 记录当前步数
   - 记录是否需要重规划

### 2.2 `pro630demo_policy.py`

文件：
- [src/openpi/policies/pro630demo_policy.py](/c:/Users/w'y'f/OneDrive/Desktop/tal-vla/openpi/src/openpi/policies/pro630demo_policy.py)

这个文件负责把外部请求转换成 OpenPI 模型输入。

当前有三个关键类：

1. `Pro630InputAdapter`
   - 兼容训练数据格式和运行时格式
   - 把外部的 `cam_high`、`cam_wrist`、`state`、`prompt` 等键统一成内部格式

2. `CustomInputs`
   - 真正构造 OpenPI 模型输入
   - 会把图像映射到：
     - `base_0_rgb`
     - `left_wrist_0_rgb`
     - `right_wrist_0_rgb`
   - 会在推理时调用 TAL 闭环管理器，得到融合后的 prompt

3. `CustomOutputs`
   - 将 OpenPI 模型输出的动作裁剪回 pro630 使用的动作维度

### 2.3 `config.py`

文件：
- [src/openpi/training/config.py](/c:/Users/w'y'f/OneDrive/Desktop/tal-vla/openpi/src/openpi/training/config.py)

这里已经把：
- `pi05_pro630_lora`
- `LeRobotpro630DataConfig`

接入到了新的 TAL-aware 闭环路径中。

默认配置包括：
- `tal_enabled=True`
- `tal_repo_root="."`
- `tal_qwen_api_key_env="DASHSCOPE_API_KEY"`
- `replan_every_n_steps=1`
- `fallback_to_raw_prompt=True`

注意：
- `tal_repo_root="."` 时，代码会自动尝试：
  - 当前目录本身是不是 TAL2 根目录
  - 当前目录下有没有 `TAL2`
  - 当前目录上一级有没有 `TAL2`

因此你在服务器上通常有两种目录结构都可以工作：

结构 1：
```text
/path/to/workspace/
  openpi/
  TAL2/
```

结构 2：
```text
/path/to/TAL2/
  src/
  settings/
```

## 3. 当前支持的推理输入字段

当前 pro630 推理链路支持以下请求字段。

### 3.1 必需字段

- `cam_high`
  - 主视角 RGB 图像
  - 推荐格式：`numpy.ndarray`
  - 形状：`(H, W, 3)` 或 `(3, H, W)`

- `cam_wrist`
  - 腕部 RGB 图像
  - 推荐格式：`numpy.ndarray`
  - 形状：`(H, W, 3)` 或 `(3, H, W)`

- `state`
  - 机器人状态
  - 推荐格式：`numpy.ndarray`
  - 当前假设是 7 维

- `prompt`
  - 用户原始自然语言指令

### 3.2 可选字段

- `current_scene_graph_json`
  - 当前场景图 JSON
  - 如果你直接传这个字段，系统就不会调用 `SceneGraphProvider`
  - 这是当前最推荐的联调方式

- `reset_tal_context`
  - 布尔值
  - 用于手动清空闭环上下文
  - 当你开始新任务、场景强制重置、机器人 episode 切换时建议设置为 `True`

## 4. 当前场景图获取优先级

当前代码中，场景图来源优先级如下：

1. 如果推理请求中直接包含 `current_scene_graph_json`
   - 直接使用这个场景图
   - 不调用视觉接口

2. 如果请求里没有 `current_scene_graph_json`
   - 尝试调用 `SceneGraphProvider.extract(images, state)`

3. 如果视觉接口没有实现，或者调用失败
   - 如果 `fallback_to_raw_prompt=True`
     - 回退到原始用户 prompt
     - OpenPI 仍可继续执行
   - 如果 `fallback_to_raw_prompt=False`
     - 直接抛异常

## 5. 服务器测试前建议检查

建议先确认以下条件：

1. `openpi` 的 Python 依赖已经装好
2. `TAL2` 的 Python 依赖已经装好
3. `TAL2` 所需的 AFE / APN / `action_effect_features_avg.pkl` 已经存在
4. DashScope / Qwen 所需环境变量已经设置
5. `pi05_pro630_lora` 使用的 OpenPI checkpoint 已经存在

常见环境变量示例：

```bash
export DASHSCOPE_API_KEY=你的key
```

如果你不是用环境变量，也可以在后面直接修改：
- [src/openpi/training/config.py](/c:/Users/w'y'f/OneDrive/Desktop/tal-vla/openpi/src/openpi/training/config.py)
- `LeRobotpro630DataConfig(... tal_qwen_api_key_env=...)`

## 6. 推荐的目录结构

推荐在服务器上使用下面这种结构：

```text
/data/your_workspace/
  openpi/
  TAL2/
```

这样 `pi05_pro630_lora` 中的：
- `tal_repo_root="."`

可以比较容易改成你自己的绝对路径，例如：

```python
tal_repo_root="/data/your_workspace"
```

或者：

```python
tal_repo_root="/data/your_workspace/TAL2"
```

都可以。

## 7. 如何启动 OpenPI policy server

当前 `scripts/serve_policy.py` 支持从指定 config 和 checkpoint 启动策略服务。

在服务器上建议进入 `openpi` 根目录执行：

```bash
cd /data/your_workspace/openpi
python scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_pro630_lora \
  --policy.dir=/path/to/your/checkpoint_dir
```

说明：
- `--policy.config=pi05_pro630_lora`
  - 使用当前已经接好的 pro630 + TAL 闭环配置
- `--policy.dir`
  - 指向你服务器上的实际 checkpoint 目录

如果你已经把 `weight_loader` 固定在配置里，也可以按你自己的运行方式调整。

## 8. 如何发送一次推理请求

OpenPI 默认用 websocket server，你可以用你自己的控制程序，也可以参考 `examples/simple_client` 的格式。

一次最小可用请求建议包含：

```python
request_data = {
    "cam_high": cam_high_image,          # numpy.ndarray
    "cam_wrist": cam_wrist_image,        # numpy.ndarray
    "state": robot_state,                # numpy.ndarray, shape [7]
    "prompt": "pick up the blue bottle",
    "current_scene_graph_json": {
        "nodes": [
            {"name": "husky", "states": ["free"]},
            {"name": "bottle_blue", "states": ["outside", "not_fueled"]},
            {"name": "tray", "states": ["same_height"]},
            {"name": "table", "states": ["same_height"]}
        ],
        "edges": [
            {"from": "bottle_blue", "to": "table", "relation": "On"}
        ]
    }
}
```

当前这是一种最稳的联调方式，因为：
- 你还没实现 RGB 到 scene graph 的视觉模块
- 但你已经可以先验证：
  - TAL 是否会根据场景图和指令规划
  - TAL 首动作是否会拼进 prompt
  - OpenPI 是否会按融合 prompt 输出动作

## 9. 闭环测试时的建议调用方式

一个简单的闭环控制逻辑可以是：

1. 从相机读取最新 RGB 图像
2. 从机器人读取最新状态
3. 生成或提供 `current_scene_graph_json`
4. 调用一次 policy server
5. 执行输出动作
6. 重复 1 到 5

如果你要切换到新的任务或新的 episode，建议下一轮请求额外加上：

```python
"reset_tal_context": True
```

这样会清空：
- 当前记录的任务
- TAL 上一轮规划结果
- 重规划计数器

## 10. 如何运行当前新增的单元测试

当前我新增的测试文件是：
- [src/openpi/policies/pro630demo_policy_test.py](/c:/Users/w'y'f/OneDrive/Desktop/tal-vla/openpi/src/openpi/policies/pro630demo_policy_test.py)

建议在 `openpi` 根目录执行：

```bash
pytest src/openpi/policies/pro630demo_policy_test.py
```

如果你环境里没有 GPU，也没关系，`conftest.py` 会尝试设置 CPU backend。

这些测试主要验证：
- prompt 是否正确融合
- 是否优先使用 `current_scene_graph_json`
- 是否按步数重规划
- TAL 失败时是否正确回退到原始 prompt
- 训练批次是否跳过 TAL
- `reset_tal_context` 是否生效
- timeout 是否能触发重规划

## 11. 如何确认 TAL 已真正参与推理

最简单的确认方式有三种：

### 11.1 查看融合后的 prompt

你可以在：
- [src/openpi/policies/pro630demo_policy.py](/c:/Users/w'y'f/OneDrive/Desktop/tal-vla/openpi/src/openpi/policies/pro630demo_policy.py)

中的 `CustomInputs._compute_prompt()` 附近临时加日志，打印：
- 原始 prompt
- TAL 首动作
- 最终融合 prompt

最终你应该能看到类似内容：

```text
User task: pick up the blue bottle.
Current subtask: moveTo bottle_blue
```

### 11.2 开启 TAL debug

在：
- [src/openpi/training/config.py](/c:/Users/w'y'f/OneDrive/Desktop/tal-vla/openpi/src/openpi/training/config.py)

里将：

```python
enable_tal_debug=True
```

这样 `ClosedLoopTALManager` 会缓存最近一次 TAL 结果。

### 11.3 使用可控场景图做对照实验

建议准备两组不同的 `current_scene_graph_json`：
- 场景图 A
- 场景图 B

用户指令保持不变，比较 TAL 首动作是否变化。

如果变化了，说明 TAL 链路已经真正参与了推理。

## 12. 如何修改重规划频率

当前重规划逻辑在：
- [src/openpi/tal_runtime.py](/c:/Users/w'y'f/OneDrive/Desktop/tal-vla/openpi/src/openpi/tal_runtime.py)

核心逻辑在：
- `ClosedLoopTALManager._should_replan()`

你可以通过配置控制：

### 12.1 每 N 步重规划

在：
- [src/openpi/training/config.py](/c:/Users/w'y'f/OneDrive/Desktop/tal-vla/openpi/src/openpi/training/config.py)

设置：

```python
replan_every_n_steps=5
```

表示：
- TAL 第一次会立即规划
- 后续每执行 5 次推理，再重新规划一次

### 12.2 超时重规划

例如：

```python
replan_timeout_s=2.0
```

表示：
- 如果距离上次成功重规划已经超过 2 秒
- 下一次推理时将触发重新规划

## 13. 如何添加真正的 `RGB -> scene graph` 功能

这是你后续最关键的扩展点。

### 13.1 当前的接口定义在哪里

接口定义在：
- [src/openpi/tal_runtime.py](/c:/Users/w'y'f/OneDrive/Desktop/tal-vla/openpi/src/openpi/tal_runtime.py)

具体是：

```python
class SceneGraphProvider(Protocol):
    def extract(self, images: Mapping[str, Any], state: Any) -> dict[str, Any]:
        ...
```

当前默认实现是：

```python
class MissingSceneGraphProvider:
    def extract(self, images, state):
        raise NotImplementedError(...)
```

### 13.2 你应该如何接入自己的视觉模块

你需要新建一个类，实现 `extract(images, state)`。

建议新建文件：

```text
openpi/src/openpi/scene_graph_provider_pro630.py
```

示例骨架如下：

```python
from __future__ import annotations

from typing import Any
from collections.abc import Mapping


class Pro630SceneGraphProvider:
    def __init__(self):
        # 在这里加载你的检测模型、分割模型、关系推理模块等
        pass

    def extract(self, images: Mapping[str, Any], state: Any) -> dict[str, Any]:
        cam_high = images["cam_high"]
        cam_wrist = images["cam_wrist"]

        # 1. 目标检测 / 分割
        # 2. 物体追踪或实例匹配
        # 3. 关系判断：On / Inside / Close / Stuck
        # 4. 状态判断：Grabbed / Free / Up / Down / Fueled ...
        # 5. 拼成 TAL 需要的 scene_graph_json

        scene_graph_json = {
            "nodes": [
                {"name": "husky", "states": ["free"]},
                {"name": "bottle_blue", "states": ["outside"]},
                {"name": "table", "states": ["same_height"]},
            ],
            "edges": [
                {"from": "bottle_blue", "to": "table", "relation": "On"}
            ],
        }
        return scene_graph_json
```

### 13.3 然后如何让系统使用这个类

在：
- [src/openpi/training/config.py](/c:/Users/w'y'f/OneDrive/Desktop/tal-vla/openpi/src/openpi/training/config.py)

找到 `pi05_pro630_lora` 使用的：
- `LeRobotpro630DataConfig(...)`

加上：

```python
scene_graph_provider_cls="openpi.scene_graph_provider_pro630.Pro630SceneGraphProvider"
```

这样运行时就会自动动态导入你的类。

### 13.4 `extract()` 的输入是什么

`extract(images, state)` 中：

`images` 当前会包含：
- `images["cam_high"]`
- `images["cam_wrist"]`

`state` 当前就是：
- `numpy.ndarray`
- 通常是 7 维机器人状态

### 13.5 `extract()` 的输出必须是什么样

必须返回 TAL 能接受的 `scene_graph_json`，也就是：

```python
{
    "nodes": [...],
    "edges": [...],
}
```

其中：
- `nodes` 表示物体及其状态
- `edges` 表示物体间关系

建议你先参考：
- [TAL2/src/tal/scene_graph_translator.py](/c:/Users/w'y'f/OneDrive/Desktop/tal-vla/TAL2/src/tal/scene_graph_translator.py)

特别是：
- `scene_graph_json_to_dgl(...)`
- `canonicalize_scene_graph_json(...)`

因为最终 TAL 会把你的 JSON 转成图结构。

### 13.6 推荐的开发顺序

建议按下面顺序开发：

1. 先人工构造 `current_scene_graph_json`，验证 TAL + OpenPI 闭环可跑
2. 再实现一个最简单的 `SceneGraphProvider`
   - 只识别少量核心物体
   - 只支持少量关系
3. 再逐步增加：
   - 物体类别
   - 状态识别
   - 关系识别
   - 多相机融合

## 14. 未来如果你想加入“子任务完成判定”

当前版本没有在 OpenPI 里判断“当前 TAL 子任务是否完成”，只是按固定步数或超时重规划。

如果你未来要加这个能力，推荐在：
- [src/openpi/tal_runtime.py](/c:/Users/w'y'f/OneDrive/Desktop/tal-vla/openpi/src/openpi/tal_runtime.py)

里扩展 `ClosedLoopTALManager`，例如增加：

```python
def subtask_completed(self, current_scene_graph_json) -> bool:
    ...
```

可用的判定方式包括：
- 基于场景图变化
- 基于目标检测结果
- 基于末端执行器状态
- 基于机器人执行器回传的完成标志

## 15. 常见问题

### 15.1 没有提供 `current_scene_graph_json`，又没实现视觉接口，会怎样

如果：
- `tal_enabled=True`
- 没有 `current_scene_graph_json`
- `scene_graph_provider_cls` 也没配置或没实现

那么：
- 如果 `fallback_to_raw_prompt=True`
  - 会自动回退到原始 prompt
- 如果 `fallback_to_raw_prompt=False`
  - 会直接报错

### 15.2 TAL 权重加载失败怎么办

先检查：
- `TAL2` 根目录是否正确
- `checkpoints/home/` 下相关文件是否存在
- `action_effect_features_avg.pkl` 是否存在

### 15.3 Qwen 接口报错怎么办

先检查：
- `DASHSCOPE_API_KEY`
- 网络连通性
- `tal_qwen_model` 是否可用

### 15.4 OpenPI 正常运行但 TAL 好像没生效

优先检查：
- 是否传了 `current_scene_graph_json`
- 是否开启了 `tal_enabled=True`
- 是否被回退到了 raw prompt
- 是否在训练路径中带了 `actions`，从而触发了“训练批次跳过 TAL”

## 16. 建议的第一轮服务器联调顺序

建议按下面顺序进行，最稳：

1. 启动 `pi05_pro630_lora` policy server
2. 先手工传入 `current_scene_graph_json`
3. 观察返回动作是否稳定
4. 打印融合后的 prompt
5. 多次发送请求，观察重规划是否生效
6. 确认闭环正常后，再接入你自己的视觉 scene graph provider

## 17. 结论

当前代码已经具备：
- TAL 与 OpenPI 的闭环运行时接入
- 提示词融合
- 固定步数/超时重规划
- 手动注入场景图
- 后续视觉模块可扩展接口

你下一步最推荐的测试路线是：
- 先用手写 `current_scene_graph_json` 在服务器上跑通整条闭环
- 再实现 `SceneGraphProvider.extract(images, state)`
- 最后再根据你的真实机器人执行反馈增加更精细的“子任务完成判定”

如果你后面愿意，我也可以继续帮你补两样东西：
- 一个最小可运行的 websocket 客户端示例
- 一个 `scene_graph_provider_pro630.py` 模板文件，直接留好你后续填视觉算法的位置
