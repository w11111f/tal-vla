# 

1. Task-agnostic exploration to collect data.

```shell
/root/isaacsim/python.sh -m src.generate_Aall
/root/isaacsim/python.sh -m src.exploration
```

Notes:
- The Isaac Lab environment now loads `/root/Desktop/Collected_exp3/expff.usd`.
- The active scene objects are `Cube`, `SmallPallet`, `BigPallet`, `Bottle2`, `Stool`, `table`, and `Mobie_grasper2`.
- Generated graphs are saved under `data/home/world_expff/`.
- The repo now ships a lightweight local `dgl` compatibility package, so Isaac Lab's
  Python 3.11 / PyTorch 2.7 / CUDA 12.8 environment can run the project without
  installing the legacy external DGL build from `settings/tal.yaml`.
- The reduced symbolic state set is:
  `Outside`, `Inside`, `Up`, `Down`, `Grabbed`, `Free`, `Sticky`, `Non_Sticky`,
  `Fueled`, `Not_Fueled`, `Driven`, `Not_Driven`, `Different_Height`, `Same_Height`.
- The reduced action set is:
  `drop`, `pick`, `moveTo`, `pushTo`, `changeState`, `pickNplaceAonB`.

2. Generate dataset.

```shell
/root/isaacsim/python.sh -m src.generate_and_split_dataset
```

3. Train action effect feature extractor.

```shell
/root/isaacsim/python.sh -m scripts.train_feature_extractor
```

4. Extract action effect features.

```shell
/root/isaacsim/python.sh -m scripts.generate_action_effect_feartures
```

5. Train action proposal (BC)

```shell
/root/isaacsim/python.sh -m scripts.train_action_proposal
```

6. Test BC.

```shell
/root/isaacsim/python.sh -m scripts.test_policy_bc --max_samples N
```。

7. Test TAL.

```shell
/root/isaacsim/python.sh -m scripts.test_policy_tal
```

8. Test TAL with natural language goal translation via DashScope HTTP API.

```shell
/root/isaaclab/isaaclab.sh scripts/test_policy_tal_nl.py --instruction "Put the milk into the fridge."
```

This entrypoint uses `curl` to call DashScope. Set `DASHSCOPE_API_KEY` or pass `--qwen_api_key`.

8.5 接受自然语言指令，输出下一个动作
export DASHSCOPE_API_KEY=你的key
/root/isaaclab/isaaclab.sh -p scripts/test_next_action_tal_nl.py --instruction "pick up the cube."

9. Train baseline CQL.

```shell
python baseline_train_cql.py
```

10. Train baseline Plan Transformer.

```shell
python baseline_train_pt.py
```





* Since pickup - moveTo - drop can be a minimum sequence length to complete a small task, the length of sequence in training set is set to 1-3.
