import torch
from tqdm import tqdm
from src.envs import approx
from src.tal.action_proposal_network import vec2action_grammatical


def _safe_numeric_suffix(name, default=0):
    digits = ''.join(ch for ch in str(name) if ch.isdigit())
    return int(digits) if digits else default


def test_policy_pt(config, dataset, model, num_objects=0, TQDM=True, STATE_FORMAT_GOAL=True,
                   INIT_DATAPOINT=True):
    model.eval()
    assert 'action' in config.training
    e = config.embeddings
    context_len = config.context_len
    correct, incorrect, error = 0, 0, 0
    lenHuman, lenModel = [], []
    data_num = {}
    data_correct_num = {}

    data_container = tqdm(dataset, desc='Policy Testing', ncols=80) if TQDM else dataset
    for data_item in data_container:
        if STATE_FORMAT_GOAL:  # * GraphDataset_State
            graphSeq, goal2vec, goal_json, actionSeq, action2vec, world_name, start_node = data_item
        else:  # * GraphDataset
            graphSeq, goal2vec, goalObjects2vec, actionSeq, action2vec, goal_json, world_name, start_node = data_item

        # * Store action length to dict.
        if str(len(actionSeq)) in data_num:
            data_num[str(len(actionSeq))] += 1
        else:
            data_num[str(len(actionSeq))] = 1
        if str(len(actionSeq)) not in data_correct_num:
            data_correct_num[str(len(actionSeq))] = 0

        world_num = _safe_numeric_suffix(world_name, default=0)
        plan_len = len(actionSeq)
        lenHuman.append(plan_len)
        predActionSeq = []  # * Initialize
        graphSeq_t = []
        action2vec_t = []

        # * Initialize environment.
        if INIT_DATAPOINT:
            approx.initPolicy(config, config.domain, goal_json, world_num, SET_GAOL_JSON=True,
                              INPUT_DATAPOINT=start_node)
            graphSeq_t.append(graphSeq[0])
        else:
            # * Use wold_home1 to test.
            # world_num = 1
            approx.initPolicy(config, config.domain, goal_json, world_num, SET_GAOL_JSON=True)
            init_g = approx.getInitializeDGLGraph(config)
            if config.device is not None:
                init_g = init_g.to(config.device)
            graphSeq_t.append(init_g)

        # * Test model.
        y_pred_list = []
        while True:
            traj_len = len(graphSeq_t)
            if len(graphSeq_t) >= context_len:
                states = graphSeq_t[-context_len:]
                actions = action2vec_t[-context_len:] if len(
                    action2vec_t) > context_len else action2vec_t
                start_time_steps = max(0, traj_len - context_len)
                time_steps = torch.arange(start=start_time_steps,
                                          end=start_time_steps + context_len, step=1).to(
                    config.device)
            else:  # * Less than context_len.
                states = graphSeq_t
                actions = action2vec_t
                time_steps = torch.arange(start=0, end=context_len, step=1).to(config.device)

            with torch.no_grad():
                action_preds = model(time_steps=time_steps, prompt_state=goal2vec, states=states,
                                     actions=actions)
                # action_preds = model(prompt_state=goal2vec, states=states, actions=actions)
                # action_preds = action_preds[0, -1].detach()

                if len(graphSeq_t) >= context_len:
                    action_preds = action_preds[0, -1].detach()
                else:
                    action_preds = action_preds[0, len(graphSeq_t) - 1].detach()
                action2vec_t.append(action_preds)
                y_pred_list.append(action_preds)

            y_pred = y_pred_list[-1]
            action_hl = vec2action_grammatical(
                config, y_pred, num_objects, len(config.possibleStates), config.idx2object
            )

            # * !!!
            res, g, err = approx.execAction(config, action_hl, e)
            predActionSeq.append(action_hl)
            if g is not None and config.device is not None:
                g = g.to(config.device)
            graphSeq_t.append(g)

            # * If (res == False) and (err == '') and (len(actionSeq) < xxx): continue
            if res:
                data_correct_num[str(len(actionSeq))] += 1
                correct += 1
                lenModel.append(len(predActionSeq))
                break
            elif err == '' and len(predActionSeq) > 60:
                incorrect += 1
                break
            elif err != '':
                error += 1
                break

    print('--' * 20)
    fmt = '{:^20}\t{:^15,.4f}\t{:^10}\t{:^10}'
    print('Action sequence length  |  Accuracy  |  Data num  |  Correct num')
    key_items = list(data_num.keys())
    key_items.sort(key=lambda x: int(x))  # * Convert str to int.
    for key in key_items:
        value = data_num[key]
        data_accuracy = data_correct_num[key] / value * 100
        print(fmt.format(key, data_accuracy, value, data_correct_num[key]))
    print('--' * 20)

    den = correct + incorrect + error
    print('Correct num, incorrect num, error num: ', correct, incorrect, error)
    print(
        'Correct, Incorrect, Error: ',
        (correct * 100 / den), (incorrect * 100 / den), (error * 100 / den)
    )
    return (correct * 100 / den), (incorrect * 100 / den), (error * 100 / den), lenHuman, lenModel
