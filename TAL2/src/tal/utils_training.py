"""
@File             : utils_training.py
@Project          : tango_i
@Time             : 2021/11/26 13:54
@Author           : Xianqi ZHANG
@Last Modify Time : 2022/06/20
@Desciption       : None   
"""
import os
import pickle
import random
import torch
from tqdm import tqdm
import src.tal.action_proposal_network
import src.tal.action_feature_extractor
from src.envs import approx
from src.utils.misc import Color
from src.datasets.utils_dataset import DGLDataset
from src.tal.action_proposal_network import vec2action_grammatical


def _safe_numeric_suffix(name, default=0):
    digits = ''.join(ch for ch in str(name) if ch.isdigit())
    return int(digits) if digits else default


def random_split(config, data):
    test_size = int(0.1 * len(data.graphs))
    random.shuffle(data.graphs)
    test_set = data.graphs[:test_size]
    train_set = data.graphs[test_size:]
    return train_set, test_set


def world_split(config, data):
    test_set = []
    train_set = []
    counter = 0
    for i in data.graphs:
        for j in range(1, 9):
            if (i[0], i[1]) == (j, (j if config.domain == 'home' else j - 1)):
                test_set.append(i)
                break
        else:
            counter += 1
            train_set.append(i)
    return train_set, test_set


def tool_split(config, data):
    train_set, test_set = world_split(config, data)
    tool_set, notool_set = [], []
    for graph in train_set:
        if 'no-tool' in graph[2]:
            notool_set.append(graph)
        else:
            tool_set.append(graph)
    new_set = []
    for i in range(len(tool_set) - len(notool_set)):
        new_set.append(random.choice(notool_set))
    train_set = tool_set + notool_set + new_set
    return train_set, test_set


def split_data(config, data):
    if config.split == 'world':
        train_set, test_set = world_split(config, data)
    elif config.split == 'random':
        train_set, test_set = random_split(config, data)
    else:
        train_set, test_set = tool_split(config, data)
    print('Size before split was', len(data.graphs))
    print('The size of the training set is', len(train_set))
    print('The size of the test set is', len(test_set))
    return train_set, test_set


def get_model(config, model_name, data_features, data_num_classes, n_layers=3):
    if model_name == 'DDQN':
        from src.baselines.cql_dqn.networks import DDQN
        model_class = DDQN
    elif model_name == 'PlanTransformer':
        from src.baselines.plan_transformer.networks import PlanTransformer
        model_class = PlanTransformer
    else:
        try:
            model_class = getattr(src.tal.action_proposal_network, model_name)
        except:
            model_class = getattr(src.tal.action_feature_extractor, model_name)

    model = model_class(config, data_features, data_num_classes, 2 * config.GRAPH_HIDDEN,
                        len(config.possibleStates), n_layers, config.etypes, torch.tanh, 0.5)
    return model


def load_model(config, filename, model, file_path=None, lr=None, strict=True):
    lr = lr if lr is not None else 0.0005
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    if file_path is None:
        file_path = config.MODEL_SAVE_PATH + '/' + filename + '.ckpt'
    if os.path.exists(file_path):
        print(Color.GREEN + 'Loading pre-trained model: ' + file_path + Color.ENDC)
        checkpoint = torch.load(file_path, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        epoch = checkpoint['epoch']
        accuracy_list = checkpoint['accuracy_list']
    else:
        if 'train' not in config.exec_type:
            print("File '%s' not found!" % filename)
            exit()
        epoch = -1
        accuracy_list = []
        print(Color.GREEN + 'Creating new model: ' + model.name + Color.ENDC)
    return model, optimizer, epoch, accuracy_list


def save_model(config, model, optimizer, epoch, accuracy_list, file_path=None, target_net=False):
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    seqTool = 'Seq_' if config.training == 'gcn_seq' else ''
    if file_path == None:
        if target_net:
            file_path = config.MODEL_SAVE_PATH + '/' + seqTool + model.name + '_target_' + str(
                epoch) + '.ckpt'
        else:
            file_path = config.MODEL_SAVE_PATH + '/' + seqTool + model.name + '_' + str(
                epoch) + '.ckpt'
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            'accuracy_list': accuracy_list
        },
        file_path
    )
    return file_path


def convert_model(config, filename, training_type, features, num_objects):
    file_path = config.MODEL_SAVE_PATH + '/' + filename + '.pt'
    assert (os.path.exists(file_path))
    model = torch.load(file_path)
    flag = False
    if str(type(
            model)) == "<class 'collections.OrderedDict'>":  # If only model state dict is saved
        flag = True
        model = get_model(config, config.model_name, features, num_objects)
        model.load_state_dict(torch.load(file_path))
    lr = 0.0005 if 'action' in training_type else 0.00005
    if training_type == 'gcn_seq': lr = 0.0005
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if flag:
        save_model(config, model, optimizer, 0, [], file_path=file_path)
    else:
        save_model(config, model, optimizer, 0, [])


def load_dataset(config, root_dir='data/'):
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    filename = (root_dir + config.domain + '_' +
                ('global_' if config.globalnode else '') +
                ('NoTool_' if not config.ignoreNoTool else '') +
                ('seq_' if config.sequence else '') +
                (config.embedding) +
                str(config.AUGMENTATION) + '.pkl')
    print(filename)
    if config.globalnode: config.etypes.append('Global')
    if os.path.exists(filename):
        return pickle.load(open(filename, 'rb'))
    data = DGLDataset(config,
                      root_dir + config.domain + '/',
                      augmentation=config.AUGMENTATION,
                      globalNode=config.globalnode,
                      ignoreNoTool=config.ignoreNoTool,
                      sequence=config.sequence,
                      embedding=config.embedding)
    pickle.dump(data, open(filename, 'wb'))
    return data


def grammatical_action(config, action):
    if (action['name'] in ['pushTo', 'pickNplaceAonB', 'dropTo', 'apply', 'stick']):
        if (len(action['args']) != 2):
            return False
        for i in action['args']:
            if i not in config.object2idx:
                return False
    elif action['name'] in ['moveTo', 'pick', 'drop', 'climbUp', 'climbDown', 'clean']:
        if (len(action['args']) != 1):
            return False
    elif action['name'] in config.noArgumentActions:
        if (len(action['args']) != 0):
            return False
    return True


# Graph data.
def accuracy_score_graph_dataset(config, dataset, model, num_objects=0, TQDM=True):
    model.eval()
    total_correct = 0
    denominator = 0
    stuttering = 0
    data_container = tqdm(dataset, desc='Accuracy Score', ncols=80) if TQDM else dataset
    for (
            graphSeq, goal2vec, goalObjects2vec, actionSeq, action2vec, goal_json, world_name,
            start_node
    ) in data_container:
        with torch.no_grad():
            # y_pred_list = model(graphSeq, goal2vec, goalObjects2vec, actionSeq)
            y_pred_list = []
            for i in range(len(graphSeq)):
                if 'GoalObject' in config.model_name:
                    output = model(graphSeq[i], goal2vec, goalObjects2vec)
                else:
                    output = model(graphSeq[i], goal2vec)
                y_pred_list.append(output)

        for i, y_pred in enumerate(y_pred_list):
            denominator += 1
            action_pred = vec2action_grammatical(
                config, y_pred, num_objects, len(config.possibleStates), config.idx2object
            )
            if (action_pred == actionSeq[i]):
                total_correct += 1
            if (
                    len(action_pred['args']) == 2
                    and action_pred['args'][0] == action_pred['args'][1]
            ):
                stuttering += 1

    return ((total_correct / denominator) * 100)


def accuracy_score_graph_dataset_state(config, dataset, model, num_objects=0, TQDM=True):
    """GraphDataset_State: Use final state as goal vector."""
    model.eval()
    total_correct = 0
    denominator = 0
    data_container = tqdm(dataset, desc='Accuracy Score', ncols=80) if TQDM else dataset
    for (graphSeq, goal2vec, goal_json, actionSeq, action2vec, world_name,
         start_node) in data_container:
        with torch.no_grad():
            y_pred_list = []
            for i in range(len(graphSeq)):
                output = model(graphSeq[i], goal2vec)
                y_pred_list.append(output)

        for i, y_pred in enumerate(y_pred_list):
            denominator += 1
            action_pred = vec2action_grammatical(
                config, y_pred, num_objects, len(config.possibleStates), config.idx2object
            )
            if (action_pred == actionSeq[i]):
                total_correct += 1
    return ((total_correct / denominator) * 100)


def accuracy_score_graph_dataset_state_mask(config, dataset, model, num_objects=0, TQDM=True):
    """GraphDataset_State: Use final state as goal vector."""
    model.eval()
    total_correct = 0
    denominator = 0
    data_container = tqdm(dataset, desc='Accuracy Score', ncols=80) if TQDM else dataset
    for (graphSeq, goal2vec, goal_json, actionSeq, action2vec, world_name,
         start_node) in data_container:
        start_state = graphSeq[0].ndata['feat']
        target_state = goal2vec.ndata['feat']
        goal_attn = target_state - start_state
        mask = torch.abs(goal_attn) > 0.1

        with torch.no_grad():
            y_pred_list = []
            for i in range(len(graphSeq)):
                output = model(graphSeq[i], goal2vec, mask)
                y_pred_list.append(output)

        for i, y_pred in enumerate(y_pred_list):
            denominator += 1
            action_pred = vec2action_grammatical(
                config, y_pred, num_objects, len(config.possibleStates), config.idx2object
            )
            if (action_pred == actionSeq[i]):
                total_correct += 1
    return ((total_correct / denominator) * 100)


def test_policy_graph_dataset(
        config,
        dataset,
        model,
        num_objects=0,
        TQDM=True,
        TEST_GT=False,
        STATE_FORMAT_GOAL=True,
        INIT_DATAPOINT=True
):
    model.eval()
    assert 'action' in config.training
    e = config.embeddings

    correct, incorrect, error = 0, 0, 0
    buckets = []
    lenHuman, lenModel = [], []
    for i in range(30): buckets.append([0, 0, 0])
    for i in range(8):
        lenHuman.append([])
        lenModel.append([])

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

        # * ------------------------------------------------------------
        if TEST_GT:
            # * Test ground truth.
            for action_gt in actionSeq:
                # * !!!
                res, g, err = approx.execAction(config, action_gt, e)
                predActionSeq.append(action_gt)

                # * If (res == False) and (err == '') and (len(actionSeq) < xxx): continue
                if res:
                    data_correct_num[str(len(actionSeq))] += 1
                    correct += 1
                    lenModel.append(len(predActionSeq))
                    break
                elif (err == '') and (len(predActionSeq) == len(actionSeq)):
                    incorrect += 1
                    break
                elif err != '':
                    error += 1
                    break
        else:
            # * Test model.
            y_pred_list = []
            while True:
                with torch.no_grad():
                    if 'GoalObject' in config.model_name:
                        output = model(graphSeq_t[-1], goal2vec, goalObjects2vec)
                    else:
                        output = model(graphSeq_t[-1], goal2vec)
                    y_pred_list.append(output)
                y_pred = y_pred_list[-1]
                action_pred = vec2action_grammatical(
                    config, y_pred, num_objects, len(config.possibleStates), config.idx2object
                )

                # * !!!
                res, g, err = approx.execAction(config, action_pred, e)
                predActionSeq.append(action_pred)
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

    # print('key_items:')
    # print(key_items)

    key_items.sort(key=lambda x: int(x))  # * Convert str to int.
    for key in key_items:
        value = data_num[key]
        data_accuracy = data_correct_num[key] / value * 100
        print(fmt.format(key, data_accuracy, value, data_correct_num[key]))
    print('--' * 20)

    den = correct + incorrect + error
    print('Correct num, incorrect num, error num: ', correct, incorrect, error)
    print('Correct, Incorrect, Error: ', (correct * 100 / den), (incorrect * 100 / den),
          (error * 100 / den))
    return (correct * 100 / den), (incorrect * 100 / den), (error * 100 / den), lenHuman, lenModel


def test_policy_graph_dataset_mask(
        config,
        dataset,
        model,
        num_objects=0,
        TQDM=True,
        TEST_GT=False,
        STATE_FORMAT_GOAL=False,
        INIT_DATAPOINT=True
):
    model.eval()
    assert 'action' in config.training
    e = config.embeddings

    correct, incorrect, error = 0, 0, 0
    buckets = []
    lenHuman, lenModel = [], []
    for i in range(30): buckets.append([0, 0, 0])
    for i in range(8):
        lenHuman.append([])
        lenModel.append([])

    data_num = {}
    data_correct_num = {}

    data_container = tqdm(dataset, desc='Policy Testing', ncols=80) if TQDM else dataset
    for data_item in data_container:
        if STATE_FORMAT_GOAL:  # * GraphDataset_State
            graphSeq, goal2vec, goal_json, actionSeq, action2vec, world_name, start_node = data_item
        else:  # * GraphDataset
            graphSeq, goal2vec, goalObjects2vec, actionSeq, action2vec, goal_json, world_name, start_node = data_item

        # * Generate mask.
        start_state = graphSeq[0].ndata['feat']
        target_state = goal2vec.ndata['feat']
        goal_attn = target_state - start_state
        mask = torch.abs(goal_attn) > 0.1

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

        # * ------------------------------------------------------------
        if TEST_GT:
            # * Test ground truth.
            for action_gt in actionSeq:
                # * !!!
                res, g, err = approx.execAction(config, action_gt, e)
                predActionSeq.append(action_gt)

                # * If (res == False) and (err == '') and (len(actionSeq) < xxx): continue
                if res:
                    data_correct_num[str(len(actionSeq))] += 1
                    correct += 1
                    lenModel.append(len(predActionSeq))
                    break
                elif (err == '') and (len(predActionSeq) == len(actionSeq)):
                    incorrect += 1
                    break
                elif err != '':
                    error += 1
                    break
        else:
            # * Test model.
            y_pred_list = []
            while True:
                with torch.no_grad():
                    output = model(graphSeq_t[-1], goal2vec, mask)
                    y_pred_list.append(output)
                y_pred = y_pred_list[-1]
                action_pred = vec2action_grammatical(
                    config, y_pred, num_objects, len(config.possibleStates), config.idx2object
                )

                # * !!!
                res, g, err = approx.execAction(config, action_pred, e)
                predActionSeq.append(action_pred)
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

    # print('key_items:')
    # print(key_items)

    key_items.sort(key=lambda x: int(x))  # * Convert str to int.
    for key in key_items:
        value = data_num[key]
        data_accuracy = data_correct_num[key] / value * 100
        print(fmt.format(key, data_accuracy, value, data_correct_num[key]))
    print('--' * 20)

    den = correct + incorrect + error
    print('Correct num, incorrect num, error num: ', correct, incorrect, error)
    print('Correct, Incorrect, Error: ', (correct * 100 / den), (incorrect * 100 / den),
          (error * 100 / den))
    return (correct * 100 / den), (incorrect * 100 / den), (error * 100 / den), lenHuman, lenModel


def accuracy_score_feature_extractor(config, dataset, model, num_objects=0, TQDM=True):
    """GraphDataset_State: Use final state as goal vector."""
    model.eval()
    total_correct = 0
    denominator = 0
    data_container = tqdm(dataset, desc='Accuracy Score', ncols=80) if TQDM else dataset
    for (graphSeq, goal2vec, goal_json, actionSeq, action2vec, world_name,
         start_node) in data_container:
        with torch.no_grad():
            y_pred_list = []
            graphSeq.append(goal2vec)
            for i in range(len(graphSeq) - 1):
                output, _ = model(graphSeq[i], graphSeq[i + 1])
                y_pred_list.append(output)

        for i, y_pred in enumerate(y_pred_list):
            denominator += 1
            action_pred = vec2action_grammatical(
                config, y_pred, num_objects, len(config.possibleStates), config.idx2object
            )
            if (action_pred == actionSeq[i]):
                total_correct += 1

    return ((total_correct / denominator) * 100)
