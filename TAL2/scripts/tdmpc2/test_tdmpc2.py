"""
@Project     ：TAL_2024
@File        ：test_tdmpc2.py
@Author      ：Xianqi-Zhang
@Date        ：2024/11/13
@Last        : 2024/11/13
@Description : 
"""
import os
import torch
import hydra
from termcolor import cprint

from src.envs import approx
from src.utils.misc import setup_seed
from src.config.config import init_args
from src.envs.CONSTANTS import EnvironmentConfig
from src.utils.buffer import ReplayBuffer
from src.baselines.tdmpc2.tdmpc2 import TDMPC2
from src.baselines.tdmpc2.common.parser import parse_cfg
from src.tal.action_proposal_network import vec2action_grammatical

global agent_cfg


@hydra.main(version_base=None, config_name='tdmpc2_config.yaml', config_path='.')
def get_agent_cfg(cfg: dict):
    cfg = parse_cfg(cfg)
    # cfg.multitask = True
    # cfg.tasks = 512
    cfg.obs_shape = {'state': (1024,)}  # * obs(512) + task(512)
    cfg.latent_dim = 512
    cfg.action_dim = 111
    cfg.task_dim = 1024
    cfg.episode_length = 20
    cfg.seed_steps = 1000
    cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)
    cfg.action_name_dim = 11
    cfg.action_obj_dim = 36
    cfg.action_state_dim = 28
    global agent_cfg
    agent_cfg = cfg


def test_policy_tdmpc2(config, dataset, policy, INIT_DATAPOINT=True):
    policy.eval()
    correct, incorrect, error = 0, 0, 0
    data_num = {}
    data_correct_num = {}
    for i in range(len(dataset)):
        world_name = dataset.node_sequences[i]['world_name']
        node_seq = dataset.node_sequences[i]['nodes']
        goal_json = dataset.goal_jsons[i]
        start_node = dataset.graphs[world_name].nodes[node_seq[0]]['state']
        obs, action, reward, task = dataset.sample(i)

        # * Store action length to dict.
        if str(len(action)) in data_num:
            data_num[str(len(action))] += 1
        else:
            data_num[str(len(action))] = 1
        if str(len(action)) not in data_correct_num:
            data_correct_num[str(len(action))] = 0

        world_num = int(world_name[-1])
        predActionSeq = []  # * Initialize
        graphSeq_t = []

        # * Initialize environment.
        if INIT_DATAPOINT:
            approx.initPolicy(config, config.domain, goal_json, world_num, SET_GAOL_JSON=True,
                              INPUT_DATAPOINT=start_node)
            graphSeq_t.append(obs[0])
        else:
            # * Use wold_home1 to test.
            # world_num = 1
            approx.initPolicy(config, config.domain, goal_json, world_num, SET_GAOL_JSON=True)
            init_g = approx.getInitializeDGLGraph(config)
            if config.device is not None:
                init_g = init_g.to(config.device)
            graphSeq_t.append(init_g)

        # * Test model.
        while True:
            step = 0
            with torch.no_grad():
                action_pred = policy(obs=graphSeq_t[-1], t0=(step == 0), eval_mode=False,
                                     task=task)
                action_hl = vec2action_grammatical(config, action_pred, config.num_objects,
                                                   len(config.possibleStates), config.idx2object)
            # * !!!
            res, g, err = approx.execAction(config, action_hl, config.embeddings)
            predActionSeq.append(action_hl)
            if g is not None and config.device is not None:
                g = g.to(config.device)
            graphSeq_t.append(g)

            # * If (res == False) and (err == '') and (len(actionSeq) < xxx): continue
            if res:
                data_correct_num[str(len(action))] += 1
                correct += 1
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
    return (correct * 100 / den), (incorrect * 100 / den), (error * 100 / den)


def main():
    rnd_seed = 0
    setup_seed(seed=rnd_seed)
    cprint('Set random seed = {}'.format(rnd_seed), 'green')

    args = init_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = EnvironmentConfig(args)

    global agent_cfg
    get_agent_cfg()
    agent = TDMPC2(agent_cfg, config=config)

    graphs_dir = 'data/1/home/'
    train_data_path = 'data/1/train_dataset.pkl'
    train_dataset = ReplayBuffer(config, graphs_dir, train_data_path)
    val_data_path = 'data/1/val_dataset.pkl'
    val_dataset = ReplayBuffer(config, graphs_dir, val_data_path)
    test_data_path = 'data/1/test_dataset.pkl'
    test_dataset = ReplayBuffer(config, graphs_dir, test_data_path)

    checkpoint_dir = 'checkpoint/tdmpc2/'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'tdmpcc2_epoch_{}.pth'.format(300))
    agent.load(checkpoint_path)

    print('\n\nTraining set...')
    test_policy_tdmpc2(config, train_dataset, agent, INIT_DATAPOINT=True)
    print('\n\nVal set...')
    test_policy_tdmpc2(config, val_dataset, agent, INIT_DATAPOINT=True)
    print('\n\nTest set...')
    test_policy_tdmpc2(config, test_dataset, agent, INIT_DATAPOINT=True)


if __name__ == '__main__':
    main()
