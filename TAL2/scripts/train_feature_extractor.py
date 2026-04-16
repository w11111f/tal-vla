import os
import pickle
import re
import torch
import torch.nn as nn
from torch import optim
<<<<<<< ours
=======
from torch.utils.data import DataLoader
>>>>>>> theirs
from termcolor import cprint
import colorama
import warnings
from src.config.config import init_args
from src.utils.misc import setup_seed
from src.envs.CONSTANTS import EnvironmentConfig
from src.tal.utils_training import get_model, load_model, save_model
from src.tal.utils_training import accuracy_score_feature_extractor
from src.datasets.graph_dataset import GraphDataset_State

colorama.init()
warnings.filterwarnings('ignore')
<<<<<<< ours
=======
torch.backends.cudnn.benchmark = True
>>>>>>> theirs


def resolve_resume_checkpoint(config, model_name, seq_prefix=''):
    """Prefer an explicit stable checkpoint, otherwise resume from the latest epoch file."""
    model_dir = config.MODEL_SAVE_PATH
    stable_ckpt = os.path.join(model_dir, f'{seq_prefix}{model_name}_Trained.ckpt')
    if os.path.exists(stable_ckpt):
        return stable_ckpt

    pattern = re.compile(rf'^{re.escape(seq_prefix + model_name)}_(\d+)\.ckpt$')
    latest_epoch = -1
    latest_ckpt = None
    if not os.path.isdir(model_dir):
        return None

    for filename in os.listdir(model_dir):
        match = pattern.match(filename)
        if match is None:
            continue
        epoch = int(match.group(1))
        if epoch > latest_epoch:
            latest_epoch = epoch
            latest_ckpt = os.path.join(model_dir, filename)
    return latest_ckpt


<<<<<<< ours
def backprop(config, optimizer, dataset, model):
=======
def _move_sample_to_device(config, sample):
    graphSeq, goal2vec, goal_json, actionSeq, action2vec, world_name, start_node = sample
    if config.device is None:
        return sample
    graphSeq = [graph.to(config.device) for graph in graphSeq]
    goal2vec = goal2vec.to(config.device)
    action2vec = [action.to(config.device) for action in action2vec]
    return graphSeq, goal2vec, goal_json, actionSeq, action2vec, world_name, start_node


def _single_item_collate(batch):
    return batch[0]


def backprop(config, optimizer, dataset, model, accum_steps=1):
>>>>>>> theirs
    model.train()
    total_loss = 0.0
    action_loss = 0.0
    feature_loss = 0.0
    diff_action_loss = 0.0
    criterion_action = nn.BCELoss()
    criterion_diff_actions = nn.CosineEmbeddingLoss(margin=0.2)
    criterion_feature = nn.MSELoss()
<<<<<<< ours

    for iter_num, (graphSeq, goal2vec, _, actionSeq, action2vec, _, _) in enumerate(dataset):
=======
    optimizer.zero_grad(set_to_none=True)
    micro_step_count = 0

    for iter_num, sample in enumerate(dataset):
        graphSeq, goal2vec, _, actionSeq, action2vec, _, _ = _move_sample_to_device(
            config, sample
        )
>>>>>>> theirs

        graphSeq.append(goal2vec)
        data_len = len(graphSeq)
        if data_len < 3:  # * Action
            y_pred, _ = model(graphSeq[0], graphSeq[1])
            y_true = action2vec[0]

            loss = criterion_action(y_pred, y_true)
            total_loss += loss.item()
            action_loss += loss.item()

<<<<<<< ours
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
=======
            (loss / accum_steps).backward()
            micro_step_count += 1
            if micro_step_count % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
>>>>>>> theirs
        else:  # * Action and feature
            for i in range(data_len - 2):
                # * state_A --> Action[i] --> state_B --> Action[i+1] --> state_C
                state_A = graphSeq[i]
                state_B = graphSeq[i + 1]
                state_C = graphSeq[i + 2]

                y_true_AB = action2vec[i]
                y_true_BC = action2vec[i + 1]

                y_pred_AB, delta_feature_AB = model(state_A, state_B)
                y_pred_BC, delta_feature_BC = model(state_B, state_C)
                y_pred_AC, delta_feature_AC = model(state_A, state_C)

                loss_action_1 = criterion_action(y_pred_AB, y_true_AB)
                loss_action_2 = criterion_action(y_pred_BC, y_true_BC)
                loss_feature = criterion_feature(delta_feature_AC,
                                                 delta_feature_AB + delta_feature_BC)

                # * Different actions --> different features.
                if torch.equal(y_true_AB, y_true_BC):
                    y_cosine_embed = torch.tensor([1], device=config.device)
                else:
                    y_cosine_embed = torch.tensor([-1], device=config.device)
                loss_diff_actions = criterion_diff_actions(delta_feature_AB, delta_feature_BC,
                                                           y_cosine_embed)
                diff_action_loss += loss_diff_actions.item()

                loss = loss_action_1 + loss_action_2 + loss_feature + loss_diff_actions

                # loss = loss_action_1 + loss_action_2 + loss_feature
                # if not torch.equal(y_true_AB, y_true_BC):
                #     loss_diff_actions = 0.1 * criterion_diff_actions(delta_feature_AB, delta_feature_BC)
                #     diff_action_loss += loss_diff_actions.item()
                #     loss = loss + loss_diff_actions

                total_loss += loss.item()
                action_loss += (loss_action_1.item() + loss_action_2.item())
                feature_loss += loss_feature.item()

<<<<<<< ours
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
=======
                (loss / accum_steps).backward()
                micro_step_count += 1
                if micro_step_count % accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

    if micro_step_count % accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
>>>>>>> theirs

    return total_loss, action_loss, feature_loss, diff_action_loss


if __name__ == '__main__':
    rnd_seed = 42
    setup_seed(seed=rnd_seed)
    print('==' * 10)
    print('Set random seed = {}'.format(rnd_seed))
    print('==' * 10)

    args = init_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.model_name = 'AFE'
    args.num_epochs = 300
    config = EnvironmentConfig(args)

    # * ------------------------------------------------------------------------------------------
    # * Load data.
    graphs_dir = './data/home/'
    train_data_path = './data/train_dataset.pkl'
    train_dataset = GraphDataset_State(config, graphs_dir, train_data_path)

    val_data_path = './data/val_dataset.pkl'
    val_dataset = GraphDataset_State(config, graphs_dir, val_data_path)

    train_data_num = len(train_dataset)
    val_data_num = len(val_dataset)

    cprint('--' * 20, 'green')
    print('Train data num: {}'.format(train_data_num))
    print('Val data num: {}'.format(val_data_num))
    cprint('--' * 20, 'green')

<<<<<<< ours
=======
    num_workers = int(
        os.environ.get(
            'TAL_AFE_NUM_WORKERS',
            '0' if os.name == 'nt' else str(min(8, os.cpu_count() or 1))
        )
    )
    accum_steps = int(os.environ.get('TAL_AFE_ACCUM_STEPS', '8'))
    train_loader_kwargs = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': max(num_workers, 0),
        'collate_fn': _single_item_collate,
        'pin_memory': torch.cuda.is_available(),
    }
    if train_loader_kwargs['num_workers'] > 0:
        train_loader_kwargs['persistent_workers'] = True
        train_loader_kwargs['prefetch_factor'] = 2
    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    print('AFE train loader workers: {}'.format(train_loader_kwargs['num_workers']))
    print('AFE gradient accumulation steps: {}'.format(accum_steps))

>>>>>>> theirs
    # * ------------------------------------------------------------------------------------------
    model = get_model(config, config.model_name, config.features_dim, config.num_objects)
    seqTool = 'Seq_' if config.training == 'gcn_seq' else ''

    lr = 1e-4
    resume_ckpt = resolve_resume_checkpoint(config, model.name, seq_prefix=seqTool)
    if resume_ckpt is not None:
        print('Resuming AFE training from latest checkpoint: {}'.format(resume_ckpt))
    else:
        print('No existing AFE checkpoint found. Training will start from scratch.')
    model, optimizer, epoch, accuracy_list = load_model(
        config,
        seqTool + model.name + '_Trained',
        model,
        file_path=resume_ckpt,
        lr=lr
    )
    model = model.to(config.device)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

    test_frequency = 5
    if config.exec_type == 'train':
        print('Training ' + model.name + ' with ' + config.embedding)
        for epoch_num in range(epoch + 1, config.NUM_EPOCHS):
            scheduler.step(epoch=epoch_num)
            print('EPOCH {}, lr: {}'.format(epoch_num, optimizer.param_groups[0]['lr']))
<<<<<<< ours
            total_loss, action_loss, feature_loss, diff_action_loss = backprop(config, optimizer,
                                                                               train_dataset,
                                                                               model)
=======
            total_loss, action_loss, feature_loss, diff_action_loss = backprop(
                config, optimizer, train_loader, model, accum_steps=accum_steps
            )
>>>>>>> theirs
            # total_loss, action_loss, feature_loss = backprop_batch(config, optimizer, train_dataset, model)
            print('Total loss: {}'.format(total_loss))
            print('Action loss: {}'.format(action_loss))
            print('Feature loss: {}'.format(feature_loss))
            print('Diff action feature loss: {}'.format(diff_action_loss))

            # * ------------------------------------------------------------------------------------------
            # * Update learning rate.
            # if epoch_num in [100]:  # * 1e-5
            #     lr = lr / 10
            #     print('Change learning rate to {}'.format(lr))
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = lr

            # * ------------------------------------------------------------------------------------------
            t1, t2, t3 = 0, 0, 0
            if ((epoch_num > 50) and (total_loss < 50) and (epoch_num + 1) % test_frequency == 0):
                # * Test action predict accuracy.
                t1 = accuracy_score_feature_extractor(config, train_dataset, model,
                                                      config.num_objects, TQDM=False)
                print('Train Accuracy (action): {}'.format(t1))
                t2 = accuracy_score_feature_extractor(config, val_dataset, model,
                                                      config.num_objects, TQDM=False)
                print('Val Accuracy (action): {}'.format(t2))

            accuracy_list.append((t1, t2, t3, total_loss, action_loss, feature_loss))

            # * Save model
            file_path = save_model(config, model, optimizer, epoch_num, accuracy_list)

        train_acc = [i[0] for i in accuracy_list]
        val_acc = [i[1] for i in accuracy_list]
        print('The maximum acc on train set is ', str(max(train_acc)), ' at epoch ',
              train_acc.index(max(train_acc)))
        print('The maximum acc on val set is ', str(max(val_acc)), ' at epoch ',
              val_acc.index(max(val_acc)))

        results_save_path = './' + config.MODEL_SAVE_PATH + 'feature_extractor_results.pkl'
        with open(results_save_path, 'wb') as f:
            pickle.dump(accuracy_list, f)
