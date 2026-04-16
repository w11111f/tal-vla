import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.baselines.metadt.configs import args_ant_dir
from src.baselines.metadt.utils import setup_seed
from src.baselines.metadt.utils_model import save_context_checkpoint, create_env_AntDir, \
    load_context_dataset, load_model


def main():
    args = args_ant_dir.get_args()
    setup_seed(args.seed)
    np.set_printoptions(precision=4, suppress=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_dir = f'runs/{args.env_name}/context/{args.data_quality}/horizon{args.context_horizon}'
    writer = SummaryWriter(log_dir)

    # * Env and task.
    env, task_info = create_env_AntDir(args)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # * Dataset.
    train_dataset, test_dataset = load_context_dataset(args, device)
    train_dataloader = DataLoader(train_dataset, batch_size=args.context_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.context_batch_size, shuffle=True)

    # The models
    _, context_encoder, reward_decoder = load_model(args, state_dim, action_dim, device,
                                                    train_context=True)
    optimizer = torch.optim.Adam([*context_encoder.parameters(), *reward_decoder.parameters()],
                                 lr=args.context_lr)

    save_model_path = f'metadt_saves/{args.env_name}/context/{args.data_quality}/{args.seed}/horizon{args.context_horizon}'
    os.makedirs(save_model_path, exist_ok=True)

    global_step = 0
    best_loss = float('inf')
    for epoch in range(args.context_train_epochs):
        print(f'\n========== Epoch {epoch} ==========')

        # * Model training.
        context_encoder.train()
        reward_decoder.train()
        for step, (transition, segment, next_segment) in tqdm(enumerate(train_dataloader)):
            state, action, reward, next_state, _, _ = transition
            state_segment, action_segment, reward_segment = segment
            context = context_encoder(
                state_segment.transpose(0, 1),
                action_segment.transpose(0, 1),
                reward_segment.transpose(0, 1)
            )
            reward_predict = reward_decoder(state, action, next_state, context)
            loss = F.mse_loss(reward_predict, reward)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [*context_encoder.parameters(), *reward_decoder.parameters()], 1.0
            )
            optimizer.step()
            global_step += 1
            writer.add_scalar('loss/train', loss.item(), global_step)

        # * Model evaluation.
        with torch.no_grad():
            context_encoder.eval()
            reward_decoder.eval()
            transition, segment, next_segment = next(iter(test_dataloader))
            state, action, reward, next_state, _, _ = transition
            state_segment, action_segment, reward_segment = segment
            context = context_encoder(
                state_segment.transpose(0, 1),
                action_segment.transpose(0, 1),
                reward_segment.transpose(0, 1)
            )
            reward_predict = reward_decoder(state, action, next_state, context)
            loss = F.mse_loss(reward_predict, reward).detach().cpu().numpy()
            writer.add_scalar('loss/test', loss, epoch + 1)
            print(f'Model Evaluation, test loss: {loss}')

            if loss < best_loss:
                print('Save the best model...')
                best_loss = loss
                model_name = f'{save_model_path}/context_models_best.pt'
                save_context_checkpoint(context_encoder, reward_decoder, model_name)
            print(f'Predicted rewards: {reward_predict.detach().cpu().numpy()[:8].reshape(-1)}')
            print(f'Real rewards: {reward.detach().cpu().numpy()[:8].reshape(-1)}')

        if (epoch + 1) % args.save_context_model_every == 0:
            model_name = f'{save_model_path}/context_models_{epoch + 1}.pt'
            save_context_checkpoint(context_encoder, reward_decoder, model_name)


if __name__ == '__main__':
    main()
