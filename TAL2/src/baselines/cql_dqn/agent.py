"""
Source Code: https://github.com/BY571/CQL
Modified
"""
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.nn.utils import clip_grad_norm_
from src.baselines.cql_dqn.utils import generate_model
from src.tal.action_proposal_network import vec2action_grammatical


class CQLAgent():

    def __init__(self, config, action_set):
        self.config = config
        self.action_set = action_set
        self.tau = 1e-3
        self.gamma = 0.99
        self.optimizer_lr = 1e-4
        self.criterion = nn.BCELoss()
        self.network, self.optimizer, self.epoch, accuracy_list = generate_model(config,
                                                                                 lr=self.optimizer_lr)
        self.target_net, _, _, _ = generate_model(config, lr=self.optimizer_lr, target_net=True)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)

    def get_action(self, state, goal_state, epsilon):
        if random.random() > epsilon:
            self.network.eval()
            with torch.no_grad():
                output = self.network(state, goal_state)
            self.network.train()
            action = vec2action_grammatical(self.config,
                                            output,
                                            self.config.num_objects,
                                            len(self.config.possibleStates),
                                            self.config.idx2object)
        else:
            action = random.choices(self.action_set, k=1)
        return action

    def soft_update(self, local_model=None, target_model=None, tau=None):
        if local_model is None:
            local_model = self.network
        if target_model is None:
            target_model = self.target_net
        if tau is None:
            tau = self.tau

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def learn(self, experiences, episode):
        # * Update learning rate.
        if episode in [400]:  # * 1e-5
            self.optimizer_lr = self.optimizer_lr / 10
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.optimizer_lr

        # states, actions, rewards, next_states, dones = experiences
        graphSeq, goal2vec, goal_json, actionSeq, action2vec, world_name, start_node, rewards, dones = experiences

        total_q1_loss = 0.0
        total_cql1_loss = 0.0
        total_bellman_error = 0.0

        for i in range(len(graphSeq)):
            # * Q-learning
            # * Q(s, a) := Q(s, a) + alpha * (reward + gamma * maxQ(s', a') - Q(s, a))
            # *          = (1- alpha) * Q(s, a) + alpha * (reward + gamma * maxQ(s', a'))
            if (random.random() < 0.7) or (i == (len(graphSeq) - 1)):
                if i == (len(graphSeq) - 1):  # dones
                    Q_targets = torch.tensor([rewards[i], rewards[i], rewards[i], rewards[i]],
                                             dtype=torch.float32, device=self.config.device)
                else:
                    with torch.no_grad():
                        # * action: [11] | pred1_output: [36] | pred2_object: [36] | pred2_state: [28]
                        Q_targets_next = self.target_net(graphSeq[i + 1], goal2vec).detach()
                        Q_targets_act_name = Q_targets_next[:11].max()[None]
                        Q_targets_obj1 = Q_targets_next[11:47].max()[None]
                        Q_targets_obj2 = Q_targets_next[47:83].max()[None]
                        Q_targets_state = Q_targets_next[83:111].max()[None]

                        r_i = rewards[i]
                        d = 1 - dones[i]
                        Q_targets_act_name = r_i + (self.gamma * Q_targets_act_name * d)
                        Q_targets_obj1 = r_i + (self.gamma * Q_targets_obj1 * d)
                        Q_targets_obj2 = r_i + (self.gamma * Q_targets_obj2 * d)
                        Q_targets_state = r_i + (self.gamma * Q_targets_state * d)
                        Q_targets = torch.cat(
                            [Q_targets_act_name, Q_targets_obj1, Q_targets_obj2, Q_targets_state],
                            dim=0)

                Q_a_s = self.network(graphSeq[i], goal2vec)
                action_name = torch.tensor(
                    [action2vec[i][:11].argmax()], device=self.config.device
                )
                action_obj1 = torch.tensor(
                    [action2vec[i][11:47].argmax()], device=self.config.device
                )
                action_obj2 = torch.tensor(
                    [action2vec[i][47:83].argmax()], device=self.config.device
                )
                action_state = torch.tensor(
                    [action2vec[i][83:111].argmax()], device=self.config.device
                )
                Q_expected_name = Q_a_s[:11].gather(0, action_name)[None]
                Q_expected_obj1 = Q_a_s[11:47].gather(0, action_obj1)[None]
                Q_expected_obj2 = Q_a_s[47:83].gather(0, action_obj2)[None]
                Q_expected_state = Q_a_s[83:111].gather(0, action_state)[None]
                Q_expected = torch.cat(
                    [Q_expected_name, Q_expected_obj1, Q_expected_obj2, Q_expected_state], dim=0)

            else:  # * Random select other action and reward = 0.
                with torch.no_grad():
                    # * action: [11] | pred1_output: [36] | pred2_object: [36] | pred2_state: [28]
                    idx_name = torch.tensor(random.randrange(0, 11), device=self.config.device)
                    idx_obj1 = torch.tensor(random.randrange(0, 36), device=self.config.device)
                    idx_obj2 = torch.tensor(random.randrange(0, 36), device=self.config.device)
                    idx_state = torch.tensor(random.randrange(0, 28), device=self.config.device)
                    if idx_name == action2vec[i][:11].argmax():
                        idx_name = (idx_name + 1) % 11
                    if idx_obj1 == action2vec[i][11:47].argmax():
                        idx_obj1 = (idx_obj1 + 1) % 36
                    if idx_obj2 == action2vec[i][47:83].argmax():
                        idx_obj2 = (idx_obj2 + 1) % 36
                    if idx_state == action2vec[i][83:111].argmax():
                        idx_state = (idx_state + 1) % 28

                    Q_targets_next = self.target_net(graphSeq[i + 1], goal2vec).detach()
                    Q_targets_act_name = Q_targets_next[idx_name][None]
                    Q_targets_obj1 = Q_targets_next[idx_obj1][None]
                    Q_targets_obj2 = Q_targets_next[idx_obj2][None]
                    Q_targets_state = Q_targets_next[idx_state][None]

                    Q_targets_act_name = self.gamma * Q_targets_act_name * (1 - dones[i])
                    Q_targets_obj1 = self.gamma * Q_targets_obj1 * (1 - dones[i])
                    Q_targets_obj2 = self.gamma * Q_targets_obj2 * (1 - dones[i])
                    Q_targets_state = self.gamma * Q_targets_state * (1 - dones[i])
                    Q_targets = torch.cat(
                        [Q_targets_act_name, Q_targets_obj1, Q_targets_obj2, Q_targets_state],
                        dim=0)

                Q_a_s = self.network(graphSeq[i], goal2vec)

                Q_expected_name = Q_a_s[:11].gather(0, idx_name)[None]
                Q_expected_obj1 = Q_a_s[11:47].gather(0, idx_obj1)[None]
                Q_expected_obj2 = Q_a_s[47:83].gather(0, idx_obj2)[None]
                Q_expected_state = Q_a_s[83:111].gather(0, idx_state)[None]
                Q_expected = torch.cat(
                    [Q_expected_name, Q_expected_obj1, Q_expected_obj2, Q_expected_state], dim=0)

            # cql1_loss = torch.logsumexp(Q_a_s, dim=0) - Q_a_s.mean()
            cql1_loss = torch.logsumexp(Q_expected_name, dim=0) - Q_expected_name.mean() + \
                        torch.logsumexp(Q_expected_obj1, dim=0) - Q_expected_obj1.mean() + \
                        torch.logsumexp(Q_expected_obj2, dim=0) - Q_expected_obj2.mean() + \
                        torch.logsumexp(Q_expected_state, dim=0) - Q_expected_state.mean()

            bellman_error = F.mse_loss(Q_expected, Q_targets)
            q1_loss = cql1_loss + 0.5 * bellman_error

            self.optimizer.zero_grad()
            q1_loss.backward()
            clip_grad_norm_(self.network.parameters(), 1)
            self.optimizer.step()

            total_q1_loss += q1_loss.detach().item()
            total_cql1_loss += cql1_loss.detach().item()
            total_bellman_error += bellman_error.detach().item()

            # ------------------- update target network ------------------- #
            # self.soft_update(self.network, self.target_net)

        return total_q1_loss, total_cql1_loss, total_bellman_error
