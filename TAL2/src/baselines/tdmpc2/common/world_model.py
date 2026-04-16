import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from dgl import DGLGraph
from src.baselines.tdmpc2.common import layers, math, init
from src.modules.modules import GraphFeatureExtractor


class WorldModel(nn.Module):
    """
    Modified for TAL dataset.
    Offline-RL training.

    TD-MPC2 implicit world model architecture.
    Can be used for both single-task and multi-task experiments.
    """

    def __init__(self, cfg, config=None):
        super().__init__()
        self.cfg = cfg
        self._graph_emb = GraphFeatureExtractor(config)
        self._task_emb = GraphFeatureExtractor(config)
        self._encoder = layers.enc(cfg)
        self._dynamics = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim,
                                    2 * [cfg.mlp_dim], cfg.latent_dim, act=layers.SimNorm(cfg))
        self._reward = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim,
                                  2 * [cfg.mlp_dim], max(cfg.num_bins, 1))

        # * Modified.
        self._pi_action_name = layers.mlp(cfg.latent_dim + cfg.task_dim, 2 * [cfg.mlp_dim],
                                          2 * cfg.action_name_dim)
        self._pi_object1 = layers.mlp(cfg.latent_dim + cfg.task_dim, 2 * [cfg.mlp_dim],
                                      2 * cfg.action_obj_dim)
        self._pi_object2 = layers.mlp(cfg.latent_dim + cfg.task_dim, 2 * [cfg.mlp_dim],
                                      2 * cfg.action_obj_dim)
        self._pi_state = layers.mlp(cfg.latent_dim + cfg.task_dim, 2 * [cfg.mlp_dim],
                                    2 * cfg.action_state_dim)

        self._Qs = layers.Ensemble([layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim,
                                               2 * [cfg.mlp_dim], max(cfg.num_bins, 1),
                                               dropout=cfg.dropout) for _ in range(cfg.num_q)])
        self.apply(init.weight_init)
        init.zero_([self._reward[-1].weight, self._Qs.params[-2]])
        self._target_Qs = deepcopy(self._Qs).requires_grad_(False)
        self.log_std_min = torch.tensor(cfg.log_std_min)
        self.log_std_dif = torch.tensor(cfg.log_std_max) - self.log_std_min

    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def to(self, *args, **kwargs):
        """
        Overriding `to` method to also move additional tensors to device.
        """
        super().to(*args, **kwargs)
        self.log_std_min = self.log_std_min.to(*args, **kwargs)
        self.log_std_dif = self.log_std_dif.to(*args, **kwargs)
        return self

    def train(self, mode=True):
        """
        Overriding `train` method to keep target Q-networks in eval mode.
        """
        super().train(mode)
        self._target_Qs.train(False)
        return self

    def track_q_grad(self, mode=True):
        """
        Enables/disables gradient tracking of Q-networks.
        Avoids unnecessary computation during policy optimization.
        This method also enables/disables gradients for task embeddings.
        """
        for p in self._Qs.parameters():
            p.requires_grad_(mode)
        for p in self._graph_emb.parameters():
            p.requires_grad_(mode)
        for p in self._task_emb.parameters():
            p.requires_grad_(mode)

    def soft_update_target_Q(self):
        """
        Soft-update target Q-networks using Polyak averaging.
        """
        with torch.no_grad():
            for p, p_target in zip(self._Qs.parameters(), self._target_Qs.parameters()):
                p_target.data.lerp_(p.data, self.cfg.tau)

    def _raw_emb(self, x):
        """Embed obs or task."""
        if isinstance(x, list):
            emb = torch.cat([self._graph_emb(item) for item in x])
        elif isinstance(x, DGLGraph):
            emb = self._graph_emb(x)
        elif isinstance(x, torch.Tensor):
            emb = x
        else:
            raise NotImplementedError
        return emb

    def task_emb(self, x, task):
        """
        Continuous task embedding for multi-task experiments.
        Retrieves the task embedding for a given task ID `task`
        and concatenates it to the input `x`.
        """
        o_emb = self._raw_emb(x)
        t_emb = self._raw_emb(task)
        if t_emb.shape[0] == 1:
            t_emb = t_emb.repeat(o_emb.shape[0], 1)
        embed = torch.cat([o_emb, t_emb], dim=-1)
        return embed

    def encode(self, obs, task):
        """
        Encodes an observation into its latent representation.
        This implementation assumes a single state-based observation.
        """
        obs = self.task_emb(obs, task)
        return self._encoder[self.cfg.obs](obs)

    def next(self, z, a, task):
        """
        Predicts the next latent state given the current latent state and action.
        """
        z = self.task_emb(z, task)
        if len(a.shape) == 1:
            a = a.unsqueeze(0)
        z = torch.cat([z, a], dim=-1)
        return self._dynamics(z)

    def reward(self, z, a, task):
        """
        Predicts instantaneous (single-step) reward.
        """
        z = self.task_emb(z, task)
        if isinstance(a, list):
            a = torch.stack(a)
        if len(a.shape) == 1:
            a = a.unsqueeze(0)
        z = torch.cat([z, a], dim=-1)
        return self._reward(z)

    # def pi(self, z, task):
    #     """
    #     Samples an action from the policy prior.
    #     The policy prior is a Gaussian distribution with
    #     mean and (log) std predicted by a neural network.
    #     """
    #     z = self.task_emb(z, task)
    #
    #     # Gaussian policy prior
    #     mu, log_std = self._pi(z).chunk(2, dim=-1)
    #     log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
    #     eps = torch.randn_like(mu)
    #     action_dims = None
    #     log_pi = math.gaussian_logprob(eps, log_std, size=action_dims)
    #     pi = mu + eps * log_std.exp()
    #     mu, pi, log_pi = math.squash(mu, pi, log_pi)
    #
    #     return mu, pi, log_pi, log_std

    def _pi_gaussian_prior(self, mu, log_std):
        # Gaussian policy prior
        log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
        eps = torch.randn_like(mu)
        action_dims = None
        log_pi = math.gaussian_logprob(eps, log_std, size=action_dims)
        pi = mu + eps * log_std.exp()
        mu, pi, log_pi = math.squash(mu, pi, log_pi)
        return mu, pi, log_pi, log_std

    def pi(self, z, task):
        """
        Samples an action from the policy prior.
        The policy prior is a Gaussian distribution with
        mean and (log) std predicted by a neural network.
        """
        z = self.task_emb(z, task)
        mu_name, log_std_name = self._pi_action_name(z).chunk(2, dim=-1)
        mu_obj1, log_std_obj1 = self._pi_object1(z).chunk(2, dim=-1)
        mu_obj2, log_std_obj2 = self._pi_object2(z).chunk(2, dim=-1)
        mu_state, log_std_state = self._pi_state(z).chunk(2, dim=-1)

        mu_name, pi_name, log_pi_name, log_std_name = self._pi_gaussian_prior(mu_name, log_std_name)
        mu_obj1, pi_obj1, log_pi_obj1, log_std_obj1 = self._pi_gaussian_prior(mu_obj1, log_std_obj1)
        mu_obj2, pi_obj2, log_pi_obj2, log_std_obj2 = self._pi_gaussian_prior(mu_obj2, log_std_obj2)
        mu_state, pi_state, log_pi_state, log_std_state = self._pi_gaussian_prior(mu_state, log_std_state)

        mu = torch.cat([mu_name, mu_obj1, mu_obj2, mu_state], dim=-1)
        pi = torch.cat([pi_name, pi_obj1, pi_obj2, pi_state], dim=-1)
        log_std = torch.cat([log_std_name, log_std_obj1, log_std_obj2, log_std_state], dim=-1)
        eps = torch.randn_like(mu)
        log_pi = math.gaussian_logprob(eps, log_std, size=None)
        return mu, pi, log_pi, log_std

    def Q(self, z, a, task, return_type='min', target=False):
        """
        Predict state-action value.
        `return_type` can be one of [`min`, `avg`, `all`]:
            - `min`: return the minimum of two randomly subsampled Q-values.
            - `avg`: return the average of two randomly subsampled Q-values.
            - `all`: return all Q-values.
        `target` specifies whether to use the target Q-networks or not.
        """
        assert return_type in {'min', 'avg', 'all'}
        z = self.task_emb(z, task)
        if isinstance(a, list):
            a = torch.stack(a)
        if len(a.shape) == 1:
            a = a.unsqueeze(0)
        z = torch.cat([z, a], dim=-1)
        out = (self._target_Qs if target else self._Qs)(z)
        if return_type == 'all':
            return out
        Q1, Q2 = out[np.random.choice(self.cfg.num_q, 2, replace=False)]
        Q1, Q2 = math.two_hot_inv(Q1, self.cfg), math.two_hot_inv(Q2, self.cfg)
        value = torch.min(Q1, Q2) if return_type == 'min' else (Q1 + Q2) / 2
        return value
