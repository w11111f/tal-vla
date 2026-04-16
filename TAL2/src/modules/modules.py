import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from typing import Literal
from torchvision import models

from src.tal.layers import GatedHeteroRGCNLayer


class GraphFeatureExtractor(nn.Module):

    def __init__(
            self,
            config,
            n_objects=36,
            n_hidden=64,
            n_states=28,
            n_layers=3,
            output_dim=1024,
            activation=nn.LeakyReLU(),
            layer_type: Literal['linear', 'conv'] = 'linear'
    ):
        super().__init__()
        self.name = 'GraphFeatureExtractor'
        self.config = config
        self.n_hidden = n_hidden
        self.n_objects = n_objects
        self.n_states = n_states
        in_feats = config.features_dim  # * 338
        etypes = config.etypes  # * ['Close', 'Inside', 'On', 'Stuck']
        self.activation = activation
        self.layer_type = layer_type
        self.layers = nn.ModuleList()
        self.layers.append(
            GatedHeteroRGCNLayer(config, in_feats, n_hidden, etypes, activation=activation)
        )
        for i in range(n_layers - 1):
            self.layers.append(
                GatedHeteroRGCNLayer(config, n_hidden, n_hidden, etypes, activation=activation)
            )
        # * reset_parameters
        for layer in self.layers:
            layer.reset_parameters()
        if self.layer_type == 'linear':
            self.fc_1 = nn.Linear(in_feats, n_hidden)
            self.fc_2 = nn.Sequential(
                nn.Linear(n_objects * n_hidden * 2, output_dim),
                nn.LeakyReLU(),
            )
        elif self.layer_type == 'conv':
            self.fc_1 = nn.Conv1d(in_feats, n_hidden, kernel_size=1)
            self.fc_2 = nn.Sequential(
                nn.Conv1d(n_objects * n_hidden * 2, output_dim, kernel_size=1),
                nn.LeakyReLU(),
            )
        else:
            raise NotImplementedError

    def forward(self, g: DGLGraph):
        g_feat = g.ndata['feat']  # * (36, 338)
        for i, layer in enumerate(self.layers):
            g_feat = layer(g, g_feat)
        if self.layer_type == 'linear':
            feature = self.activation(self.fc_1(g.ndata['feat']))
            output = torch.cat([g_feat, feature], dim=1)  # * (36, 512)
            output = self.fc_2(output.reshape(1, -1))  # * (1, 512)
        elif self.layer_type == 'conv':
            feature = self.activation(self.fc_1(g.ndata['feat'].unsqueeze(0).permute(0, 2, 1)))
            feature = feature.permute(0, 2, 1).squeeze(0)
            output = torch.cat([g_feat, feature], dim=1)  # * (36, 512)
            output = self.fc_2(output.reshape(1, -1, 1))  # * (1, 512)
            output = output.reshape(1, -1)
        else:
            raise NotImplementedError
        return output


class SimNorm(nn.Module):
    """
    Simplicial normalization.
    Adapted from https://arxiv.org/abs/2204.00616.
    """

    def __init__(self, dim=8):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shp)

    def __repr__(self):
        return f"SimNorm(dim={self.dim})"


def MLP(input_dim, hidden_dim, output_dim, activation=None):
    if activation is None:
        activation = nn.ReLU()
    network = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        activation,
        nn.Linear(hidden_dim, hidden_dim),
        activation,
        nn.Linear(hidden_dim, output_dim),
        # activation
    )
    return network


class NormedLinear(nn.Linear):
    """
    Linear layer with LayerNorm, activation, and optionally dropout.
    https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/common/layers.py
    """

    def __init__(self, *args, act=nn.Mish(inplace=True), **kwargs):
        super().__init__(*args, **kwargs)
        self.ln = nn.LayerNorm(self.out_features)
        self.act = act

    def forward(self, x):
        x = super().forward(x)
        return self.act(self.ln(x))


class BasicBlock(nn.Module):
    """
    Basic block for ResNet.
    """

    def __init__(self, in_dim, out_dim, act=None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = act if act is not None else nn.Mish(inplace=True)
        self.shortcut_type = 'conv'  # * 'linear'
        # self.blocks = MLP(in_dim, out_dim * 2, out_dim)
        self.blocks = NormedLinear(in_dim, out_dim)

        if self.shortcut_type == 'linear':
            self.shortcut = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU()
            ) if self.should_apply_shortcut else nn.Identity()
        elif self.shortcut_type == 'conv':
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_dim, out_dim, kernel_size=1, stride=1, bias=False),
                nn.ReLU()
            ) if self.should_apply_shortcut else nn.Identity()
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.shortcut_type == 'linear':
            residual = self.shortcut(x)
            x = self.blocks(x)
            x += residual
        elif self.shortcut_type == 'conv':
            residual = self.shortcut(x.T)
            x = self.blocks(x)
            x += residual.T
        else:
            raise NotImplementedError

        return x

    @property
    def should_apply_shortcut(self):
        return self.in_dim != self.out_dim


class ResnetLayers(nn.Module):

    def __init__(self, in_dim, out_dim, act=nn.ReLU(), layer_num=2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layer_num = layer_num
        self.activation = act  # * nn.Mish(inplace=True)

        layers = [BasicBlock(in_dim, out_dim, act)]
        for _ in range(1, self.layer_num):
            layers.append(BasicBlock(out_dim, out_dim, act))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def make_layers(in_dim, mlp_dims, out_dim, act=None):
    if isinstance(mlp_dims, int):
        mlp_dims = [mlp_dims]
    dims = [in_dim] + mlp_dims + [out_dim]
    mlp = nn.ModuleList()
    for i in range(len(dims) - 2):
        mlp.append(ResnetLayers(dims[i], dims[i + 1]))
        mlp.append(ResnetLayers(dims[i + 1], dims[i + 1]))  # * Add.
    mlp.append(
        ResnetLayers(dims[-2], dims[-1], act=act)
        if act else nn.Linear(dims[-2], dims[-1])
    )
    return nn.Sequential(*mlp)


def load_resnet18(in_channels, out_channels):
    resnet18 = models.resnet18()
    resnet18.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    resnet18.fc = nn.Linear(resnet18.fc.in_features, out_channels)
    return resnet18
