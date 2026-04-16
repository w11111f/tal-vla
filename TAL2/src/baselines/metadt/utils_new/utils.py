"""
@Project     ：TAL_2024
@File        ：utils.py
@Author      ：Xianqi-Zhang
@Date        ：2024/12/2
@Last        : 2024/12/2
@Description : 
"""
import torch
from typing import Literal


def load_context_model(
        config,
        device,
        reward_type: Literal["regression", "classification"] = 'classification',
        train_context=True,
):
    if reward_type == 'regression':
        from src.baselines.metadt.utils_new.context_model import RNNContextEncoderReg as Encoder
        from src.baselines.metadt.utils_new.context_model import RewardDecoderReg as Decoder
    elif reward_type == 'classification':
        from src.baselines.metadt.utils_new.context_model import RNNContextEncoderCls as Encoder
        from src.baselines.metadt.utils_new.context_model import RewardDecoderCls as Decoder
    else:
        raise NotImplementedError
    context_encoder = Encoder(config, device).to(device)
    reward_decoder = Decoder(config, device).to(device)
    if train_context:
        context_encoder.train()
        reward_decoder.train()
    else:
        for name, param in context_encoder.named_parameters():
            param.requires_grad = False
        for name, param in reward_decoder.named_parameters():
            param.requires_grad = False
        context_encoder.eval()
        reward_decoder.eval()
    return context_encoder, reward_decoder


def save_metadt_checkpoint(metadt_model, checkpoint_path):
    torch.save(
        {
            'metadt': metadt_model.state_dict(),
        },
        checkpoint_path
    )


def load_metadt_checkpoint(metadt_model, checkpoint_path):
    metadt_model.load_state_dict(torch.load(checkpoint_path)['metadt'])
    print('[LOAD] metadt checkpoint from {}'.format(checkpoint_path))
    return metadt_model
