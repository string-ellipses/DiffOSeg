from abc import abstractmethod

import math
from copy import deepcopy 

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from ddpm.models.generic_UNet import ConvDropoutNormNonlin
from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from .attention import SpatialTransformer
import logging
LOGGER = logging.getLogger(__name__)


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride=stride)


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act, no_use_ca=False):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        if not no_use_ca:
            self.CA = CALayer(n_feat, reduction, bias=bias)
        else:
            self.CA = nn.Identity()
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res

class StackedFusionConvLayers(nn.Module):
    def __init__(self, input_feature_channels, bottleneck_feature_channel, output_feature_channels, num_convs,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, first_stride=None, basic_block=ConvDropoutNormNonlin):
        '''
        stacks ConvDropoutNormLReLU layers. initial_stride will only be applied to first layer in the stack. The other parameters affect all layers
        :param input_feature_channels:
        :param output_feature_channels:
        :param num_convs:
        :param dilation:
        :param kernel_size:
        :param padding:
        :param dropout:
        :param initial_stride:
        :param conv_op:
        :param norm_op:
        :param dropout_op:
        :param inplace:
        :param neg_slope:
        :param norm_affine:
        :param conv_bias:
        '''
        self.input_channels = input_feature_channels
        self.output_channels = output_feature_channels

        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        if first_stride is not None:
            self.conv_kwargs_first_conv = deepcopy(conv_kwargs)
            self.conv_kwargs_first_conv['stride'] = first_stride
        else:
            self.conv_kwargs_first_conv = conv_kwargs

        super(StackedFusionConvLayers, self).__init__()
        self.blocks = nn.Sequential(
            *([basic_block(input_feature_channels, bottleneck_feature_channel, self.conv_op,
                           self.conv_kwargs_first_conv,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs)] +
              [basic_block(bottleneck_feature_channel, bottleneck_feature_channel, self.conv_op,
                           self.conv_kwargs,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs) for _ in range(num_convs - 2)] +
              [basic_block(bottleneck_feature_channel, output_feature_channels, self.conv_op,
                           self.conv_kwargs_first_conv,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs)]
              ))

    def forward(self, x):
        return self.blocks(x)
    

class PromptBlock(nn.Module):
    def __init__(self, prompt_dim=128, prompt_len=5, prompt_size=96, lin_dim=192, middle_dim=32, learnable_input_prompt = False):
        super(PromptBlock, self).__init__()
        self.intermedia_prompt = nn.Parameter(th.rand(
            1, prompt_len, prompt_dim, prompt_size, prompt_size), requires_grad=learnable_input_prompt)
        self.linear_layer = nn.Linear(lin_dim, prompt_len)
        self.dec_conv3x3 = nn.Conv2d(
            prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)

        # # fuse
        # self.fuse = nn.Sequential(*[CAB((lin_dim+4+prompt_dim), kernel_size=3, reduction=4, bias=False, act=nn.PReLU(), no_use_ca=False) for _ in range(3)])
        # self.reduce = nn.Conv2d((lin_dim+4+prompt_dim), lin_dim, kernel_size=1, bias=False)

        # self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        #                         nn.Conv2d(lin_dim, lin_dim, 1, stride=1, padding=0, bias=False))

        # self.ca = nn.Sequential(nn.Conv2d(lin_dim, lin_dim, 1, stride=1, padding=0, bias=False),
        #                         CAB(lin_dim, kernel_size=3, reduction=4, bias=False, act=nn.PReLU(), no_use_ca=False))
        self.ca = nn.Sequential(
            *[CAB((lin_dim+1), kernel_size=3, reduction=4, bias=False, act=nn.PReLU(), no_use_ca=False) for _ in range(3)],
            nn.Conv2d((lin_dim+1), lin_dim, 1, stride=1, padding=0, bias=False)
        )
        #self.ca = nn.Conv2d((lin_dim+1), lin_dim, 1, stride=1, padding=0, bias=False)
        self.fusion_layer = StackedFusionConvLayers((lin_dim + prompt_dim), middle_dim, output_feature_channels=4, num_convs=3)
        
    def forward(self, x, reviewer_id, timesteps):
        # x = th.cat([x, reviewer_id.unsqueeze(2).unsqueeze(3).expand(-1, -1, x.size(2), x.size(3))], dim=1)
        B, C, H, W = x.shape
        emb = x.mean(dim=(-2, -1))
        prompt_weights = F.softmax(self.linear_layer(emb), dim=1)
        prompt_param = self.intermedia_prompt.unsqueeze(0).repeat(B, 1, 1, 1, 1, 1).squeeze(1)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * prompt_param
        prompt = th.sum(prompt, dim=1)

        prompt = F.interpolate(prompt, (H, W), mode="bilinear")
        now_prompt = self.dec_conv3x3(prompt) # [100, 128, 8, 8]

        # # 融合
        # x = th.cat([x, prompt], dim=1)
        # x = self.fuse(x)
        # x = self.reduce(x)

        # # x = self.up(x)
        # x = self.ca(x)linl
        # now_prompt = self.intermedia_prompt.repeat(B,1,1,1,1)
        # th.save(now_prompt.cpu(), "now_prompts.pt")
        dynamic_prompt = self.fusion_layer(th.cat([x, now_prompt], dim=1)) # [100, 4, 8, 8]
        
        
        sample_indices = th.arange(x.size(0)).to(x.device)
        
        task_prompt = dynamic_prompt[sample_indices, reviewer_id, :, :].unsqueeze(1) # [100, 1, 8, 8]

        # # 为了t-SNE可视化而准备
        # target_values = th.tensor([1, 125], device=timesteps.device)
        # if th.any(th.isin(timesteps, target_values)):
        #     th.save(task_prompt.cpu(), f"task_prompt_{reviewer_id[0].item()}_{timesteps[0].item()}_no_ca.pt")

        x = th.cat([x, task_prompt], dim=1)

        x = self.ca(x)

        return x

