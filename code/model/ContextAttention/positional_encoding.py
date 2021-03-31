import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_shape):
        # if large input is (b, dm1, dm2, ..., dmn)
        # max_shape should (dm1, dm2, ..., dmn)
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # add batch dim and channel dim
        self.register_buffer('signal',
                             self.position_encoding_nd(torch.zeros([1, *max_shape, d_model], dtype=torch.float)))
        # self.signal =
        self.linear = nn.Linear(2 * d_model, d_model)

    def forward(self, x):
        x = torch.cat((x, self.get_signal_like(x)), dim=-1)
        return self.linear(x)

    def get_signal_like(self, x):
        shape = [1, *x.shape[1:]]
        repeat = [x.shape[0]] + [1 for _ in x.shape[1:]]
        index = [slice(0, int(s)) for s in shape]
        signal: torch.Tensor = self.signal[index]
        return signal.repeat(repeat)

    def position_encoding_nd(self, x: torch.Tensor, min_timescale=1.0, max_timescale=1.0e4):
        """
        此处从tensor2tensor源码修改而来
        """
        input_shape = x.shape
        num_dims = len(input_shape) - 2
        channels = input_shape[-1]
        # 计算每一个位置维度在channel维度上能占用多长, 注意此处计算的值是可占用长度的一半
        num_timescales = int(channels // (num_dims * 2))
        # 计算三角函数中的角速度参量
        log_timescale_increment = (
                math.log(float(max_timescale) / float(min_timescale)) /
                (float(num_timescales) - 1))
        # 生成要用于三角函数的时间序列 (1, num_timescales)
        inv_timescales = min_timescale * torch.exp(
            torch.arange(0, num_timescales, dtype=torch.float, device=x.device) * -log_timescale_increment
        ).unsqueeze(0)
        output = x.clone()
        for dim in range(num_dims):
            # 计算当前待编码维度的长度, 生成要用于三角函数的位置序列 (len, 1)
            length = input_shape[dim + 1]
            position = torch.arange(0, length, dtype=torch.float, device=x.device)
            # 计算得到当前维度的角度编码块，之后用于三角函数 (len, num_timescales)
            scaled_time = position.unsqueeze(1).mm(inv_timescales)

            # 在position矩阵前需要补齐的通道数
            prepad_channel_nam = dim * 2 * num_timescales
            prepad = torch.zeros((scaled_time.shape[0], prepad_channel_nam), device=x.device)
            # 在position矩阵后需要补齐的通道数
            postpad_channel_nam = channels - (dim + 1) * 2 * num_timescales
            postpad = torch.zeros((scaled_time.shape[0], postpad_channel_nam), device=x.device)
            # 补齐至与输入相同的通道数,
            # (len, prepad_channel_nam + num_timescales + num_timescales + postpad_channel_nam) = (len, dmodel)
            # 注意：prepad_channel_nam + num_timescales + num_timescales + postpad_channel_nam = dmodel
            signal = torch.cat([prepad, torch.sin(scaled_time), torch.cos(scaled_time), postpad], dim=1)
            # 补齐维度 (1, 1, ..., len, dmodel)
            for _ in range(1 + dim):
                signal = signal.unsqueeze(0)
            # 补齐维度 (1, 1, ..., len, 1, 1, ..., dmodel)
            for _ in range(num_dims - 1 - dim):
                signal = signal.unsqueeze(-2)
            # 与x相加, 长度为1的维度会被自动广播
            output = output + signal
        return output
