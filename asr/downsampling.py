import torch
from torch import nn
from torch.nn import ModuleList

class ResidualDownsamplingBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride, kernel_size=4):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channel, out_channel, kernel_size, padding='same')
        self.bn = nn.BatchNorm1d(out_channel)  # 输入归一化
        self.relu = nn.ReLU()
        # stride是步长，用来下采样
        self.conv2 = nn.Conv1d(out_channel, out_channel, kernel_size, stride=stride)
        if in_channel != out_channel:
            self.projection = nn.Conv1d(in_channel, out_channel, kernel_size=1)
        else:
            self.projection = None

    def forward(self, x):
        residual = x
        y = self.conv1(x)
        y = self.bn(y)

        # 如果定义了投影层，就对输入x进行投影
        if self.projection:
            residual = self.projection(x)

        y = self.relu(y) + residual  # 现在维度匹配了
        y = self.conv2(y)
        return y


class DownsamplingModal(nn.Module):
    def __init__(self, out_channel, hidden_dim, strides=[6, 6, 8, 4, 2], initial_mean_pooling_kernel_size=2):
        super().__init__()
        self.layers = ModuleList()
        self.meaning_pool = nn.MaxPool1d(kernel_size=initial_mean_pooling_kernel_size)  # 下采样，没有学习权重
        for i in range(len(strides)):
            self.layers.append(
                ResidualDownsamplingBlock(hidden_dim if i > 0 else 1, hidden_dim, strides[i], kernel_size=8))
        self.final_conv = nn.Conv1d(hidden_dim, out_channel, 4, padding='same')

    def forward(self, x):
        x = self.meaning_pool(x)
        for layer in self.layers:
            x = layer(x)
        x = self.final_conv(x)
        x = x.transpose(1, 2)  # batch dim seq_len  transformer的维度
        return x
