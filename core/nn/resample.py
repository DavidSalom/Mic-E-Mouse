"""
Upsample using linear interpolation
Reference : https://github.com/f90/Wave-U-Net-Pytorch/blob/master/model/resample.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, Linear, ConvTranspose1d, BatchNorm1d, ModuleList


class LinearUpsample(nn.Module):
    def __init__(self, channels : int,
                 kernel_size : int,
                 stride : int,
                 transpose : bool=False,
                 padding:str="reflect",
                 trainable:bool=False
                ):

        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.transpose = transpose
        self.channels = channels

        cutoff = 0.5 / stride

        assert(kernel_size > 2)
        assert ((kernel_size - 1) % 2 == 0)
        assert(padding == "reflect" or padding == "valid")

        filter = build_sinc_filter(kernel_size, cutoff)

        self.filter = torch.nn.Parameter(torch.from_numpy(torch.repeat(torch.reshape(filter, [1, 1, kernel_size]), channels, axis=0)), requires_grad=trainable)

    def forward(self, x):
        # Pad here if not using transposed conv
        input_size = x.shape[2]
        if self.padding != "valid":
            num_pad = (self.kernel_size-1)//2
            out = F.pad(x, (num_pad, num_pad), mode=self.padding)
        else:
            out = x

        # Lowpass filter (+ 0 insertion if transposed)
        if self.transpose:
            expected_steps = ((input_size - 1) * self.stride + 1)
            if self.padding == "valid":
                expected_steps = expected_steps - self.kernel_size + 1

            out = F.conv_transpose1d(out, self.filter, stride=self.stride, padding=0, groups=self.channels)
            diff_steps = out.shape[2] - expected_steps
            if diff_steps > 0:
                assert(diff_steps % 2 == 0)
                out = out[:,:,diff_steps//2:-diff_steps//2]
        else:
            assert(input_size % self.stride == 1)
            out = F.conv1d(out, self.filter, stride=self.stride, padding=0, groups=self.channels)

        return out






def build_sinc_filter(kernel_size, cutoff):
    # FOLLOWING https://www.analog.com/media/en/technical-documentation/dsp-book/dsp_book_Ch16.pdf
    # Sinc lowpass filter
    # Build sinc kernel
    assert(kernel_size % 2 == 1)
    M = kernel_size - 1
    filter = torch.zeros(kernel_size, dtype=torch.float32)
    for i in range(kernel_size):
        if i == M//2:
            filter[i] = 2 * torch.pi * cutoff
        else:
            filter[i] = (torch.sin(2 * torch.pi * cutoff * (i - M//2)) / (i - M//2)) * \
                    (0.42 - 0.5 * torch.cos((2 * torch.pi * i) / M) + 0.08 * torch.cos(4 * torch.pi * M))

    filter = filter / np.sum(filter)
    return filter