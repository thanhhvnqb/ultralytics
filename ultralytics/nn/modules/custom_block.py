# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Block modules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv
from .transformer import TransformerBlock

__all__ = ('MBConv', 'C2mb', 'MBConv4', 'C2mb4', 'MBConvS', 'C2mbS')


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class MBConv(nn.Module):
    """Mobile Inverted Convolution block."""

    def __init__(self, c1, c2, shortcut=True, s=1, k=5, e=6.0):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k=1)
        self.dwconv = DWConv(c_, c_, k=k, s=s)
        self.cv2 = Conv(c_, c2, k=1)
        self.add = shortcut and c1 == c2 and s == 1

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.cv2(self.dwconv(self.cv1(x))) if self.add else self.cv2(self.dwconv(self.cv1(x)))


class C2mb(nn.Module):
    """CSP Bottleneck with MBConv block."""

    def __init__(self, c1, c2, n=1, shortcut=False, k=5, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(MBConv(self.c, self.c, shortcut, k=k) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class MBConv4(nn.Module):
    """Mobile Inverted Convolution block with expand_factor=4."""

    def __init__(self, c1, c2, shortcut=True, s=1, k=5, e=4.0):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k=1)
        self.dwconv = DWConv(c_, c_, k=k, s=s)
        self.cv2 = Conv(c_, c2, k=1)
        self.add = shortcut and c1 == c2 and s == 1

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.cv2(self.dwconv(self.cv1(x))) if self.add else self.cv2(self.dwconv(self.cv1(x)))


class C2mb4(nn.Module):
    """CSP Bottleneck with MBConv block with expand_factor=4."""

    def __init__(self, c1, c2, n=1, shortcut=False, k=5, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(MBConv4(self.c, self.c, shortcut, k=k) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class MBConvS(nn.Module):
    """Mobile Inverted Convolution block with just 1 activation after last operation."""

    def __init__(self, c1, c2, shortcut=True, s=1, k=5, e=6.0):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k=1, act=None)
        self.dwconv = DWConv(c_, c_, k=k, s=s, act=None)
        self.cv2 = Conv(c_, c2, k=1, act=None)
        self.add = shortcut and c1 == c2 and s == 1
        self.act = nn.SiLU()

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return self.act(x + self.cv2(self.dwconv(self.cv1(x))) if self.add else self.cv2(self.dwconv(self.cv1(x))))


class C2mbS(nn.Module):
    """CSP Bottleneck with MBConv block."""

    def __init__(self, c1, c2, n=1, shortcut=False, k=5, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(MBConvS(self.c, self.c, shortcut, k=k) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
