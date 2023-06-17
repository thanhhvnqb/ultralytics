# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Block modules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv
from .transformer import TransformerBlock

__all__ = ('MBConv', 'C2mb')


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
