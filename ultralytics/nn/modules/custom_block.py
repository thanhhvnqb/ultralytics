# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Block modules
"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, DWConv

__all__ = ('MBConv', 'C2mb', 'MBConv4', 'C2mb4', 'MBConvS', 'C2mbS', 'RepDWConv', 'MBRepConv',
           'C2mbrep', 'MB4RepConv', 'C2mb4rep', 'CFDWconv', 'ICFDWConv', 'C2ICFDW')


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


class RepDWConv(nn.Module):
    """RepDWConv is a rep-style block, including training and deploy status
    This code is based on RepConv (https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py)
    """
    default_act = nn.SiLU()  # default activation

    def __init__(self, c, k=(3, 5), s=1, g=1, act=True, bn=False, deploy=False):
        super().__init__()
        self.g = g
        self.k = k
        self.c = c
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c) if bn and s == 1 else None
        self.conv = nn.ModuleList(DWConv(c, c, kn, s, act=False) for kn in k)

    def forward_fuse(self, x):
        """Forward process"""
        return self.act(self.dwconv(x))

    def forward(self, x):
        """Forward process"""
        id_out = 0 if self.bn is None else self.bn(x)
        sum = None
        for conv in self.conv:
            sum = conv(x) if sum is None else sum + conv(x)
        return self.act(sum + id_out)

    def get_equivalent_kernel_bias(self):
        kernel, bias = None, None
        for dwconv in self.conv:
            if kernel is None:
                kernel, bias = self._fuse_bn_tensor(dwconv)
                kernel = self._pad_to_max_size_tensor(kernel, max(self.k))
            else:
                kernel3x3, bias3x3 = self._fuse_bn_tensor(dwconv)
                kernel += self._pad_to_max_size_tensor(kernel3x3, max(self.k))
                bias += bias3x3
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel + kernelid, bias + biasid

    def _pad_to_max_size_tensor(self, kernel1x1, max_size):
        if kernel1x1 is None:
            return 0
        elif kernel1x1.shape[2] == max_size:
            return kernel1x1
        else:
            return torch.nn.functional.pad(kernel1x1, [(max_size - kernel1x1.shape[2]) // 2, (max_size - kernel1x1.shape[2]) // 2,
                                                       (max_size - kernel1x1.shape[3]) // 2, (max_size - kernel1x1.shape[3]) // 2])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, 'id_tensor'):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        if hasattr(self, 'dwconv'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.dwconv = nn.Conv2d(in_channels=self.conv[-1].conv.in_channels,
                                out_channels=self.conv[-1].conv.out_channels,
                                kernel_size=self.conv[-1].conv.kernel_size,
                                stride=self.conv[-1].conv.stride,
                                padding=self.conv[-1].conv.padding,
                                dilation=self.conv[-1].conv.dilation,
                                groups=self.conv[-1].conv.groups,
                                bias=True).requires_grad_(False)
        self.dwconv.weight.data = kernel
        self.dwconv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv")
        if hasattr(self, 'nm'):
            self.__delattr__('nm')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')


class MBRepConv(nn.Module):
    """Mobile Inverted Convolution block."""

    def __init__(self, c1, c2, shortcut=True, s=1, k=(3, 5), e=6.0):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k=1)
        self.dwconv = RepDWConv(c_, k=k, s=s)
        self.cv2 = Conv(c_, c2, k=1)
        self.add = shortcut and c1 == c2 and s == 1

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.cv2(self.dwconv(self.cv1(x))) if self.add else self.cv2(self.dwconv(self.cv1(x)))


class C2mbrep(nn.Module):
    """CSP Bottleneck with MBConvRep block."""

    def __init__(self, c1, c2, n=1, shortcut=False, kmin=3, kmax=5, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        k = tuple([kt for kt in range(kmin, kmax + 1, 2)])
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(MBRepConv(self.c, self.c, shortcut, k=k) for _ in range(n))

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


class MB4RepConv(nn.Module):
    """Mobile Inverted Convolution block."""

    def __init__(self, c1, c2, shortcut=True, s=1, k=(3, 5), e=4.0):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k=1)
        self.dwconv = RepDWConv(c_, k=k, s=s)
        self.cv2 = Conv(c_, c2, k=1)
        self.add = shortcut and c1 == c2 and s == 1

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.cv2(self.dwconv(self.cv1(x))) if self.add else self.cv2(self.dwconv(self.cv1(x)))


class C2mb4rep(nn.Module):
    """CSP Bottleneck with MBConvRep block."""

    def __init__(self, c1, c2, n=1, shortcut=False, kmin=3, kmax=5, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        k = tuple([kt for kt in range(kmin, kmax + 1, 2)])
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(MB4RepConv(self.c, self.c, shortcut, k=k) for _ in range(n))

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


class ChannelCoords(nn.Module):
    """This code is based on CoordConv (https://github.com/walsvid/CoordConv/blob/master/coordconv.py)
    """
    __constants__ = ['coord_channels',]

    def __init__(self, rank, max_batch_size=128, stride=4, with_r=False, max_size=640):
        super().__init__()
        max_size = max_size // stride
        self.coord_channels = self.compute_coord_channels(rank, with_r, max_batch_size, max_size)
        if rank == 1:
            self.forward = self.forward_rank1
        elif rank == 2:
            self.forward = self.forward_rank2
        elif rank == 3:
            self.forward = self.forward_rank3
        else:
            raise NotImplementedError

    def forward_rank1(self, input_tensor):
        """
        :param input_tensor: shape (N, C_in, H, W)
        :return:
        """
        batch_size_shape, _, dim_x = input_tensor.shape
        return torch.cat([input_tensor, self.coord_channels[:batch_size_shape, :, :dim_x]], dim=1)

    def forward_rank2(self, input_tensor):
        """
        :param input_tensor: shape (N, C_in, H, W)
        :return:
        """
        batch_size_shape, _, dim_y, dim_x = input_tensor.shape
        return torch.cat([input_tensor, self.coord_channels[:batch_size_shape, :, :dim_y, :dim_x]], dim=1)

    def forward_rank3(self, input_tensor):
        """
        :param input_tensor: shape (N, C_in, H, W)
        :return:
        """
        batch_size_shape, _, dim_z, dim_y, dim_x = input_tensor.shape
        return torch.cat([input_tensor, self.coord_channels[:batch_size_shape, :, dim_z, dim_y, dim_x]], dim=1)

    def compute_coord_channels(self, rank, with_r, max_batch_size, max_size):
        if rank == 1:
            dim_x = max_size
            xx_range = torch.arange(dim_x, dtype=torch.int32)
            xx_channel = xx_range[None, None, :]

            xx_channel = xx_channel.float() / (dim_x - 1)
            xx_channel = xx_channel * 2 - 1

            if with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2))
                return torch.cat([xx_channel, rr], dim=1)
            else:
                return xx_channel
        elif rank == 2:
            dim_y, dim_x = max_size, max_size
            xx_ones = torch.ones([1, 1, 1, dim_x], dtype=torch.int32)
            yy_ones = torch.ones([1, 1, 1, dim_y], dtype=torch.int32)

            xx_range = torch.arange(dim_y, dtype=torch.int32)
            yy_range = torch.arange(dim_x, dtype=torch.int32)
            xx_range = xx_range[None, None, :, None]
            yy_range = yy_range[None, None, :, None]

            xx_channel = torch.matmul(xx_range, xx_ones)
            yy_channel = torch.matmul(yy_range, yy_ones)

            # transpose y
            yy_channel = yy_channel.permute(0, 1, 3, 2)

            xx_channel = xx_channel.float() / (dim_y - 1)
            yy_channel = yy_channel.float() / (dim_x - 1)

            xx_channel = xx_channel * 2 - 1
            yy_channel = yy_channel * 2 - 1
            xx_channel = xx_channel.repeat(max_batch_size, 1, 1, 1)
            yy_channel = yy_channel.repeat(max_batch_size, 1, 1, 1)

            if with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
                return torch.cat([xx_channel, yy_channel, rr], dim=1)
            else:
                return torch.cat([xx_channel, yy_channel], dim=1)

        elif rank == 3:
            dim_z, dim_y, dim_x = max_size, max_size, max_size
            xx_ones = torch.ones([1, 1, 1, 1, dim_x], dtype=torch.int32)
            yy_ones = torch.ones([1, 1, 1, 1, dim_y], dtype=torch.int32)
            zz_ones = torch.ones([1, 1, 1, 1, dim_z], dtype=torch.int32)

            xy_range = torch.arange(dim_y, dtype=torch.int32)
            xy_range = xy_range[None, None, None, :, None]

            yz_range = torch.arange(dim_z, dtype=torch.int32)
            yz_range = yz_range[None, None, None, :, None]

            zx_range = torch.arange(dim_x, dtype=torch.int32)
            zx_range = zx_range[None, None, None, :, None]

            xy_channel = torch.matmul(xy_range, xx_ones)
            xx_channel = torch.cat([xy_channel + i for i in range(dim_z)], dim=2)

            yz_channel = torch.matmul(yz_range, yy_ones)
            yz_channel = yz_channel.permute(0, 1, 3, 4, 2)
            yy_channel = torch.cat([yz_channel + i for i in range(dim_x)], dim=4)

            zx_channel = torch.matmul(zx_range, zz_ones)
            zx_channel = zx_channel.permute(0, 1, 4, 2, 3)
            zz_channel = torch.cat([zx_channel + i for i in range(dim_y)], dim=3)

            if with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) +
                                torch.pow(yy_channel - 0.5, 2) +
                                torch.pow(zz_channel - 0.5, 2))
                return torch.cat([xx_channel, yy_channel, zz_channel, rr], dim=1)
            else:
                return torch.cat([xx_channel, yy_channel, zz_channel], dim=1)
        else:
            raise NotImplementedError

    def _apply(self, fn):
        super()._apply(fn)
        self.coord_channels = fn(self.coord_channels)
        return self


class CFDWconv(nn.Module):
    """CFDWconv is a rep-style block, including training and deploy status
    This code is based on RepConv (https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py)
    """
    default_act = nn.SiLU()  # default activation

    def __init__(self, c, k=(3, 5), s=1, g=1, act=True, bn=False, deploy=False, stride=4, with_r=False, rank=2):
        super().__init__()
        self.g = g
        self.k = k
        self.c = c
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.channel_coords = ChannelCoords(stride=stride, rank=rank, with_r=False)
        self.bn = nn.BatchNorm2d(num_features=c + rank + int(with_r)) if bn and s == 1 else None
        self.conv = nn.ModuleList(DWConv(c + rank + int(with_r), c + rank + int(with_r), kn, s, act=False) for kn in k)

    def forward_fuse(self, x):
        """Forward process"""
        return self.act(self.dwconv(self.channel_coords(x)))

    def forward(self, x):
        """Forward process"""
        x = self.channel_coords(x)
        id_out = 0 if self.bn is None else self.bn(x)
        sum = None
        for conv in self.conv:
            sum = conv(x) if sum is None else sum + conv(x)
        return self.act(sum + id_out)

    def get_equivalent_kernel_bias(self):
        kernel, bias = None, None
        for dwconv in self.conv:
            if kernel is None:
                kernel, bias = self._fuse_bn_tensor(dwconv)
                kernel = self._pad_to_max_size_tensor(kernel, max(self.k))
            else:
                kernel3x3, bias3x3 = self._fuse_bn_tensor(dwconv)
                kernel += self._pad_to_max_size_tensor(kernel3x3, max(self.k))
                bias += bias3x3
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel + kernelid, bias + biasid

    def _pad_to_max_size_tensor(self, kernel1x1, max_size):
        if kernel1x1 is None:
            return 0
        elif kernel1x1.shape[2] == max_size:
            return kernel1x1
        else:
            return torch.nn.functional.pad(kernel1x1, [(max_size - kernel1x1.shape[2]) // 2, (max_size - kernel1x1.shape[2]) // 2,
                                                       (max_size - kernel1x1.shape[3]) // 2, (max_size - kernel1x1.shape[3]) // 2])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, 'id_tensor'):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        if hasattr(self, 'dwconv'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.dwconv = nn.Conv2d(in_channels=self.conv[-1].conv.in_channels,
                                out_channels=self.conv[-1].conv.out_channels,
                                kernel_size=self.conv[-1].conv.kernel_size,
                                stride=self.conv[-1].conv.stride,
                                padding=self.conv[-1].conv.padding,
                                dilation=self.conv[-1].conv.dilation,
                                groups=self.conv[-1].conv.groups,
                                bias=True).requires_grad_(False)
        self.dwconv.weight.data = kernel
        self.dwconv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv")
        if hasattr(self, 'nm'):
            self.__delattr__('nm')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')


class ICFDWConv(nn.Module):
    """Inverted Coordinate Fusion Depthwise Convolution block."""

    # ch_in, ch_out, shortcut, groups, kernels, expand
    def __init__(self, c1, c2, shortcut=True, s=1, k=(3, 5), e=6.0, stride=4, with_r=False, rank=2):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k=1)
        self.cfdwconv = CFDWconv(c_, k=k, s=s, stride=stride, with_r=with_r, rank=rank)
        self.cv2 = Conv(c_ + rank + int(with_r), c2, k=1)
        self.add = shortcut and c1 == c2 and s == 1

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.cv2(self.cfdwconv(self.cv1(x))) if self.add else self.cv2(self.cfdwconv(self.cv1(x)))


class C2ICFDW(nn.Module):
    """CSP Bottleneck with ICFDWConv block."""

    # ch_in, ch_out, number, shortcut, groups, expansion
    def __init__(self, c1, c2, n=1, shortcut=False, kmin=3, kmax=5, stride=4, e=0.5, with_r=False, rank=2):
        super().__init__()
        k = tuple([kt for kt in range(kmin, kmax + 1, 2)])
        rank = 2
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((rank + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(ICFDWConv(self.c, self.c, shortcut, k=k, stride=stride,
                               with_r=with_r, rank=rank) for _ in range(n))

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
