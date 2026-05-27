from __future__ import annotations

import math
import warnings

import einops
import torch
import torch.nn as nn
from torch.nn import functional as F


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in trunc_normal_. The distribution may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


class LayerNormProxy(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = einops.rearrange(x, "b c h w -> b h w c")
        x = self.norm(x)
        return einops.rearrange(x, "b h w c -> b c h w")


class RountingFunction(nn.Module):
    def __init__(self, in_channels: int, kernel_number: int, dropout_rate: float = 0.2, proportion: float = 40.0):
        super().__init__()
        self.kernel_number = kernel_number
        self.dwc = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False
        )
        self.norm = LayerNormProxy(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc_alpha = nn.Linear(in_channels, kernel_number, bias=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc_theta = nn.Linear(in_channels, kernel_number, bias=False)
        self.act_func = nn.Softsign()
        self.proportion = proportion / 180.0 * math.pi

        trunc_normal_(self.dwc.weight, std=0.02)
        trunc_normal_(self.fc_alpha.weight, std=0.02)
        trunc_normal_(self.fc_theta.weight, std=0.02)

    def forward(self, x: torch.Tensor):
        x = self.dwc(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.avg_pool(x).squeeze(dim=-1).squeeze(dim=-1)
        alphas = torch.sigmoid(self.fc_alpha(self.dropout1(x)))
        angles = self.act_func(self.fc_theta(self.dropout2(x))) * self.proportion
        return alphas, angles


def _get_rotation_matrix(thetas: torch.Tensor) -> torch.Tensor:
    bs, groups = thetas.shape
    device = thetas.device
    thetas = thetas.reshape(-1)
    x = torch.cos(thetas).unsqueeze(0).unsqueeze(0)
    y = torch.sin(thetas).unsqueeze(0).unsqueeze(0)
    a = x - y
    b = x * y
    c = x + y

    rot_mat_positive = torch.cat((
        torch.cat((a, 1 - a, torch.zeros(1, 7, bs * groups, device=device)), dim=1),
        torch.cat((torch.zeros(1, 1, bs * groups, device=device), x - b, b, torch.zeros(1, 1, bs * groups, device=device), 1 - c + b, y - b, torch.zeros(1, 3, bs * groups, device=device)), dim=1),
        torch.cat((torch.zeros(1, 2, bs * groups, device=device), a, torch.zeros(1, 2, bs * groups, device=device), 1 - a, torch.zeros(1, 3, bs * groups, device=device)), dim=1),
        torch.cat((b, y - b, torch.zeros(1, 1, bs * groups, device=device), x - b, 1 - c + b, torch.zeros(1, 4, bs * groups, device=device)), dim=1),
        torch.cat((torch.zeros(1, 4, bs * groups, device=device), torch.ones(1, 1, bs * groups, device=device), torch.zeros(1, 4, bs * groups, device=device)), dim=1),
        torch.cat((torch.zeros(1, 4, bs * groups, device=device), 1 - c + b, x - b, torch.zeros(1, 1, bs * groups, device=device), y - b, b), dim=1),
        torch.cat((torch.zeros(1, 3, bs * groups, device=device), 1 - a, torch.zeros(1, 2, bs * groups, device=device), a, torch.zeros(1, 2, bs * groups, device=device)), dim=1),
        torch.cat((torch.zeros(1, 3, bs * groups, device=device), y - b, 1 - c + b, torch.zeros(1, 1, bs * groups, device=device), b, x - b, torch.zeros(1, 1, bs * groups, device=device)), dim=1),
        torch.cat((torch.zeros(1, 7, bs * groups, device=device), 1 - a, a), dim=1),
    ), dim=0)

    rot_mat_negative = torch.cat((
        torch.cat((c, torch.zeros(1, 2, bs * groups, device=device), 1 - c, torch.zeros(1, 5, bs * groups, device=device)), dim=1),
        torch.cat((-b, x + b, torch.zeros(1, 1, bs * groups, device=device), b - y, 1 - a - b, torch.zeros(1, 4, bs * groups, device=device)), dim=1),
        torch.cat((torch.zeros(1, 1, bs * groups, device=device), 1 - c, c, torch.zeros(1, 6, bs * groups, device=device)), dim=1),
        torch.cat((torch.zeros(1, 3, bs * groups, device=device), x + b, 1 - a - b, torch.zeros(1, 1, bs * groups, device=device), -b, b - y, torch.zeros(1, 1, bs * groups, device=device)), dim=1),
        torch.cat((torch.zeros(1, 4, bs * groups, device=device), torch.ones(1, 1, bs * groups, device=device), torch.zeros(1, 4, bs * groups, device=device)), dim=1),
        torch.cat((torch.zeros(1, 1, bs * groups, device=device), b - y, -b, torch.zeros(1, 1, bs * groups, device=device), 1 - a - b, x + b, torch.zeros(1, 3, bs * groups, device=device)), dim=1),
        torch.cat((torch.zeros(1, 6, bs * groups, device=device), c, 1 - c, torch.zeros(1, 1, bs * groups, device=device)), dim=1),
        torch.cat((torch.zeros(1, 4, bs * groups, device=device), 1 - a - b, b - y, torch.zeros(1, 1, bs * groups, device=device), x + b, -b), dim=1),
        torch.cat((torch.zeros(1, 5, bs * groups, device=device), 1 - c, torch.zeros(1, 2, bs * groups, device=device), c), dim=1),
    ), dim=0)

    mask = (thetas >= 0).unsqueeze(0).unsqueeze(0).float()
    rot_mat = mask * rot_mat_positive + (1 - mask) * rot_mat_negative
    rot_mat = rot_mat.permute(2, 0, 1).reshape(bs, groups, rot_mat.shape[0], rot_mat.shape[1])
    return rot_mat


def batch_rotate_multiweight(weights: torch.Tensor, lambdas: torch.Tensor, thetas: torch.Tensor) -> torch.Tensor:
    assert thetas.shape == lambdas.shape
    assert lambdas.shape[1] == weights.shape[0]
    batch_size, kernel_number = thetas.shape
    kernel_size = weights.shape[-1]
    _, cout, cin, _, _ = weights.shape

    if kernel_size != 3:
        raise NotImplementedError("Only 3x3 adaptive rotated kernels are currently supported.")

    rotation_matrix = _get_rotation_matrix(thetas)
    lambdas = lambdas.unsqueeze(2).unsqueeze(3)
    rotation_matrix = torch.mul(rotation_matrix, lambdas)
    rotation_matrix = rotation_matrix.permute(0, 2, 1, 3).reshape(batch_size * kernel_size * kernel_size, kernel_number * kernel_size * kernel_size)

    weights = weights.permute(0, 3, 4, 1, 2).contiguous().view(kernel_number * kernel_size * kernel_size, cout * cin)
    weights = torch.mm(rotation_matrix, weights)
    weights = weights.contiguous().view(batch_size, kernel_size, kernel_size, cout, cin)
    weights = weights.permute(0, 3, 4, 1, 2).reshape(batch_size * cout, cin, kernel_size, kernel_size)
    return weights


class AdaptiveRotatedConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        kernel_number: int = 1,
        rounting_func: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.kernel_number = kernel_number
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.rounting_func = rounting_func
        self.weight = nn.Parameter(
            torch.Tensor(kernel_number, out_channels, in_channels // groups, kernel_size, kernel_size)
        )
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alphas, angles = self.rounting_func(x)
        rotated_weight = batch_rotate_multiweight(self.weight, alphas, angles)
        bs, cin, height, width = x.shape
        x = x.reshape(1, bs * cin, height, width)
        out = F.conv2d(
            input=x,
            weight=rotated_weight,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups * bs,
        )
        return out.reshape(bs, self.out_channels, *out.shape[2:])
