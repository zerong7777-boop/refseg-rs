from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .arc import AdaptiveRotatedConv2d, RountingFunction


class XeMapSegSimpleDecoding(nn.Module):
    def __init__(self, c4_dims: int, factor: int = 2) -> None:
        super().__init__()
        hidden_size = c4_dims // factor
        c4_size = c4_dims
        c3_size = c4_dims // (factor ** 1)
        c2_size = c4_dims // (factor ** 2)
        c1_size = c4_dims // (factor ** 3)

        self.adjust_c4 = self._build_adjust_block(c4_dims, c4_size)
        self.adjust_c3 = self._build_adjust_block(c4_dims, c3_size)
        self.adjust_c2 = self._build_adjust_block(c4_dims, c2_size)
        self.adjust_c1 = self._build_adjust_block(c4_dims, c1_size)

        self.conv1_4 = nn.Conv2d(c4_size + c3_size, hidden_size, 3, padding=1, bias=False)
        self.conv2_4 = AdaptiveRotatedConv2d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=3,
            padding=1,
            rounting_func=RountingFunction(in_channels=hidden_size, kernel_number=1),
            bias=False,
            kernel_number=1,
        )
        self.bn1_4 = nn.BatchNorm2d(hidden_size)
        self.relu1_4 = nn.ReLU()
        self.bn2_4 = nn.BatchNorm2d(hidden_size)
        self.relu2_4 = nn.ReLU()

        self.conv1_3 = nn.Conv2d(hidden_size + c2_size, hidden_size, 3, padding=1, bias=False)
        self.conv2_3 = AdaptiveRotatedConv2d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=3,
            padding=1,
            rounting_func=RountingFunction(in_channels=hidden_size, kernel_number=1),
            bias=False,
            kernel_number=1,
        )
        self.bn1_3 = nn.BatchNorm2d(hidden_size)
        self.relu1_3 = nn.ReLU()
        self.bn2_3 = nn.BatchNorm2d(hidden_size)
        self.relu2_3 = nn.ReLU()

        self.conv1_2 = nn.Conv2d(hidden_size + c1_size, hidden_size, 3, padding=1, bias=False)
        self.conv2_2 = AdaptiveRotatedConv2d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=3,
            padding=1,
            rounting_func=RountingFunction(in_channels=hidden_size, kernel_number=1),
            bias=False,
            kernel_number=1,
        )
        self.bn1_2 = nn.BatchNorm2d(hidden_size)
        self.relu1_2 = nn.ReLU()
        self.bn2_2 = nn.BatchNorm2d(hidden_size)
        self.relu2_2 = nn.ReLU()

        self.conv1_1 = nn.Conv2d(hidden_size, 1, 1)
        self._initialize_modules()

    @staticmethod
    def _build_adjust_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        )

    def _initialize_modules(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight.data)
                if module.bias is not None:
                    nn.init.constant_(module.bias.data, 0)
            elif isinstance(module, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(module.weight.data, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias.data, 0)
            elif isinstance(module, RountingFunction):
                nn.init.xavier_uniform_(module.dwc.weight.data)
                nn.init.xavier_uniform_(module.fc_alpha.weight.data)
                nn.init.xavier_uniform_(module.fc_theta.weight.data)
            elif isinstance(module, AdaptiveRotatedConv2d):
                nn.init.kaiming_normal_(module.weight.data, mode="fan_out", nonlinearity="relu")
                if module.bias is not None and isinstance(module.bias, torch.Tensor):
                    nn.init.constant_(module.bias.data, 0)

    def forward(self, img_features_with_text: Sequence[torch.Tensor]) -> torch.Tensor:
        x_c4 = self.adjust_c4(img_features_with_text[3])
        x_c3 = self.adjust_c3(img_features_with_text[2])
        x_c2 = self.adjust_c2(img_features_with_text[1])
        x_c1 = self.adjust_c1(img_features_with_text[0])

        if x_c4.size(-2) < x_c3.size(-2) or x_c4.size(-1) < x_c3.size(-1):
            x_c4 = F.interpolate(x_c4, scale_factor=2, mode="bilinear", align_corners=True)
        x = self.relu2_4(self.bn2_4(self.conv2_4(self.relu1_4(self.bn1_4(self.conv1_4(torch.cat([x_c4, x_c3], dim=1)))))))

        if x.size(-2) < x_c2.size(-2) or x.size(-1) < x_c2.size(-1):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = self.relu2_3(self.bn2_3(self.conv2_3(self.relu1_3(self.bn1_3(self.conv1_3(torch.cat([x, x_c2], dim=1)))))))

        if x.size(-2) < x_c1.size(-2) or x.size(-1) < x_c1.size(-1):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = self.relu2_2(self.bn2_2(self.conv2_2(self.relu1_2(self.bn1_2(self.conv1_2(torch.cat([x, x_c1], dim=1)))))))
        return self.conv1_1(x)
