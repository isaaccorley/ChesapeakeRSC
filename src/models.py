import torch.nn as nn
from torch import Tensor
from torch.nn.modules import Module


class CustomFCN(Module):
    """A simple 5 layer FCN with 'same' padding."""

    def __init__(self, in_channels: int, classes: int, num_filters: int = 64) -> None:
        """Initializes the 5 layer FCN model.

        Args:
            in_channels: Number of input channels that the model will expect
            classes: Number of filters in the final layer
            num_filters: Number of filters in initial conv layer, halved in every
                successive layer
        """
        super().__init__()

        modules = [
            nn.modules.Conv2d(in_channels, num_filters, kernel_size=3, stride=1, padding=1)  # noqa: E501
        ]
        current_filters = num_filters
        for i in range(4):
            modules.append(nn.modules.Conv2d(current_filters, current_filters // 2, kernel_size=3, stride=1, padding=1))  # noqa: E501
            modules.append(nn.modules.ReLU(inplace=False))
            modules.append(nn.modules.BatchNorm2d(current_filters // 2))
            current_filters = current_filters // 2

        self.backbone = nn.modules.Sequential(*modules)
        self.last = nn.modules.Conv2d(
            current_filters, classes, kernel_size=1, stride=1, padding=0
        )

        # initialize all kernels
        for module in modules:
            if isinstance(module, nn.modules.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        nn.init.kaiming_normal_(self.last.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.last.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model."""
        x = self.backbone(x)
        x = self.last(x)
        return x
