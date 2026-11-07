"""Optional PyTorch oracle for the trainable speckle score detector."""

from __future__ import annotations

from typing import Any


def build_dic_score_model() -> Any:
    """Build the small convolutional network used by the research trainer.

    PyTorch is imported only when this function is called. The network consists
    of a 1→8 3×3 convolution, an 8-channel depthwise 3×3 convolution, and an
    8→1 pointwise score head. Its parameter layout is intentionally simple so
    it can be evaluated by a small WGSL kernel.
    """
    try:
        from torch import nn
    except ImportError as error:
        raise ImportError(
            "The trainable DIC detector requires the optional PyTorch package"
        ) from error

    class DICScoreNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
            self.depthwise = nn.Conv2d(8, 8, kernel_size=3, padding=1, groups=8)
            self.score = nn.Conv2d(8, 1, kernel_size=1)
            self.activation = nn.SiLU()

        def forward(self, image: Any) -> Any:
            features = self.activation(self.conv1(image))
            features = self.activation(self.depthwise(features))
            return self.score(features)

    return DICScoreNet()
