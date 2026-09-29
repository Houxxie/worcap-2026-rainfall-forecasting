"""A compact U-Net for continuous rainfall anomalies, not image classification."""
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Some CUDA interpolation backward kernels have no deterministic implementation.
    # Keep the warning visible and archive the environment; do not promise bit equality.
    torch.use_deterministic_algorithms(True, warn_only=True)


class Block(nn.Sequential):
    def __init__(self, inputs, outputs):
        super().__init__(nn.Conv2d(inputs, outputs, 3, padding=1, padding_mode='replicate'),
                         nn.GroupNorm(4, outputs), nn.SiLU(),
                         nn.Conv2d(outputs, outputs, 3, padding=1, padding_mode='replicate'),
                         nn.GroupNorm(4, outputs), nn.SiLU())


class RainfallUNet(nn.Module):
    def __init__(self, inputs=31, widths=(16, 32, 64)):
        super().__init__()
        a, b, c = widths
        self.encoder1 = Block(inputs, a)
        self.encoder2 = Block(a, b)
        self.bottleneck = Block(b, c)
        self.decoder2 = Block(c + b, b)
        self.decoder1 = Block(b + a, a)
        self.output = nn.Conv2d(a, 1, 1)
        # Start at the climatology. Negative anomalies remain learnable.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(F.avg_pool2d(e1, 2))
        b = self.bottleneck(F.avg_pool2d(e2, 2))
        d2 = self.decoder2(torch.cat([F.interpolate(b, size=e2.shape[-2:], mode='bilinear', align_corners=False), e2], 1))
        d1 = self.decoder1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode='bilinear', align_corners=False), e1], 1))
        return self.output(d1)[:, 0]


def restore(path, device='cpu'):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = RainfallUNet(widths=tuple(checkpoint['widths'])).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, checkpoint
