import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class IntensityLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, gts):

        return torch.mean((preds - gts) ** 2)

class GradientLoss(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.in_channels = in_channels

        kernel_x = np.array([-1, 1])
        kernel_x = torch.stack([torch.from_numpy(kernel_x.astype(np.float32))] * self.in_channels)
        self.kernel_x = kernel_x.view(self.in_channels, 1, 1, 2)

        kernel_y = np.array([[-1], [1]])
        kernel_y = torch.stack([torch.from_numpy(kernel_y.astype(np.float32))] * self.in_channels)
        self.kernel_y = kernel_y.view(self.in_channels, 1, 2, 1)

    def forward(self, preds, gts):

        preds_x = F.pad(preds, [1,0,0,0])
        preds_y = F.pad(preds, [0,0,1,0])

        gts_x = F.pad(gts, [1,0,0,0])
        gts_y = F.pad(gts, [0,0,1,0])

        preds_dx = torch.abs(F.conv2d(preds_x, self.kernel_x, groups=self.in_channels))
        preds_dy = torch.abs(F.conv2d(preds_y, self.kernel_y, groups=self.in_channels))

        gts_dx = torch.abs(F.conv2d(gts_x, self.kernel_x, groups=self.in_channels))
        gts_dy = torch.abs(F.conv2d(gts_y, self.kernel_y, groups=self.in_channels))

        diff_dx = torch.abs(preds_dx - gts_dx)
        diff_dy = torch.abs(preds_dy - gts_dy)

        return torch.mean(diff_dx + diff_dy)

class FlowLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, gts):

        return torch.mean(torch.abs(preds - gts))

class GeneratorAdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, fake):

        return torch.mean((fake - 1) ** 2 / 2)

class DiscriminatorAdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, real, fake):
        return torch.mean((real - 1) ** 2 / 2) + torch.mean(fake ** 2 / 2)