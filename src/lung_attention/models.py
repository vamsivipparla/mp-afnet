"""Multi-patch custom, residual, and dense attention CNNs."""
import torch
from torch import nn
from torch.nn import functional as F


class AttentionGate3D(nn.Module):
    """Additive gate: sigmoid(psi(ReLU(Wx*x + Wg*g))), then alpha*x."""

    def __init__(self, channels):
        super().__init__()
        hidden = max(1, channels // 2)
        self.x_projection = nn.Conv3d(channels, hidden, 1)
        self.g_projection = nn.Conv3d(channels, hidden, 1)
        self.score = nn.Conv3d(hidden, 1, 1)

    def forward(self, x, g):
        context = F.interpolate(self.g_projection(g), size=x.shape[2:],
                                mode="trilinear", align_corners=False)
        alpha = torch.sigmoid(self.score(F.relu(self.x_projection(x) + context)))
        return x * alpha


class ContextGate(nn.Module):
    """Use a coarser view of the stage output as gating context."""

    def __init__(self, channels, enabled):
        super().__init__()
        self.gate = AttentionGate3D(channels) if enabled else None

    def forward(self, x):
        if self.gate is None:
            return x
        size = tuple(max(1, d // 2) for d in x.shape[2:])
        return self.gate(x, F.adaptive_avg_pool3d(x, size))


def conv(in_channels, out_channels, kernel=3, stride=1, activate=True):
    layers = [nn.Conv3d(in_channels, out_channels, kernel, stride,
                        padding=kernel // 2, bias=False), nn.BatchNorm3d(out_channels)]
    if activate:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class CustomBranch(nn.Module):
    def __init__(self, size, width, attention):
        super().__init__()
        c = [max(2, int(v * width)) for v in (64, 128, 256, 512)]
        layers = [nn.AvgPool3d((2, 1, 1))]
        previous = 1
        for i, channels in enumerate(c):
            layers += [nn.Conv3d(previous, channels, 3, padding=1), nn.ReLU()]
            # Figure 9 depicts repeated convolutions in the last two stages.
            if i >= 2:
                layers += [nn.Conv3d(channels, channels, 3, padding=1), nn.ReLU()]
            layers += [ContextGate(channels, attention),
                       nn.MaxPool3d((1, 2, 2) if i == 0 else 2)]
            previous = channels
        if size > 16:
            last = c[-1] if size == 32 else c[0]
            layers += [nn.Conv3d(previous, last, 2), nn.ReLU()]
            self.output_features = last * (1 if size == 32 else 8)
        else:
            self.output_features = c[-1]
        self.features = nn.Sequential(*layers)

    def forward(self, x):
        return self.features(x).flatten(1)


class ResidualBlock(nn.Module):
    def __init__(self, channels, shrink):
        super().__init__()
        out = channels * 2 if shrink else channels
        inner = max(1, channels // (2 if shrink else 4))
        stride = 2 if shrink else 1
        self.path = nn.Sequential(conv(channels, inner, 1, stride),
                                  conv(inner, inner), conv(inner, out, 1, activate=False))
        self.shortcut = conv(channels, out, 3, stride, False) if shrink else nn.Identity()

    def forward(self, x):
        return F.relu(self.path(x) + self.shortcut(x))


class ResidualBranch(nn.Module):
    def __init__(self, size, width, attention):
        super().__init__()
        channels = max(4, int(16 * width))
        layers = [conv(1, channels)]
        for index in range(16):
            shrink = index % 4 == 0
            layers.append(ResidualBlock(channels, shrink))
            channels *= 2 if shrink else 1
            if index % 4 == 3:
                layers.append(ContextGate(channels, attention))
        self.features = nn.Sequential(*layers)
        self.output_features = channels

    def forward(self, x):
        return F.adaptive_max_pool3d(self.features(x), 1).flatten(1)


class DenseBlock(nn.Module):
    def __init__(self, channels, growth):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(nn.BatchNorm3d(channels + i * growth), nn.ReLU(),
                          nn.Conv3d(channels + i * growth, growth, 3, padding=1, bias=False))
            for i in range(5)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = torch.cat((x, layer(x)), dim=1)
        return x


class DenseBranch(nn.Module):
    def __init__(self, size, width, attention):
        super().__init__()
        channels, growth = max(2, int(16 * width)), max(1, int(8 * width))
        layers = [nn.Conv3d(1, channels, 3, padding=1)]
        for index in range(3):
            layers.append(DenseBlock(channels, growth))
            channels += 5 * growth
            layers.append(ContextGate(channels, attention))
            if index < 2:
                # Preserve the accumulated channels at each spatial transition.
                layers += [nn.Conv3d(channels, channels, 1), nn.AvgPool3d(2)]
        self.features = nn.Sequential(*layers)
        self.output_features = channels

    def forward(self, x):
        return F.adaptive_avg_pool3d(self.features(x), 1).flatten(1)


class MultiPatchCNN(nn.Module):
    """Three independent scale branches with a two-class logit head."""

    def __init__(self, family="custom", attention=True, width=1.0, sizes=(16, 32, 48)):
        super().__init__()
        if family not in {"custom", "residual", "dense"} or not 0 < width <= 1:
            raise ValueError("Invalid family or width (must be in (0, 1])")
        if not sizes or len(set(sizes)) != len(sizes) or any(s not in (16, 32, 48) for s in sizes):
            raise ValueError("Sizes must be a nonempty unique subset of 16, 32, 48")
        self.spec = dict(family=family, attention=attention, width=width, sizes=list(sizes))
        branch = {"custom": CustomBranch, "residual": ResidualBranch, "dense": DenseBranch}[family]
        self.branches = nn.ModuleList([branch(s, width, attention) for s in sizes])
        hidden = max(8, int({"custom": 512, "residual": 256, "dense": 136}[family] * width))
        self.classifier = nn.Sequential(nn.Dropout(0.5),
                                        nn.Linear(sum(b.output_features for b in self.branches), hidden),
                                        nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, patches):
        if len(patches) != len(self.branches):
            raise ValueError("Provide one tensor per configured patch size")
        for patch, size in zip(patches, self.spec["sizes"]):
            if tuple(patch.shape[1:]) != (1, size, size, size):
                raise ValueError("Expected [batch, 1, size, size, size]")
        return self.classifier(torch.cat([b(x) for b, x in zip(self.branches, patches)], dim=1))
