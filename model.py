from torch.nn import Module
import torch.nn as nn
from torchvision.models.resnet import resnet50, resnet18, resnet101
from torchvision.models import convnext_tiny
import torch


def _backbone_resnet18(*args, **kwargs):
    model = resnet18(*args, **kwargs)
    model.fc = nn.Identity()
    return model, 512


def _backbone_resnet50(*args, **kwargs):
    model = resnet50(*args, **kwargs)
    model.fc = nn.Identity()
    return model, 2048


def _backbone_resnet101(*args, **kwargs):
    model = resnet101(*args, **kwargs)
    model.fc = nn.Identity()
    return model, 2048


def _backbone_convnext_tiny(*args, **kwargs):
    model = convnext_tiny(*args, **kwargs)
    model.classifier = nn.Identity()
    return model, 768


class Model(Module):
    def __init__(self, backbone='resnet50', output_size=10):
        super().__init__()
        backbone, feat_size = globals(
        )[f'_backbone_{backbone}'](pretrained=True)
        self.backbone = backbone
        self.fc = nn.Sequential(
            nn.LayerNorm(feat_size, eps=1e-6, elementwise_affine=True),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(feat_size, output_size, bias=True),
        )
        self.forward_feat = {}

    def forward(self, x):
        self.forward_feat.clear()
        feat = self.backbone(x)
        feat = feat.view(feat.shape[0], -1)
        out = self.fc(feat)
        return out


if __name__ == "__main__":
    m = Model(backbone='resnet50',output_size=10)
    output = m(torch.randn(1, 3, 512, 512))
    print(output)
