import torch
from torch import nn
from torchvision.models import ResNet
from torchvision.models.resnet import Bottleneck

PRETRAINED_WEIGHTS_PATH = './notebooks/p2m/resnet.pth'

class P2MResNet(ResNet):
    def __init__(self, *args, **kwargs):
        self.output_dim = 0
        super().__init__(*args, **kwargs)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        res = super()._make_layer(block, planes, blocks, stride=stride, dilate=dilate)
        self.output_dim += self.inplanes
        return res

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        features = []
        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        x = self.layer4(x)
        features.append(x)

        return features

    @property
    def features_dim(self):
        return self.output_dim


def resnet50(backbone_ckpt: str, in_ch: int = 3):
    model = P2MResNet(Bottleneck, [3, 4, 6, 3])

    ckpt = torch.load(backbone_ckpt, map_location="cpu")
    state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))

    # strip common prefixes
    clean_dict = {}
    for k, v in state_dict.items():
        for prefix in ("module.", "backbone.", "model."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        clean_dict[k] = v

    # -------- adapt first conv if we want 1-channel images --------
    if in_ch == 1:
        # find the key that ends with "conv1.weight"
        conv1_key = next(
            k for k in clean_dict.keys() if k.endswith("conv1.weight")
        )
        w = clean_dict[conv1_key]                # (64,3,7,7) typically
        clean_dict[conv1_key] = w.mean(1, keepdim=True)

        # replace layer in the model definition too
        model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

    # load what we have, ignore extras (fc.* etc.)
    model.load_state_dict(clean_dict, strict=False)
    return model
