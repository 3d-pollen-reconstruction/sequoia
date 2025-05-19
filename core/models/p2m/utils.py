from .vgg_backbone import VGG16TensorflowAlign, VGG16P2M, VGG16Recons
from .resnet_backbone import resnet50

PRETRAINED_WEIGHTS_PATH = './notebooks/p2m/resnet.pth.tar'

def get_backbone(options):
    if options.backbone.startswith("vgg16"):
        if options.align_with_tensorflow:
            nn_encoder = VGG16TensorflowAlign()
        else:
            print("Loading vgg16-p2m...")
            nn_encoder = VGG16P2M(pretrained=True)
        nn_decoder = VGG16Recons()
    elif options.backbone == "resnet50":
        nn_encoder = resnet50(backbone_ckpt=PRETRAINED_WEIGHTS_PATH, in_ch=3)
        nn_decoder = None
    else:
        raise NotImplementedError("No implemented backbone called '%s' found" % options.backbone)
    return nn_encoder, nn_decoder