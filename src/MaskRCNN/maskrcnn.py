from collections import OrderedDict
from torch import nn
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from torchvision.ops import misc as misc_nn_ops
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from MaskRCNN import resnet
from torchvision.models.detection.rpn import AnchorGenerator
import torch
from torch.nn import functional as F
import numpy as np
import torchvision

def normalize(self, image):
    dtype, device = image.dtype, image.device
    mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
    std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
    return (image - mean[:, None, None, None]) / std[:, None, None, None]

def forward(self, images, targets=None):
    for i in range(len(images)):
        image = images[i]
        target = targets[i] if targets is not None else targets
        if image.dim() != 3:
            raise ValueError("images is expected to be a list of 3d tensors "
                                "of shape [C, H, W], got {}".format(image.shape))
        # image = self.normalize(image)
        # image, target = self.resize(image, target)
        images[i] = image
        if targets is not None:
            targets[i] = target
    image_sizes = [img.shape[-2:] for img in images]
    images = self.batch_images(images)
    image_list = ImageList(images, image_sizes)
    return image_list, targets

class IntermediateLayerGetter(nn.ModuleDict):

    def __init__(self, model, return_layers, confidence=None, fusion=None):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break
        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers
        self.confidence = confidence
        if confidence == 'map':
            self.conf_layer1 = nn.Conv2d(2, 1, 3, 1, 2)
            self.conf_layer2 = nn.Conv2d(1, 1, 3, 1, 2)
            self.conf_layer3 = nn.Conv2d(1, 1, 1, 1, 0)
            self.conf_layer4 = nn.Conv2d(1, 1, 1, 1, 0)
            self.conf_layer5 = nn.Conv2d(1, 1, 1, 1, 0)
        self.fusion = fusion

    def forward(self, x_in):
        out = OrderedDict()

        if self.confidence != 'None' or self.fusion:
            x = x_in[:, :4, :, :]
        else:
            x = x_in[:, :3, :, :]

        if self.confidence == 'map':
            conf_map = x_in[:, -2:, :, :]
            conf_map = self.conf_layer1(conf_map)
            conf_map = self.conf_layer2(conf_map)
            conf_map = self.conf_layer3(conf_map)
            conf_map = self.conf_layer4(conf_map)
            conf_map = self.conf_layer5(conf_map)
        elif self.confidence == 'mask':
            conf_map = x_in[:, -1, :, :].unsqueeze(1)

        if self.confidence != "None":
            conf_maps = []
            _, _, H, W = x_in.shape
            for i in range(4):
                conf_maps.append(F.interpolate(conf_map, \
                    size=(int(H/(2**(i+2))), int(W/(2**(i+2)))), mode='bilinear', align_corners=True))

        for name, module in self.named_children():

            if 'conf' in name:
                continue
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                if self.confidence != "None":
                    out[out_name] = x * conf_maps[out_name]
                else:
                    out[out_name] = x

        return out

class IntermediateLayerGetterRGBD(nn.ModuleDict):

    def __init__(self, rgb_model, depth_model, return_layers, confidence=None):
        if not set(return_layers).issubset([name for name, _ in rgb_model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        rgb_return_layers = {k: v for k, v in return_layers.items()}
        depth_return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()

        for name, module in rgb_model.named_children():
            layers['rgb_' + name] = module
            if name in rgb_return_layers:
                del rgb_return_layers[name]
            if not rgb_return_layers:
                break

        for name, module in depth_model.named_children():
            layers['depth_' + name] = module
            if name in depth_return_layers:
                del depth_return_layers[name]
            if not depth_return_layers:
                break

        super(IntermediateLayerGetterRGBD, self).__init__(layers)

        self.fuse_layer1 = nn.Conv2d(512, 256, 3, 1, 2)
        self.fuse_layer2 = nn.Conv2d(1024, 512, 3, 1, 2)
        self.fuse_layer3 = nn.Conv2d(2048, 1024, 3, 1, 2)
        self.fuse_layer4 = nn.Conv2d(4096, 2048, 3, 1, 2)
        self.return_layers = orig_return_layers
        self.confidence = confidence
        if confidence == 'map':
            self.conf_layer1 = nn.Conv2d(2, 1, 3, 1, 2)
            self.conf_layer2 = nn.Conv2d(1, 1, 3, 1, 2)
            self.conf_layer3 = nn.Conv2d(1, 1, 1, 1, 0)
            self.conf_layer4 = nn.Conv2d(1, 1, 1, 1, 0)
            self.conf_layer5 = nn.Conv2d(1, 1, 1, 1, 0)

    def forward(self, x_in):
        out = OrderedDict()
        rgb_out = OrderedDict()
        depth_out = OrderedDict()
        rgb_x = x_in[:, :3, :, :]
        depth_x = x_in[:, 3:6, :, :]

        if self.confidence == 'map':
            conf_map = x_in[:, -2:, :, :]
            conf_map = self.conf_layer1(conf_map)
            conf_map = self.conf_layer2(conf_map)
            conf_map = self.conf_layer3(conf_map)
            conf_map = self.conf_layer4(conf_map)
            conf_map = self.conf_layer5(conf_map)
        elif self.confidence == 'mask':
            conf_map = x_in[:, -1, :, :].unsqueeze(1)
        if self.confidence != "None":
            conf_maps = []
            _, _, H, W = x_in.shape
            for i in range(4):
                conf_maps.append(F.interpolate(conf_map, \
                    size=(int(H/(2**(i+2))), int(W/(2**(i+2)))), mode='bilinear', align_corners=True))

        for i, (name, module) in enumerate(self.named_children()):
            layer_name = name.split('_')[-1]
            if 'rgb' in name:
                rgb_x = module(rgb_x)
                if layer_name in self.return_layers:
                    out_name = self.return_layers[layer_name]
                    rgb_out[out_name] = rgb_x
            elif 'depth' in name:
                # Confidence
                depth_x = module(depth_x)
                if layer_name in self.return_layers:
                    out_name = self.return_layers[layer_name]
                    if self.confidence != "None":
                        depth_out[out_name] = depth_x * conf_maps[out_name]
                    else:
                        depth_out[out_name] = depth_x

        # RGB-D Fusion
        for i in range(4):
            out[i] = torch.cat((rgb_out[i], depth_out[i]), dim=1)

        out[0] = self.fuse_layer1(out[0])
        out[1] = self.fuse_layer2(out[1])
        out[2] = self.fuse_layer3(out[2])
        out[3] = self.fuse_layer4(out[3])
        return out

class BackboneWithFPN(nn.Sequential):

    def __init__(self, backbone, return_layers, in_channels_list, out_channels, confidence, fusion=None):

        body = IntermediateLayerGetter(backbone, return_layers, confidence, fusion)
        fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool(),
        )
        super(BackboneWithFPN, self).__init__(OrderedDict(
            [("body", body), ("fpn", fpn)]))
        self.out_channels = out_channels

class BackboneWithFPNRGBD(nn.Sequential):

    def __init__(self, rgb_backbone, depth_backbone, return_layers, in_channels_list, out_channels, confidence):

        body = IntermediateLayerGetterRGBD(rgb_backbone, depth_backbone, return_layers=return_layers, confidence=confidence)
        fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool(),
        )
        super(BackboneWithFPNRGBD, self).__init__(OrderedDict(
            [("body", body), ("fpn", fpn)]))
        self.out_channels = out_channels


def resnet_fpn_backbone(backbone_name, pretrained, config):

    return_layers = {'layer1': 0, 'layer2': 1, 'layer3': 2, 'layer4': 3}

    in_channels_stage2 = 256
    in_channels_list = [
        in_channels_stage2,
        in_channels_stage2 * 2,
        in_channels_stage2 * 4,
        in_channels_stage2 * 8,
    ]
    out_channels = 256

    if config["input_type"] == 'rgbd' and config["fusion"] == "late":
        rgb_backbone = resnet.__dict__[backbone_name](
            pretrained=True,
            norm_layer=misc_nn_ops.FrozenBatchNorm2d)

        depth_backbone = resnet.__dict__[backbone_name](
            pretrained=False,
            norm_layer=misc_nn_ops.FrozenBatchNorm2d)
        # freeze layers
        for name, parameter in rgb_backbone.named_parameters():
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        for name, parameter in depth_backbone.named_parameters():
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        return BackboneWithFPNRGBD(rgb_backbone, depth_backbone, return_layers, \
                in_channels_list, out_channels, config["confidence"])

    elif config["input_type"] == 'rgbd' and config["fusion"] == "early":
        backbone = resnet.__dict__[backbone_name](
            pretrained=True,
            norm_layer=misc_nn_ops.FrozenBatchNorm2d)
        backbone.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # freeze layers
        for name, parameter in backbone.named_parameters():
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        return BackboneWithFPN(backbone, return_layers, \
                in_channels_list, out_channels, config["confidence"], config["fusion"])

    else:
        if config["input_type"] == 'rgb':
            pretrained = True
        elif config["input_type"] == 'depth':
            pretrained = False
        backbone = resnet.__dict__[backbone_name](
            pretrained=pretrained,
            norm_layer=misc_nn_ops.FrozenBatchNorm2d)

        if config["confidence"] != "None":
            backbone.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # freeze layers
        for name, parameter in backbone.named_parameters():
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        return BackboneWithFPN(backbone, return_layers, \
            in_channels_list, out_channels, config["confidence"], config["fusion"])

def maskrcnn_resnet50_fpn(pretrained=False, progress=True,
                          num_classes=91, pretrained_backbone=True, config=None, **kwargs):

    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False
    backbone = resnet_fpn_backbone('resnet50', pretrained_backbone, config)
    sizes = (32, 64, 128, 256, 512)
    # roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'],
    #                                                 output_size=7,
    #                                                 sampling_ratio=2)
    anchor_generator = AnchorGenerator(sizes=(sizes), aspect_ratios=((0.5, 1.0, 2.0)))
    model = MaskRCNN(backbone, num_classes, rpn_anchor_generator=anchor_generator, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['maskrcnn_resnet50_fpn_coco'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def get_model_instance_segmentation(num_classes, config):

    GeneralizedRCNNTransform.forward = forward

    model = maskrcnn_resnet50_fpn(num_classes=num_classes, pretrained=False, config=config)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model
