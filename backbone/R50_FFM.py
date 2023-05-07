from train_utils.paths_catalog import check_pretrain_file
from typing import TypeVar
from torch.nn.modules.batchnorm import _BatchNorm
import torch
import torch.nn as nn
from torchvision.ops.misc import FrozenBatchNorm2d
from .feature_pyramid_network import BackboneWithFPN, LastLevelMaxPool, IntermediateLayerGetter
from torch.jit.annotations import Tuple, List, Dict
import torch.nn.functional as F
T = TypeVar('T', bound='Module')


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = norm_layer(out_channel)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = norm_layer(out_channel)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = norm_layer(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, blocks_num, num_classes=1000, include_top=True, norm_layer=None, frozen_stages=1):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.include_top = include_top
        self.in_channel = 64

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = norm_layer(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.frozen_stages = frozen_stages
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample,
                            stride=stride, norm_layer=norm_layer))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.bn1.eval()
            for m in [self.conv1, self.bn1]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False



    def train(self, mode=True, norm_layer=None):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(ResNet, self).train(mode)
        self._freeze_stages()
        if mode and not isinstance(norm_layer, FrozenBatchNorm2d):
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x

class FFM(nn.Module):
    def __init__(self, layer_nums, in_channels=1024, out_channels=256):
        super().__init__()
        ffm = nn.ModuleList()
        for i in range(layer_nums):
            in_channel = int(in_channels * 2 ** i)
            lis = [nn.Conv2d(in_channels=in_channel, out_channels=1024,
                             kernel_size=1, stride=1), nn.BatchNorm2d(1024), nn.ReLU(inplace=True)]
            ffm.append(nn.Sequential(*lis))
        self.ffm = ffm
        last_1v1 = nn.Conv2d(in_channels=in_channels * layer_nums, out_channels=out_channels,
                             kernel_size=1, stride=1)
        self.last_1v1 = last_1v1
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        for modules in [self.ffm, self.last_1v1]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

    def forward(self, x: Dict, mode: str ='bilinear'):
        x = list(x.values())
        feat_shape = x[0].shape[-2:]
        out_layers = []
        for i, l in enumerate(x):
            l = self.ffm[i](l)
            if l.shape[-2:] != feat_shape:
                l = F.interpolate(l, size=feat_shape, mode=mode, align_corners=False)
            out_layers.append(l)
        y = torch.cat(out_layers, axis=1)
        y = self.last_1v1(y)
        edge_feature = self.relu(self.bn(y))
        return edge_feature

class R50_FFM(nn.Module):
    """
    Adds a FPN on top of a model.
    Internally, it uses torchvision.models._utils.IntermediateLayerGetter to
    extract a submodel that returns the feature maps specified in return_layers.
    The same limitations of IntermediatLayerGetter apply here.
    Arguments:
        backbone (nn.Module)
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        in_channels_list (List[int]): number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        out_channels (int): number of channels in the FPN.
        extra_blocks: ExtraFPNBlock
    Attributes:
        out_channels (int): the number of channels in the FPN
    """

    def __init__(self,
                 backbone: nn.Module,
                 return_layers=None,
                 in_channels_list=None,
                 out_channels=256,
                 extra_blocks=None,
                 re_getter=True,):
        super().__init__()

        # if extra_blocks is None and use_extra_blocks:
        #     extra_blocks = LastLevelMaxPool()
        extra_blocks = extra_blocks

        if re_getter is True:
            assert return_layers is not None
            self.re_r50 = IntermediateLayerGetter(backbone, return_layers=return_layers)
        else:
            self.re_r50 = backbone

        layer_nums = len(return_layers)
        self.ffm = FFM(layer_nums, in_channels=1024, out_channels=256)

    def forward(self, x):
        x = self.re_r50(x)
        x = self.ffm(x)
        return x

def overwrite_eps(model, eps):
    """
    This method overwrites the default eps values of all the
    FrozenBatchNorm2d layers of the model with the provided value.
    This is necessary to address the BC-breaking change introduced
    by the bug-fix at pytorch/vision#2933. The overwrite is applied
    only when the pretrained weights are loaded to maintain compatibility
    with previous versions.

    Args:
        model (nn.Module): The model on which we perform the overwrite.
        eps (float): The new value of eps.
    """
    for module in model.modules():
        if isinstance(module, FrozenBatchNorm2d):
            module.eps = eps


def r50_ffm(norm_layer=torch.nn.BatchNorm2d, frozen_stages=1, returned_num_layers=2, adjust_stem=False, pretrain=True):
    """
    r50_ffm——backbone
    Args:
        norm_layer: torch.nn.BatchNorm2d
        frozen_stages: Specify which layers need to be frozen
        returned_num_layers: Specify which layers of output need to be returned
        adjust_stem: Adjusting the channel
        pretrain:
    Returns:

    """
    # frozen_stages = -1 if adjust_stem else frozen_stages
    resnet_backbone = ResNet(Bottleneck,
                             [3, 4, 6, 3],
                             include_top=False,
                             norm_layer=norm_layer,
                             frozen_stages=frozen_stages)

    if isinstance(norm_layer, FrozenBatchNorm2d):
        overwrite_eps(resnet_backbone, 0.0)

    if pretrain:
        pretrain_path = check_pretrain_file()
        print(f'r50_ffm loading pretrain file "{pretrain_path}".',
              resnet_backbone.load_state_dict(torch.load(pretrain_path), strict=False))

    # freeze layers
    resnet_backbone.train(norm_layer=norm_layer)
    if adjust_stem:
        resnet_backbone.conv1 = nn.Conv2d(4, 64, kernel_size=7, padding=3, stride=2, bias=False)
        nn.init.kaiming_normal_(resnet_backbone.conv1.weight, mode='fan_out', nonlinearity='relu')

    returned_layers = [int(5 - i) for i in range(returned_num_layers, 0, -1)]
    assert min(returned_layers) > 0 and max(returned_layers) < 5

    return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}

    in_channels_stage2 = resnet_backbone.in_channel // 8  # 256
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    out_channels = 256

    return R50_FFM(resnet_backbone, return_layers, in_channels_list, out_channels)
