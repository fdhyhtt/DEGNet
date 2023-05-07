import copy
import warnings
from collections import OrderedDict
from typing import Tuple, List, Dict, Optional, Union
from .domain_adapter import DAdapter
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.ops import MultiScaleRoIAlign
from .cascade_head import CascadeHead
from .roi_head import RoIHeads, TwoMLPHead, FastRCNNPredictor
from .transform import GeneralizedRCNNTransform
from .rpn_function import AnchorsGenerator, RPNHead, RegionProposalNetwork
import numpy as np


def init_weights(net, init_type='normal', gain=0.02):
    from torch.nn import init
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network eem with %s' % init_type)
    net.apply(init_func)


class Enhancer:
    def __init__(self, enhance_factor, threshold):
        self.enhance_factor = enhance_factor
        self.threshold = threshold

    def __call__(self, edge_detect):
        # obtain weight matrix
        weight = edge_detect / (edge_detect + self.threshold)
        weight = weight * (self.enhance_factor - 1) + 1
        # enhancing
        edge_detect *= weight
        edge_detect[:, 0, :, :] = torch.clamp(edge_detect[:, 0, :, :], 0, 1)
        return edge_detect


class EdgeEmbedding(nn.Module):
    def __init__(self, enhence_feature=True):
        super(EdgeEmbedding, self).__init__()
        kernel_const_hori = np.array([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype='float32')
        kernel_const_hori = torch.cuda.FloatTensor(kernel_const_hori).unsqueeze(0)
        self.weight_const_hori = nn.Parameter(data=kernel_const_hori, requires_grad=False)

        self.weight_hori = Variable(self.weight_const_hori)
        self.gamma = nn.Parameter(torch.zeros(1))

        kernel_const_vertical = np.array([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype='float32')
        kernel_const_vertical = torch.cuda.FloatTensor(kernel_const_vertical).unsqueeze(0)
        self.weight_const_vertical = nn.Parameter(data=kernel_const_vertical, requires_grad=False)

        self.weight_vertical = Variable(self.weight_const_vertical)

        self.pel_enhancer = Enhancer(enhance_factor=2.0, threshold=0.01)  # 1.6
        self.enhence_feature = enhence_feature

        att = []
        in_channels = 4
        for i in [8, 16, 8, 4]:
            att.append(
                nn.Conv2d(
                    in_channels,
                    i,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            in_channels = i
            # dis_tower.append(nn.GroupNorm(32, in_channels))
            att.append(nn.BatchNorm2d(i))
            att.append(nn.ReLU())

        self.add_module('att', nn.Sequential(*att))

        init_weights(self)

    def MaxMinNormalization(self, x):
        """[0,1] normaliaztion"""
        x = (x - x.min()) / (x.max() - x.min()) *255
        return x

    def tensor_erode(self, bin_img, ksize=3):
        # padding
        B, C, H, W = bin_img.shape
        pad = (ksize - 1) // 2
        bin_img = F.pad(bin_img, [pad, pad, pad, pad], mode='constant', value=0)
        patches = bin_img.unfold(dimension=2, size=ksize, step=1)
        patches = patches.unfold(dimension=3, size=ksize, step=1)
        # B x C x H x W x k x k
        eroded, _ = patches.reshape(B, C, H, W, -1).min(dim=-1)
        return eroded

    def forward(self, im):
        # x3 = Variable(im[:, 2].unsqueeze(1))
        x3 = Variable(torch.mean(im, dim=1, keepdim=True))
        weight_hori = self.weight_hori
        weight_vertical = self.weight_vertical

        x_hori = F.conv2d(x3, weight_hori, padding=1)
        x_vertical = F.conv2d(x3, weight_vertical, padding=1)

        #get edge image
        edge_detect = (torch.add(x_hori.pow(2),x_vertical.pow(2))).pow(0.5)

        #normalization of edge image
        edge_detect = ((edge_detect - (edge_detect.min())) / ((edge_detect.max()) - (edge_detect.min())))

        # edge_detect = self.tensor_erode(edge_detect)
        if self.enhence_feature:
            edge_enhenced = self.pel_enhancer(edge_detect)

            return torch.cat((im, edge_enhenced * 255), 1)

        out = torch.cat((im, edge_detect * 255), 1)

        out = self.att(out)

        return out


class DecoupledFeatureExtractor(nn.Module):
    def __init__(self, backbone, eg_backbone):
        super(DecoupledFeatureExtractor, self).__init__()
        self.backbone = backbone
        self.edge_layer = EdgeEmbedding() if eg_backbone else None
        self.eg_backbone = eg_backbone

    def forward(self, images):
        features = self.backbone(images)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        # Edge embedding
        if self.eg_backbone:
            edge_embedding_img = self.edge_layer(images)
            edge_feature = self.eg_backbone(edge_embedding_img)
            features.update(ef=edge_feature)
        return features


class FasterRCNNBase(nn.Module):
    """
    Main class for Generalized R-CNN.

    Arguments:
        backbone (nn.Module):
        rpn (nn.Module):
        roi_heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self, backbone, rpn, roi_heads, transform, eg_backbone,):
        super(FasterRCNNBase, self).__init__()
        self.transform = transform
        self.extractor = DecoupledFeatureExtractor(backbone, eg_backbone)
        self.rpn = rpn
        self.roi_heads = roi_heads
        self.da_adapter = DAdapter() if self.cfg['Model']['model'].get('da') else None
        # used only on torchscript mode
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                                          boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))

        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))
        original_image_sizes = [img.shape[-2:] for img in images]

        # preprocessing images
        images, targets = self.transform(images, targets)

        features = self.extractor(images.tensors)

        losses = {}
        if self.training and self.da_adapter:
            da_loss = self.da_adapter(features['0'])
            losses.update(da_loss)

        # proposals: List[Tensor], Tensor_shape: [num_proposals, 4],
        proposals, proposal_losses = self.rpn(images, features, targets)

        # pass the data generated by rpn and annotated target information into rcnn
        detections, detector_losses, _ = self.roi_heads(features, proposals, images.image_sizes, targets)

        # post processing the predicted results (Restore bboxes to the original image scale)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses.update(proposal_losses)
        losses.update(detector_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)


class FasterRCNN(FasterRCNNBase):
    """
    Implements Faster R-CNN.

    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values
          between 0 and H and 0 and W
        - labels (Int64Tensor[N]): the class label for each ground-truth box

    The model returns a Dict[Tensor] during training, containing the classification and regression
    losses for both the RPN and the R-CNN.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values between
          0 and H and 0 and W
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores or each prediction

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            It should contain a out_channels attribute, which indicates the number of output
            channels that each feature map has (and it should be the same for all feature maps).
            The backbone should return a single Tensor or and OrderedDict[Tensor].
        num_classes (int): number of output classes of the model (including the background).
            If box_predictor is specified, num_classes should be None.
        min_size (int): minimum size of the image to be rescaled before feeding it to the backbone
        max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        image_mean (Tuple[float, float, float]): mean values used for input normalization.
            They are generally the mean values of the dataset on which the backbone has been trained
            on
        image_std (Tuple[float, float, float]): std values used for input normalization.
            They are generally the std values of the dataset on which the backbone has been trained on
        rpn_anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        rpn_head (nn.Module): module that computes the objectness and regression deltas from the RPN
        rpn_pre_nms_top_n_train (int): number of proposals to keep before applying NMS during training
        rpn_pre_nms_top_n_test (int): number of proposals to keep before applying NMS during testing
        rpn_post_nms_top_n_train (int): number of proposals to keep after applying NMS during training
        rpn_post_nms_top_n_test (int): number of proposals to keep after applying NMS during testing
        rpn_nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
        rpn_fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        rpn_bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        rpn_batch_size_per_image (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        rpn_positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN
        rpn_score_thresh (float): during inference, only return proposals with a classification score
            greater than rpn_score_thresh
        box_roi_pool (MultiScaleRoIAlign): the module which crops and resizes the feature maps in
            the locations indicated by the bounding boxes
        box_head (nn.Module): module that takes the cropped feature maps as input
        box_predictor (nn.Module): module that takes the output of box_head and returns the
            classification logits and box regression deltas.
        box_score_thresh (float): during inference, only return proposals with a classification score
            greater than box_score_thresh
        box_nms_thresh (float): NMS threshold for the prediction head. Used during inference
        box_detections_per_img (int): maximum number of detections per image, for all classes.
        box_fg_iou_thresh (float): minimum IoU between the proposals and the GT box so that they can be
            considered as positive during training of the classification head
        box_bg_iou_thresh (float): maximum IoU between the proposals and the GT box so that they can be
            considered as negative during training of the classification head
        box_batch_size_per_image (int): number of proposals that are sampled during training of the
            classification head
        box_positive_fraction (float): proportion of positive proposals in a mini-batch during training
            of the classification head
        bbox_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes

    """

    def __init__(self, backbone, eg_backbone=None, num_classes=None,
                 # transform parameter
                 min_size=800,
                 max_size=1333,      # minimum and maximum size during preprocessing resize
                 image_mean=None, image_std=None,  # The mean and variance used in preprocessing normalize
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,    # number of proposals before NMS
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,  # number of proposals after NMS
                 rpn_nms_thresh=0.7,  # Iou threshold used for nms processing in rpn
                 # threshold for collecting positive and negative samples in RPN
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 # Number of samples and positive samples proportion in RPN
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 rpn_score_thresh=0.0,
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 # Remove low target probability, threshold for nms processing in rcnn, how many results are taken
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=20,
                 # The threshold for collecting positive and negative samples when rcnn calculates loss
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 # Number of samples and positive samples proportion in rcnn
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None, cfg=None):

        assert isinstance(rpn_anchor_generator, (AnchorsGenerator, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))
        self.cfg = cfg
        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor "
                                 "is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor "
                                 "is not specified")

        # The channel of the predict feature layer
        out_channels = 256

        if rpn_anchor_generator is None:
            anchor_sizes = cfg['Model']['model']['anchor_sizes']
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorsGenerator(
                anchor_sizes, aspect_ratios)

        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
            )

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        # Define the RPN framework
        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh,
            score_thresh=rpn_score_thresh,
            pds=cfg['Model']['model']['pds'],
        )

        #  Multi-scale RoIAlign pooling
        if cfg['Model']['Backbone'].get('backbone1') == 'fpn':
            featmap_names = ['0', '1', '2', '3']
        else:
            featmap_names = ['0']
        if box_roi_pool is None:
            if cfg['Model'].get('Roi') == 'cascade':
                box_roi_pool = nn.ModuleList()
                for i in range(3):
                    box_roi_pool.append(MultiScaleRoIAlign(
                        featmap_names=featmap_names,
                        output_size=[7, 7],
                        sampling_ratio=2))
            else:
                box_roi_pool = MultiScaleRoIAlign(
                    featmap_names=featmap_names,
                    output_size=[7, 7],
                    sampling_ratio=2)

        # Two parts of full connection layer after roi pooling in RCNN
        if box_head is None:
            # 默认等于7
            resolution = 7
            representation_size = 1024
            box_head = TwoMLPHead(
                out_channels * resolution ** 2,
                representation_size
            )

        # box_predictor
        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(
                representation_size,
                num_classes)

        # Define the RCNN framework
        if cfg['Model'].get('Roi') == 'cascade':
            roi_heads = CascadeHead(num_cls=num_classes,
                                    pooling_layer=box_roi_pool,
                                    stage_loss_weights=[1, 0.5, 0.25],
                                    num_stages=3,
                                    fc_out_channels=1024,
                                    channel_size=out_channels,
                                    reg_class_agnostic=False)
        else:
            roi_heads = RoIHeads(
                # box
                box_roi_pool, box_head, box_predictor,
                box_fg_iou_thresh, box_bg_iou_thresh,  # 0.5  0.5
                box_batch_size_per_image, box_positive_fraction,  # 512  0.25
                bbox_reg_weights,
                box_score_thresh, box_nms_thresh, box_detections_per_img,)  # 0.05  0.5  100



        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]

        # Standardize, scale, package into batch and other processing parts
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        super(FasterRCNN, self).__init__(backbone, rpn, roi_heads, transform, eg_backbone=eg_backbone)
