import copy
from typing import Optional, List, Dict, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from . import det_utils
from . import boxes as box_ops


class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models

    Arguments:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super(TwoMLPHead, self).__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)
        for fc in [self.fc6, self.fc7]:
            nn.init.kaiming_uniform_(fc.weight, a=1)
            nn.init.constant_(fc.bias, 0)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Arguments:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)


    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    Computes the loss for Faster R-CNN.

    Arguments:
        class_logits : Predict category probability information, shape=[num_anchors, num_classes]
        box_regression : Prediction edge target frame regression information
        labels : Real class information
        regression_targets : Real target bounding box information

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    classification_loss = F.cross_entropy(class_logits, labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    # sampled_pos_inds_subset = torch.nonzero(torch.gt(labels, 0)).squeeze(1)
    sampled_pos_inds_subset = torch.where(torch.gt(labels, 0))[0]

    labels_pos = labels[sampled_pos_inds_subset]

    # shape=[num_proposal, num_classes]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, -1, 4)

    box_loss = det_utils.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1/9,
        size_average=False,
    ) / labels.numel()

    return classification_loss, box_loss


class RoIHeads(torch.nn.Module):
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
        'proposal_matcher': det_utils.Matcher,
        'fg_bg_sampler': det_utils.BalancedPositiveNegativeSampler,
    }

    def __init__(self,
                 box_roi_pool,   # Multi-scale RoIAlign pooling
                 box_head,       # TwoMLPHead
                 box_predictor,  # FastRCNNPredictor
                 # Faster R-CNN training
                 fg_iou_thresh, bg_iou_thresh,  # default: 0.5, 0.5
                 batch_size_per_image, positive_fraction,  # default: 512, 0.25
                 bbox_reg_weights,  # None
                 # Faster R-CNN inference
                 score_thresh,        # default: 0.05
                 nms_thresh,          # default: 0.5
                 detection_per_img,
                 cascade=False
                 ):
        super(RoIHeads, self).__init__()

        self.box_similarity = box_ops.box_iou
        # assign ground-truth boxes for each proposal
        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,  # default: 0.5
            bg_iou_thresh,  # default: 0.5
            allow_low_quality_matches=False)

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image,  # default: 512
            positive_fraction)     # default: 0.25

        if bbox_reg_weights is None:
            # bbox_reg_weights = (10., 10., 5., 5.)
            bbox_reg_weights = (0.1, 0.1, 0.2, 0.2)
        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)

        self.box_roi_pool = box_roi_pool    # Multi-scale RoIAlign pooling
        self.box_head = box_head            # TwoMLPHead
        self.box_predictor = box_predictor  # FastRCNNPredictor

        self.score_thresh = score_thresh  # default: 0.05
        self.nms_thresh = nms_thresh      # default: 0.5
        self.detection_per_img = detection_per_img  # default: 100
        # self.batch_iou = []
        # self.batch_t = 0
        self.cascade = cascade

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        # type: (List[Tensor], List[Tensor], List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        """
        Each proposal is matched with the corresponding gt_box and divided into positive and negative samples
        Args:
            proposals:
            gt_boxes:
            gt_labels:

        Returns:

        """
        matched_idxs = []
        labels = []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):
            if gt_boxes_in_image.numel() == 0:
                # background image
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
            else:
                #  set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
                match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)

                # iou < low_threshold: -1, low_threshold <= iou < high_threshold: -2
                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)
                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)

                # label background (below the low threshold)
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD  # -1
                labels_in_image[bg_inds] = 0

                # label ignore proposals (between low and high threshold)
                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS  # -2
                labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
        return matched_idxs, labels

    def subsample(self, labels):
        # type: (List[Tensor]) -> List[Tensor]
        # BalancedPositiveNegativeSampler
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(zip(sampled_pos_inds, sampled_neg_inds)):
            # img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            img_sampled_inds = torch.where(pos_inds_img | neg_inds_img)[0]
            sampled_inds.append(img_sampled_inds)
        return sampled_inds

    def subsample_with_pos(self, labels):
        # BalancedPositiveNegativeSampler
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        pos_ids = []
        for img_idx, pos_ind in enumerate(sampled_pos_inds):
            pos_id = torch.where(pos_ind)[0]
            pos_ids.append(pos_id)
        sampled_inds = []
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(zip(sampled_pos_inds, sampled_neg_inds)):
            # img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            img_sampled_inds = torch.where(pos_inds_img | neg_inds_img)[0]
            sampled_inds.append(img_sampled_inds)
        return sampled_inds, pos_ids

    def add_gt_proposals(self, proposals, gt_boxes):
        # type: # (List[Tensor], List[Tensor]) -> List[Tensor]
        """
        Concatenate gt_boxes to the end of the proposal
        Args:
            proposals: boxes of rpn predictions for each image in a batch
            gt_boxes: The actual target bounding box corresponding to each image in a batch
        Returns:
        """
        proposals_ = []
        gt_flags = []
        for proposal, gt_box in zip(proposals, gt_boxes):
            proposals_.append(torch.cat((proposal, gt_box)))
            gt_ones = gt_box.new_ones(gt_box.shape[0], dtype=torch.uint8)
            # gt_flags.append(torch.cat([gt_flag, gt_ones]))
            gt_flags.append(gt_ones)

        return proposals_, gt_flags

    def check_targets(self, targets):
        # type: (Optional[List[Dict[str, Tensor]]]) -> None
        assert targets is not None
        assert all(["boxes" in t for t in targets])
        assert all(["labels" in t for t in targets])

    def get_iou(self, proposals, pos_ids, matched_idxs, gt_boxes):
        matched_idxs_ = copy.deepcopy(matched_idxs)
        matched_boxes = []
        results = []
        # if self.batch_t % 750 == 0 or (self.batch_t-1) % 750 == 0:
        for img_id in range(len(proposals)):
            pos_id = pos_ids[img_id]
            proposals[img_id] = proposals[img_id][pos_id]
            matched_idxs_[img_id] = matched_idxs_[img_id][pos_id]

            gt_boxes_in_image = gt_boxes[img_id]
            matched_boxes.append(gt_boxes_in_image[matched_idxs_[img_id]])
            rs = []
            for i in range(len(proposals[img_id])):
                r = box_ops.box_iou(proposals[img_id][i].unsqueeze(0), matched_boxes[img_id][i].unsqueeze(0))
                # r = [i for i in r if i !=1]
                if r.item() != 1:
                    rs.append(r)
            if len(rs) > 0:
                rs_in_img = torch.cat(rs, dim=0).squeeze()
            else:
                rs_in_img = torch.zeros(3)
            results.append(torch.mean(rs_in_img, 0).item())
        # zz = torch.cat(results, dim=0).squeeze()
        # zz = sum(results)/len(results)
        # self.batch_iou.append(zz)
        # with open('/gemini/iou.txt', 'w') as f:
        #     f.write(str(self.batch_iou)+'\n')
        return

    def select_training_samples(self,
                                proposals,  # type: List[Tensor]
                                targets     # type: Optional[List[Dict[str, Tensor]]]
                                ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]
        """
        Divide positive and negative samples, and calculate the corresponding labels and bounding
        box regression information for gt
        Args:
            proposals: predicted boxes
            targets:

        Returns:

        """
        self.check_targets(targets)
        assert targets is not None

        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        # append ground-truth bboxes to proposal
        proposals, gt_flags = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)

        # sampled_inds, pos_ids = self.subsample_with_pos(labels)
        # self.get_iou(proposals.copy(), pos_ids, matched_idxs, gt_boxes)
        # self.batch_t+=1
        matched_gt_boxes = []
        num_images = len(proposals)

        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            # obtain gt box information corresponding to positive and negative samples
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

        # Calculate border regression parameters based on gt and proposal (for gt)
        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, labels, regression_targets, gt_flags

    def postprocess_detections(self,
                               class_logits,    # type: Tensor
                               box_regression,  # type: Tensor
                               proposals,       # type: List[Tensor]
                               image_shapes     # type: List[Tuple[int, int]]
                               ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        """
        Post-processing of network prediction data, including
        （1）Calculate the final bbox coordinates based on the proposal and predicted regression parameters
        （2）Softmax processing of predicted category results
        （3）Crop the predicted boxes information and adjust the out of bounds coordinates to the image boundary
        （4）Remove all background information
        （5）Remove low probability targets
        （6）Remove small-sized targets
        （7）Perform NMS processing and sort by scores
        （8）Returns the top k targets sorted by scores
        Args:
            class_logits: Information that predicts category probabilities
            box_regression: Predicted boundary box regression parameters
            proposals: Proposal output from rpn
            image_shapes: Original width and height of each image

        Returns:

        """
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)

        # split boxes and scores per image
        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []

        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove prediction with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            # gt: Computes input > other element-wise.
            # inds = torch.nonzero(torch.gt(scores, self.score_thresh)).squeeze(1)
            inds = torch.where(torch.gt(scores, self.score_thresh))[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1.)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximun suppression, independently done per class
            nms_cfg=dict(type='nms', iou_threshold=0.5, class_agnostic=True)
            keep = box_ops.batched_nms_(boxes, scores, labels, nms_cfg)
            # keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.detection_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels

    def bbox2roi(self, bbox_list) -> Tensor:
        rois_list = []
        for img_id, bboxes in enumerate(bbox_list):
            img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
            rois = torch.cat([img_inds, bboxes], dim=-1)
            rois_list.append(rois)
        rois = torch.cat(rois_list, 0)
        return rois

    def forward(self,
                features,       # type: Dict[str, Tensor]
                proposals,      # type: List[Tensor]
                image_shapes,   # type: List[Tuple[int, int]]
                targets=None    # type: Optional[List[Dict[str, Tensor]]]
                ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor], Dict[str, Tensor]]
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """

        if targets is not None:
            for t in targets:
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, "target boxes must of float type"
                assert t["labels"].dtype == torch.int64, "target labels must of int64 type"
        output = {}
        if self.training:
            proposals, labels, regression_targets, gt_flags = self.select_training_samples(proposals, targets)
            output['gt_flags'] = gt_flags
            output['labels'] = torch.cat(labels, dim=0)
        else:
            labels = None
            regression_targets = None

        # box_features_shape: [num_proposals, channel, height, width]
        box_features = self.box_roi_pool(features, proposals, image_shapes)

        # box_features_shape: [num_proposals, representation_size]
        box_features = self.box_head(box_features)

        class_logits, box_regression = self.box_predictor(box_features)

        result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
        losses = {}

        rois = self.bbox2roi(proposals)
        output['rois'] = rois
        output['cls_scores'] = class_logits
        output['box_regression'] = box_regression

        if self.training:
            assert labels is not None and regression_targets is not None
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets)
            losses = {
                "loss_clas": loss_classifier,
                "loss_bbox": loss_box_reg
            }
        else:
            if not self.cascade:
                boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
                num_images = len(boxes)
                for i in range(num_images):
                    result.append(
                        {
                            "boxes": boxes[i],
                            "labels": labels[i],
                            "scores": scores[i],
                        }
                    )
        return result, losses, output
