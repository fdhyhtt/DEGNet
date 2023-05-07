import torch
from torch import nn
from .roi_head import TwoMLPHead, FastRCNNPredictor
from .roi_head import RoIHeads

detail_cfg = [{'fg_iou_thresh': 0.5, 'bg_iou_thresh': 0.5, 'batch_size_per_image': 512,'positive_fraction': 0.25,
               'bbox_reg_weights': [0.1, 0.1, 0.2, 0.2], 'detection_per_img': 20, 'nms_thresh': 0.5,
               'score_thresh': 0.05, 'cascade':True},
              {'fg_iou_thresh': 0.6, 'bg_iou_thresh': 0.6, 'batch_size_per_image': 512, 'positive_fraction': 0.25,
               'bbox_reg_weights': [0.05, 0.05, 0.1, 0.1], 'detection_per_img': 20, 'nms_thresh': 0.5,
               'score_thresh': 0.05, 'cascade':True},
              {'fg_iou_thresh': 0.7, 'bg_iou_thresh': 0.7, 'batch_size_per_image': 512, 'positive_fraction': 0.25,
               'bbox_reg_weights': [0.033, 0.033, 0.067, 0.067], 'detection_per_img': 20, 'nms_thresh': 0.5,
               'score_thresh': 0.05, 'cascade':True}]


class CascadeHead(nn.Module):
    def __init__(self, num_cls, pooling_layer, stage_loss_weights, num_stages, fc_out_channels, channel_size, reg_class_agnostic):
        super(CascadeHead, self).__init__()
        self.stage_loss_weights = stage_loss_weights
        self.num_stages = num_stages  # 3
        self.fc_out_channels = fc_out_channels  # 1024
        self.detail_cfg = detail_cfg
        self.num_classes = num_cls
        self.reg_class_agnostic = reg_class_agnostic

        assert len(self.stage_loss_weights) == num_stages == len(detail_cfg)
        roi_resolution = pooling_layer[0].output_size[0]  # 7
        head_in_channel = channel_size * roi_resolution ** 2

        # roi_heads
        self.roi_heads = list()
        for i in range(num_stages):
            box_head = TwoMLPHead(head_in_channel, self.fc_out_channels)
            box_predictor = FastRCNNPredictor(self.fc_out_channels, num_cls)
            roi_head = RoIHeads(pooling_layer[i], box_head, box_predictor, **detail_cfg[i])
            self.roi_heads.append(roi_head)
        self.roi_heads = nn.ModuleList(self.roi_heads)

    def forward_(self, feature_dict, boxes, valid_size, targets):
        loss_sum = dict()
        all_cls = list()
        num_per_batch = [len(b) for b in boxes]
        for i in range(self.num_stages):
            boxes, cls, loss = self.roi_heads[i](feature_dict, boxes, valid_size, targets)
            if self.training:
                loss_sum['{:d}_cls_loss'.format(i)] = loss['roi_cls_loss'] * self.stage_loss_weights[i]
                loss_sum['{:d}_box_loss'.format(i)] = loss['roi_box_loss'] * self.stage_loss_weights[i]
            else:
                all_cls.append(torch.cat(cls, dim=0))
        if not self.training:
            all_cls = torch.stack(all_cls, dim=-1)
            if all_cls.dtype == torch.float16:
                all_cls = all_cls.float()
            all_cls = all_cls.softmax(dim=-2).mean(-1)
            all_cls = all_cls.split(num_per_batch)
        return boxes, all_cls, loss_sum

    def refine_bboxes(self, output, stage):
        labels = output['labels']
        cls_scores = output['cls_scores']
        rois = output['rois']
        box_regression = output['box_regression']
        gt_flags = output['gt_flags']

        if cls_scores.shape[-1] == self.num_classes:
            # remove background class
            cls_scores = cls_scores[:, 1:]
        else:
            raise ValueError('The last dim of `cls_scores` should equal to '
                             '`num_classes` or `num_classes + 1`,'
                             f'but got {cls_scores.shape[-1]}.')

        num = len(gt_flags)
        labels = torch.where(labels == 0, cls_scores.argmax(1), labels)

        results_list = []
        for i in range(num):
            inds = torch.nonzero(
                rois[:, 0] == i, as_tuple=False).squeeze(dim=1)
            num_rois = inds.numel()

            bboxes_ = rois[inds, 1:]  # proposal
            label_ = labels[inds]
            bbox_pred_ = box_regression[inds]
            pos_is_gts_ = gt_flags[i]

            bboxes = self.regress_by_class(bboxes_, label_, bbox_pred_, stage)
            # filter gt bboxes
            pos_keep = 1 - pos_is_gts_
            keep_inds = pos_is_gts_.new_ones(num_rois)
            keep_inds[-len(pos_keep):] = pos_keep
            results = bboxes[keep_inds.type(torch.bool)]
            results_list.append(results)

        return results_list

    def regress_by_class(self, priors, label, bbox_pred, stage):
        """Regress the bbox for the predicted class. Used in Cascade R-CNN.

        Args:
            priors (Tensor): Priors from `rpn_head` or last stage
                `bbox_head`, has shape (num_proposals, 4).
            label (Tensor): Only used when `self.reg_class_agnostic`
                is False, has shape (num_proposals, ).
            bbox_pred (Tensor): Regression prediction of
                current stage `bbox_head`. When `self.reg_class_agnostic`
                is False, it has shape (n, num_classes * 4), otherwise
                it has shape (n, 4).

        Returns:
            Tensor: Regressed bboxes, the same shape as input rois.
        """
        reg_dim = 4
        if not self.reg_class_agnostic:
            label = label * reg_dim
            inds = torch.stack([label + i for i in range(reg_dim)], 1)
            bbox_pred = torch.gather(bbox_pred, 1, inds)
        assert bbox_pred.size()[1] == reg_dim
        regressed_bboxes = self.roi_heads[stage].box_coder.refine_decode(
            pred_bboxes=bbox_pred, bboxes=priors)
        return regressed_bboxes

    def forward(self,
                features,
                proposals,
                image_shapes,
                targets=None,
                ):
        losses = {}
        ms_scores = []
        output = {}

        for i in range(self.num_stages):
            stage_loss_weight = self.stage_loss_weights[i]
            result, roi_losses, output = self.roi_heads[i](features, proposals, image_shapes, targets)
            if self.training:
                for name, value in roi_losses.items():
                    losses[f's{i}.{name}'] = (
                        value * stage_loss_weight)
                if i < self.num_stages - 1:
                    with torch.no_grad():
                        proposals = self.refine_bboxes(output, stage=i)
            else:
                num_proposals_per_img = tuple(len(p) for p in proposals)
                cls_scores = output['cls_scores']
                bbox_preds = output['box_regression']
                rois = output['rois']
                rois = rois.split(num_proposals_per_img, 0)
                cls_scores = cls_scores.split(num_proposals_per_img, 0)
                ms_scores.append(cls_scores)
                bbox_preds = bbox_preds.split(num_proposals_per_img, 0)
                if i < self.num_stages - 1:
                    refine_bboxes_list = []
                    for j in range(len(num_proposals_per_img)):
                        if rois[j].shape[0] > 0:
                            bbox_label = cls_scores[j][:, 1:].argmax(dim=1)
                            refined_bboxes = self.regress_by_class(
                                rois[j][:, 1:], bbox_label, bbox_preds[j],
                                stage=i)
                            refine_bboxes_list.append(refined_bboxes)
                    proposals = refine_bboxes_list
                else:
                    self.num_batch = len(num_proposals_per_img)
                    self.bbox_preds = bbox_preds

        if not self.training:
            result = []
            cls_scores = [
                sum([score[i] for score in ms_scores]) / float(len(ms_scores))
                for i in range(self.num_batch)
            ]
            cls_scores = torch.cat(cls_scores)
            bbox_preds = torch.cat(self.bbox_preds)
            boxes, scores, labels = self.roi_heads[-1].postprocess_detections(cls_scores, bbox_preds, proposals, image_shapes)
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

