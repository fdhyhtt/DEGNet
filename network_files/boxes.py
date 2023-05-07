import numpy as np
import torch
from typing import Tuple
from torch import Tensor
import torchvision


def nms(boxes, scores, iou_threshold):
    # type: (Tensor, Tensor, float) -> Tensor
    """
    Performs non-maximum suppression (NMS) on the boxes according
    to their intersection-over-union (IoU).

    NMS iteratively removes lower scoring boxes which have an
    IoU greater than iou_threshold with another (higher scoring)
    box.

    Parameters
    ----------
    boxes : Tensor[N, 4])
        boxes to perform NMS on. They
        are expected to be in (x1, y1, x2, y2) format
    scores : Tensor[N]
        scores for each one of the boxes
    iou_threshold : float
        discards all overlapping
        boxes with IoU > iou_threshold

    Returns
    -------
    keep : Tensor
        int64 tensor with the indices
        of the elements that have been kept
        by NMS, sorted in decreasing order of scores
    """
    return torch.ops.torchvision.nms(boxes, scores, iou_threshold)

def batched_nms_(boxes, scores, idxs, nms_cfg, class_agnostic=False):
    """Performs non-maximum suppression in a batched fashion.

    Modified from https://github.com/pytorch/vision/blob
    /505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39.
    In order to perform NMS independently per class, we add an offset to all
    the boxes. The offset is dependent only on the class idx, and is large
    enough so that boxes from different classes do not overlap.

    Arguments:
        boxes (torch.Tensor): boxes in shape (N, 4).
        scores (torch.Tensor): scores in shape (N, ).
        idxs (torch.Tensor): each index value correspond to a bbox cluster,
            and NMS will not be applied between elements of different idxs,
            shape (N, ).
        nms_cfg (dict): specify nms type and other parameters like iou_thr.
            Possible keys includes the following.

            - iou_thr (float): IoU threshold used for NMS.
            - split_thr (float): threshold number of boxes. In some cases the
                number of boxes is large (e.g., 200k). To avoid OOM during
                training, the users could set `split_thr` to a small value.
                If the number of boxes is greater than the threshold, it will
                perform NMS on each group of boxes separately and sequentially.
                Defaults to 10000.
        class_agnostic (bool): if true, nms is class agnostic,
            i.e. IoU thresholding happens over all boxes,
            regardless of the predicted class.

    Returns:
        tuple: kept dets and indice.
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    nms_cfg_ = nms_cfg.copy()
    class_agnostic = nms_cfg_.pop('class_agnostic', class_agnostic)
    if class_agnostic:
        boxes_for_nms = boxes
    else:
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
        boxes_for_nms = boxes + offsets[:, None]

    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = eval(nms_type)

    split_thr = nms_cfg_.pop('split_thr', 10000)
    # Won't split to multiple nms nodes when exporting to onnx
    if boxes_for_nms.shape[0] < split_thr or torch.onnx.is_in_onnx_export():
        keep = nms_op(boxes_for_nms, scores, **nms_cfg_)
        # dets, keep = nms_op(boxes_for_nms, scores, **nms_cfg_)
        boxes = boxes[keep]
        # -1 indexing works abnormal in TensorRT
        # This assumes `dets` has 5 dimensions where
        # the last dimension is score.
        # TODO: more elegant way to handle the dimension issue.
        # Some type of nms would reweight the score, such as SoftNMS
        # scores = dets[:, 4]
    else:
        max_num = nms_cfg_.pop('max_num', -1)
        total_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
        # Some type of nms would reweight the score, such as SoftNMS
        scores_after_nms = scores.new_zeros(scores.size())
        for id in torch.unique(idxs):
            mask = (idxs == id).nonzero(as_tuple=False).view(-1)
            dets, keep = nms_op(boxes_for_nms[mask], scores[mask], **nms_cfg_)
            total_mask[mask[keep]] = True
            scores_after_nms[mask[keep]] = dets[:, -1]
        keep = total_mask.nonzero(as_tuple=False).view(-1)

        scores, inds = scores_after_nms[keep].sort(descending=True)
        keep = keep[inds]
        boxes = boxes[keep]

        if max_num > 0:
            keep = keep[:max_num]
            boxes = boxes[:max_num]
            scores = scores[:max_num]

    return keep

# @deprecated_api_warning({'iou_thr': 'iou_threshold'})
# def nms_(boxes, scores, iou_threshold, offset=0, score_threshold=0, max_num=-1):
#     """Dispatch to either CPU or GPU NMS implementations.
#
#     The input can be either torch tensor or numpy array. GPU NMS will be used
#     if the input is gpu tensor, otherwise CPU NMS
#     will be used. The returned type will always be the same as inputs.
#
#     Arguments:
#         boxes (torch.Tensor or np.ndarray): boxes in shape (N, 4).
#         scores (torch.Tensor or np.ndarray): scores in shape (N, ).
#         iou_threshold (float): IoU threshold for NMS.
#         offset (int, 0 or 1): boxes' width or height is (x2 - x1 + offset).
#         score_threshold (float): score threshold for NMS.
#         max_num (int): maximum number of boxes after NMS.
#
#     Returns:
#         tuple: kept dets(boxes and scores) and indice, which is always the \
#             same data type as the input.
#
#     Example:
#         >>> boxes = np.array([[49.1, 32.4, 51.0, 35.9],
#         >>>                   [49.3, 32.9, 51.0, 35.3],
#         >>>                   [49.2, 31.8, 51.0, 35.4],
#         >>>                   [35.1, 11.5, 39.1, 15.7],
#         >>>                   [35.6, 11.8, 39.3, 14.2],
#         >>>                   [35.3, 11.5, 39.9, 14.5],
#         >>>                   [35.2, 11.7, 39.7, 15.7]], dtype=np.float32)
#         >>> scores = np.array([0.9, 0.9, 0.5, 0.5, 0.5, 0.4, 0.3],\
#                dtype=np.float32)
#         >>> iou_threshold = 0.6
#         >>> dets, inds = nms(boxes, scores, iou_threshold)
#         >>> assert len(inds) == len(dets) == 3
#     """
#     assert isinstance(boxes, (torch.Tensor, np.ndarray))
#     assert isinstance(scores, (torch.Tensor, np.ndarray))
#     is_numpy = False
#     if isinstance(boxes, np.ndarray):
#         is_numpy = True
#         boxes = torch.from_numpy(boxes)
#     if isinstance(scores, np.ndarray):
#         scores = torch.from_numpy(scores)
#     assert boxes.size(1) == 4
#     assert boxes.size(0) == scores.size(0)
#     assert offset in (0, 1)
#
#     if torch.__version__ == 'parrots':
#         indata_list = [boxes, scores]
#         indata_dict = {
#             'iou_threshold': float(iou_threshold),
#             'offset': int(offset)
#         }
#         inds = ext_module.nms(*indata_list, **indata_dict)
#     else:
#         inds = NMSop.apply(boxes, scores, iou_threshold, offset,
#                            score_threshold, max_num)
#     dets = torch.cat((boxes[inds], scores[inds].reshape(-1, 1)), dim=1)
#     if is_numpy:
#         dets = dets.cpu().numpy()
#         inds = inds.cpu().numpy()
#     return dets, inds

def batched_nms(boxes, scores, idxs, iou_threshold):
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    """
    Performs non-maximum suppression in a batched fashion.

    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.

    Parameters
    ----------
    boxes : Tensor[N, 4]
        boxes where NMS will be performed. They
        are expected to be in (x1, y1, x2, y2) format
    scores : Tensor[N]
        scores for each one of the boxes
    idxs : Tensor[N]
        indices of the categories for each one of the boxes.
    iou_threshold : float
        discards all overlapping boxes
        with IoU < iou_threshold

    Returns
    -------
    keep : Tensor
        int64 tensor with the indices of
        the elements that have been kept by NMS, sorted
        in decreasing order of scores
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    max_coordinate = boxes.max()

    # to(): Performs Tensor dtype and/or device conversion
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep


def remove_small_boxes(boxes, min_size):
    # type: (Tensor, float) -> Tensor
    """
    Remove boxes which contains at least one side smaller than min_size.
    Arguments:
        boxes (Tensor[N, 4]): boxes in (x1, y1, x2, y2) format
        min_size (float): minimum size

    Returns:
        keep (Tensor[K]): indices of the boxes that have both sides
            larger than min_size
    """
    ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    # keep = (ws >= min_size) & (hs >= min_size)
    keep = torch.logical_and(torch.ge(ws, min_size), torch.ge(hs, min_size))
    # nonzero(): Returns a tensor containing the indices of all non-zero elements of input
    # keep = keep.nonzero().squeeze(1)
    keep = torch.where(keep)[0]
    return keep


def clip_boxes_to_image(boxes, size):
    # type: (Tensor, Tuple[int, int]) -> Tensor
    """
    Clip boxes so that they lie inside an image of size `size`.

    Arguments:
        boxes (Tensor[N, 4]): boxes in (x1, y1, x2, y2) format
        size (Tuple[height, width]): size of the image

    Returns:
        clipped_boxes (Tensor[N, 4])
    """
    dim = boxes.dim()
    # xx = boxes[:, 0::2]
    boxes_x = boxes[..., 0::2]  # x1, x2
    boxes_y = boxes[..., 1::2]  # y1, y2
    height, width = size

    if torchvision._is_tracing():
        boxes_x = torch.max(boxes_x, torch.tensor(0, dtype=boxes.dtype, device=boxes.device))
        boxes_x = torch.min(boxes_x, torch.tensor(width, dtype=boxes.dtype, device=boxes.device))
        boxes_y = torch.max(boxes_y, torch.tensor(0, dtype=boxes.dtype, device=boxes.device))
        boxes_y = torch.min(boxes_y, torch.tensor(height, dtype=boxes.dtype, device=boxes.device))
    else:
        boxes_x = boxes_x.clamp(min=0, max=width)
        boxes_y = boxes_y.clamp(min=0, max=height)

    clipped_boxes = torch.stack((boxes_x, boxes_y), dim=dim)
    return clipped_boxes.reshape(boxes.shape)


def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.

    Arguments:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format

    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1, boxes2):
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    #  When the shapes do not match,
    #  the shape of the returned output tensor follows the broadcasting rules
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # left-top [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # right-bottom [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou

