import tensorflow as tf

from ..builder import BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from .assign_result import AssignResult
from .base_assigner import BaseAssigner
from awsdet.models.utils.misc import trim_zeros

# TODO Add valid flags

@BBOX_ASSIGNERS.register_module()
class MaxIoUAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.
    Each proposals will be assigned with `-1`, or a semi-positive integer
    indicating the ground truth index.
    - -1: negative sample, no assigned gt
    - semi-positive integer: positive sample, index (0-based) of assigned gt
    Args:
        pos_iou_thr (float): IoU threshold for positive bboxes.
        neg_iou_thr (float or tuple): IoU threshold for negative bboxes.
        min_pos_iou (float): Minimum iou for a bbox to be considered as a
            positive bbox. Positive samples can have smaller IoU than
            pos_iou_thr due to the 4th step (assign max IoU sample to each gt).
        gt_max_assign_all (bool): Whether to assign all bboxes with the same
            highest overlap with some gt to that gt.
        ignore_iof_thr (float): IoF threshold for ignoring bboxes (if
            `gt_bboxes_ignore` is specified). Negative values mean not
            ignoring any bboxes.
        ignore_wrt_candidates (bool): Whether to compute the iof between
            `bboxes` and `gt_bboxes_ignore`, or the contrary.
        match_low_quality (bool): Whether to allow low quality matches. This is
            usually allowed for RPN and single stage detectors, but not allowed
            in the second stage. Details are demonetrated in Step 4.
        gpu_assign_thr (int): The upper bound of the number of GT for GPU
            assign. When the number of gt is above this threshold, will assign
            on CPU device. Negative values mean not assign on CPU.
    """
    def __init__(self,
                 pos_iou_thr,
                 neg_iou_thr,
                 min_pos_iou=.0,
                 gt_max_assign_all=True,
                 ignore_iof_thr=-1,
                 ignore_wrt_candidates=True,
                 match_low_quality=True,
                 gpu_assign_thr=-1,
                 iou_calculator=dict(type='BboxOverlaps2D')):
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.gt_max_assign_all = gt_max_assign_all
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates
        self.gpu_assign_thr = gpu_assign_thr
        self.match_low_quality = match_low_quality
        self.iou_calculator = build_iou_calculator(iou_calculator)
        
    def assign(self, bboxes, gt_bboxes, valid_flags=None, gt_bboxes_ignore=None, gt_labels=None):
        """Assign gt to bboxes.
        This method assign a gt bbox to every bbox (proposal/anchor), each bbox
        will be assigned with -1, or a semi-positive number. -1 means negative
        sample, semi-positive number is the index (0-based) of assigned gt.
        The assignment is done in following steps, the order matters.
        1. assign every bbox to the background
        2. assign proposals whose iou with all gts < neg_iou_thr to 0
        3. for each bbox, if the iou with its nearest gt >= pos_iou_thr,
           assign it to that bbox
        4. for each gt bbox, assign its nearest proposals (may be more than
           one) to itself
        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).
        Returns:
            :obj:`AssignResult`: The assign result.
        Example:
            >>> self = MaxIoUAssigner(0.5, 0.5)
            >>> bboxes = torch.Tensor([[0, 0, 10, 10], [10, 10, 20, 20]])
            >>> gt_bboxes = torch.Tensor([[0, 0, 10, 9]])
            >>> assign_result = self.assign(bboxes, gt_bboxes)
            >>> expected_gt_inds = torch.LongTensor([1, 0])
            >>> assert torch.all(assign_result.gt_inds == expected_gt_inds)
        """
        if isinstance(bboxes, list):
            bboxes = tf.concat(bboxes, axis=0)
        if isinstance(valid_flags, list):
            valid_flags = tf.concat(valid_flags, axis=0)
        gt_bboxes, valid_gts = trim_zeros(gt_bboxes)
        if gt_labels!=None:
            gt_labels = tf.boolean_mask(gt_labels, valid_gts)
        
        overlaps = self.iou_calculator(gt_bboxes, bboxes)
        
        assign_result = self.assign_wrt_overlaps(overlaps, valid_flags=valid_flags,
                                                 gt_labels=gt_labels)
        return assign_result
    
    def assign_wrt_overlaps(self, overlaps, valid_flags=None, gt_labels=None):
        num_gts = tf.shape(overlaps)[0]
        num_bboxes = tf.shape(overlaps)[1]
        assigned_gt_inds = tf.ones(num_bboxes, dtype=tf.int32) * -1
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = tf.zeros(num_bboxes)
            if num_gts == 0:
                # No truth, assign everything to background
                assigned_gt_inds = tf.zeros(num_bboxes, dtype=tf.int32)
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = tf.ones(num_bboxes, dtype=tf.int32) * -1
            return (num_gts,
                assigned_gt_inds,
                max_overlaps,
                assigned_labels)
        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        max_overlaps = tf.math.reduce_max(overlaps, axis=0)
        argmax_overlaps = tf.math.argmax(overlaps, axis=0)
        # for each gt, which anchor best overlaps with it
        # for each gt, the max iou of all proposals
        gt_max_overlaps = tf.math.reduce_max(overlaps, axis=1)
        gt_argmax_overlaps = tf.math.argmax(overlaps, axis=1)
        
        # 2. assign negative: below
        # the negative inds are set to be 0
        # leave invalid regions as -1
        # if no valid_flags passed, tag everything as valid
        if valid_flags==None:
            valid_flags = tf.cast(tf.ones(num_bboxes), tf.bool)
        if isinstance(self.neg_iou_thr, float):
            negative_indices = tf.cast(tf.where((max_overlaps>=0) & \
                                                (max_overlaps<self.neg_iou_thr) & \
                                                (valid_flags)), tf.int32)
            updates = tf.squeeze(tf.zeros_like(negative_indices))
            # to deal with the case of getting a scalar
            if tf.size(updates)==1:
                updates = tf.expand_dims(updates, axis=0)
            assigned_gt_inds = tf.tensor_scatter_nd_update(assigned_gt_inds, 
                                                           negative_indices,
                                                           updates)
        elif isinstance(self.neg_iou_thr, tuple):
            assert len(self.neg_iou_thr) == 2
            negative_indices = tf.cast(tf.where((max_overlaps>=self.neg_iou_thr[0]) & \
                                                (max_overlaps<self.neg_iou_thr[1]) & \
                                                (valid_flags)), tf.int32)
            updates = tf.squeeze(tf.zeros_like(negative_indices))
            if tf.size(updates)==1:
                updates = tf.expand_dims(updates, axis=0)
            assigned_gt_inds = tf.tensor_scatter_nd_update(assigned_gt_inds, 
                                                           negative_indices,
                                                           updates)
        # 3. assign positive: above positive IoU threshold
        positive_indices = tf.cast(tf.where(max_overlaps>=self.pos_iou_thr), tf.int32)
        # assign argmax gt category as update
        updates = tf.squeeze(tf.cast(tf.gather(argmax_overlaps, positive_indices) + 1, tf.int32))
        if tf.size(updates)==1:
            updates = tf.expand_dims(updates, axis=0)
        assigned_gt_inds = tf.tensor_scatter_nd_update(assigned_gt_inds, 
                                                       positive_indices,
                                                       updates)
        if self.match_low_quality:
            gt_thresholds = tf.cast(tf.where(gt_max_overlaps>self.min_pos_iou), tf.int32)
            low_quality_anchors = tf.cast(tf.gather(gt_argmax_overlaps, gt_thresholds), tf.int32)
            updates = tf.squeeze(gt_thresholds)
            if tf.size(updates)==1:
                updates = tf.expand_dims(updates, axis=0)
            assigned_gt_inds = tf.tensor_scatter_nd_update(assigned_gt_inds, 
                                                           low_quality_anchors,
                                                           updates+1)
        if gt_labels is not None:
            assigned_labels = tf.ones(num_bboxes, dtype=tf.int32) * -1
            pos_anchor_loc = tf.cast(tf.where(assigned_gt_inds > 0), tf.int32)
            pos_gt_loc = tf.cast(tf.gather(assigned_gt_inds, tf.where(assigned_gt_inds>0)), tf.int32) - 1
            updates = tf.squeeze(tf.cast(tf.gather(gt_labels, pos_gt_loc), tf.int32))
            if tf.size(updates)==1:
                updates = tf.expand_dims(updates, axis=0)
            if tf.size(pos_anchor_loc) > 0:
                assigned_labels = tf.tensor_scatter_nd_update(assigned_labels,
                                                              pos_anchor_loc,
                                                              updates)
        else:
            assigned_labels = None
        return num_gts, assigned_gt_inds, max_overlaps, assigned_labels