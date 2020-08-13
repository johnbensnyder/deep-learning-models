from abc import ABCMeta, abstractmethod

import tensorflow as tf
from awsdet.models.utils.misc import trim_zeros

from .sampling_result import SamplingResult
from ..assigners.assign_result import AssignResult


class BaseSampler(metaclass=ABCMeta):
    """Base class of samplers."""

    def __init__(self,
                 num,
                 pos_fraction,
                 neg_pos_ub=-1,
                 add_gt_as_proposals=False,
                 **kwargs):
        self.num = num
        self.pos_fraction = pos_fraction
        self.neg_pos_ub = neg_pos_ub
        self.add_gt_as_proposals = add_gt_as_proposals
        self.pos_sampler = self
        self.neg_sampler = self

    @abstractmethod
    def _sample_pos(self, assign_result, num_expected, **kwargs):
        """Sample positive samples."""
        pass

    @abstractmethod
    def _sample_neg(self, assign_result, num_expected, **kwargs):
        """Sample negative samples."""
        pass

    def sample(self,
               num_gts,
               gt_inds,
               max_overlaps,
               labels,
               bboxes,
               gt_bboxes,
               gt_labels=None,
               **kwargs):
        """Sample positive and negative bboxes.
        This is a simple implementation of bbox sampling given candidates,
        assigning results and ground truth bboxes.
        Args:
            assign_result (:obj:`AssignResult`): Bbox assigning results.
            bboxes (Tensor): Boxes to be sampled from.
            gt_bboxes (Tensor): Ground truth bboxes.
            gt_labels (Tensor, optional): Class labels of ground truth bboxes.
        Returns:
            :obj:`SamplingResult`: Sampling result.
        Example:
            >>> from mmdet.core.bbox import RandomSampler
            >>> from mmdet.core.bbox import AssignResult
            >>> from mmdet.core.bbox.demodata import ensure_rng, random_boxes
            >>> rng = ensure_rng(None)
            >>> assign_result = AssignResult.random(rng=rng)
            >>> bboxes = random_boxes(assign_result.num_preds, rng=rng)
            >>> gt_bboxes = random_boxes(assign_result.num_gts, rng=rng)
            >>> gt_labels = None
            >>> self = RandomSampler(num=32, pos_fraction=0.5, neg_pos_ub=-1,
            >>>                      add_gt_as_proposals=False)
            >>> self = self.sample(assign_result, bboxes, gt_bboxes, gt_labels)
        """
        if len(tf.shape(bboxes)) < 2:
            bboxes = tf.expand_dims(bboxes, axis=0)
        
        gt_bboxes, gt_flags = trim_zeros(gt_bboxes)
        if gt_labels!=None:
            gt_labels = tf.boolean_mask(gt_labels, gt_flags)
        if self.add_gt_as_proposals and len(gt_bboxes) > 0:
            if gt_labels is None:
                raise ValueError(
                    'gt_labels must be given when add_gt_as_proposals is True')
            bboxes = tf.concat([gt_bboxes, bboxes], axis=0)
            gt_inds, max_overlaps, labels = self.add_gt_(gt_labels, 
                                                         gt_inds, 
                                                         max_overlaps, 
                                                         labels)
            gt_ones = tf.ones(tf.shape(gt_bboxes)[0], dtype=tf.int8)
            non_gt = tf.zeros(tf.shape(bboxes)[0], dtype=tf.int8)
            gt_flags = tf.concat([gt_ones, non_gt], axis=0)
        num_expected_pos = int(self.num * self.pos_fraction)
        pos_inds = self.pos_sampler._sample_pos(
            gt_inds, num_expected_pos, bboxes=bboxes, **kwargs)
        num_sampled_pos = tf.shape(pos_inds)[0]
        num_expected_neg = self.num - num_sampled_pos
        if self.neg_pos_ub >= 0:
            _pos = tf.math.maximum(1, num_sampled_pos)
            neg_upper_bound = int(self.neg_pos_ub * _pos)
            if num_expected_neg > neg_upper_bound:
                num_expected_neg = neg_upper_bound
        neg_inds = self.neg_sampler._sample_neg(
            gt_inds, num_expected_neg, bboxes=bboxes, **kwargs)
        pos_bboxes = tf.gather(bboxes, pos_inds)
        neg_bboxes = tf.gather(bboxes, neg_inds)
        pos_assigned_gt_inds = tf.gather(gt_inds, pos_inds) - 1
        pos_gt_bboxes = tf.gather(gt_bboxes, pos_assigned_gt_inds)
        pos_gt_labels = tf.gather(labels, pos_inds)
        return (pos_inds, neg_inds, pos_bboxes, 
                neg_bboxes, pos_gt_bboxes,
                pos_assigned_gt_inds, pos_gt_labels)
    
    def add_gt_(self, gt_labels, gt_inds, max_overlaps, labels=None):
        """Add ground truth as assigned results.
        Args:
            gt_labels (torch.Tensor): Labels of gt boxes
        """
        self_inds = tf.range(
            1, len(gt_labels) + 1, dtype=tf.int32)
        gt_inds = tf.concat([self_inds, gt_inds], axis=0)

        self.max_overlaps = tf.concat(
            [tf.ones(len(gt_labels)), max_overlaps], axis=0)

        if labels is not None:
            labels = tf.concat([gt_labels, labels], axis=0)
        return gt_inds, max_overlaps, labels