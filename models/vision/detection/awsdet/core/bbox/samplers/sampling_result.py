import tensorflow as tf
from awsdet.utils import util_mixins
from awsdet.models.utils.misc import trim_zeros

class SamplingResult(util_mixins.NiceRepr):
    """Bbox sampling result.
    Example:
        >>> # xdoctest: +IGNORE_WANT
        >>> from mmdet.core.bbox.samplers.sampling_result import *  # NOQA
        >>> self = SamplingResult.random(rng=10)
        >>> print(f'self = {self}')
        self = <SamplingResult({
            'neg_bboxes': torch.Size([12, 4]),
            'neg_inds': tensor([ 0,  1,  2,  4,  5,  6,  7,  8,  9, 10, 11, 12]),
            'num_gts': 4,
            'pos_assigned_gt_inds': tensor([], dtype=torch.int64),
            'pos_bboxes': torch.Size([0, 4]),
            'pos_inds': tensor([], dtype=torch.int64),
            'pos_is_gt': tensor([], dtype=torch.uint8)
        })>
    """
    def __init__(self, pos_inds, neg_inds, bboxes, gt_bboxes, assign_result,
                 gt_flags):
        self.pos_inds = pos_inds
        self.neg_inds = neg_inds
        self.pos_bboxes = tf.gather(bboxes, pos_inds)
        self.neg_bboxes = tf.gather(bboxes, neg_inds)
        self.pos_is_gt = tf.gather(gt_flags, pos_inds)

        self.num_gts = tf.shape(gt_bboxes)[0]
        self.pos_assigned_gt_inds = tf.gather(assign_result.gt_inds, pos_inds) - 1
        
        if tf.rank(gt_bboxes)<2:
            gt_bboxes = tf.expand_dims(gt_bboxes, axis=0)
        self.pos_gt_bboxes = tf.gather(gt_bboxes, self.pos_assigned_gt_inds)
        
        self.pos_gt_labels = None
        if assign_result.labels is not None:
            self.pos_gt_labels = tf.gather(assign_result.labels, pos_inds)
    
    @property
    def bboxes(self):
        return tf.concat([self.pos_bboxes, self.neg_bboxes], axis=0)
    
    def __nice__(self):
        data = self.info.copy()
        data['pos_bboxes'] = data.pop('pos_bboxes').shape
        data['neg_bboxes'] = data.pop('neg_bboxes').shape
        parts = [f"'{k}': {v!r}" for k, v in sorted(data.items())]
        body = '    ' + ',\n    '.join(parts)
        return '{\n' + body + '\n}'
    
    def deconstruct(self):
        return (self.pos_inds, self.neg_inds, self.pos_bboxes, 
                self.neg_bboxes, self.pos_gt_bboxes, 
                self.pos_assigned_gt_inds, self.pos_gt_labels)
    
    @property
    def info(self):
        """Returns a dictionary of info about the object."""
        return {
            'pos_inds': self.pos_inds,
            'neg_inds': self.neg_inds,
            'pos_bboxes': self.pos_bboxes,
            'neg_bboxes': self.neg_bboxes,
            'pos_is_gt': self.pos_is_gt,
            'num_gts': self.num_gts,
            'pos_assigned_gt_inds': self.pos_assigned_gt_inds,
        }
