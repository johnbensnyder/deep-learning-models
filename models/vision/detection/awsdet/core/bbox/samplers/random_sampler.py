import tensorflow as tf

from ..builder import BBOX_SAMPLERS
from .base_sampler import BaseSampler

@BBOX_SAMPLERS.register_module()
class RandomSampler(BaseSampler):
    """Random sampler.
    Args:
        num (int): Number of samples
        pos_fraction (float): Fraction of positive samples
        neg_pos_up (int, optional): Upper bound number of negative and
            positive samples. Defaults to -1.
        add_gt_as_proposals (bool, optional): Whether to add ground truth
            boxes as proposals. Defaults to True.
    """
    def __init__(self,
                 num,
                 pos_fraction,
                 neg_pos_ub=-1,
                 add_gt_as_proposals=True,
                 **kwargs):
        super(RandomSampler, self).__init__(num, pos_fraction, neg_pos_ub,
                                            add_gt_as_proposals)
    
    def random_choice(self, gallery, num):
        """Random select some elements from the gallery.
        If `gallery` is a Tensor, the returned indices will be a Tensor;
        If `gallery` is a ndarray or list, the returned indices will be a
        ndarray.
        Args:
            gallery (Tensor | ndarray | list): indices pool.
            num (int): expected sample num.
        Returns:
            Tensor or ndarray: sampled indices.
        """
        gallery_size = tf.shape(gallery)[0]
        assert len(gallery_size) >= num
        perm = tf.random.shuffle(tf.range(gallery_size))[:num]
        rand_inds = tf.gather(gallery, perm)
        return rand_inds
    
    def _sample_pos(self, gt_inds, num_expected, **kwargs):
        """Randomly sample some positive samples."""
        pos_inds = tf.where(gt_inds > 0)
        pos_inds = tf.squeeze(pos_inds)
        if tf.rank(pos_inds)<1:
            pos_inds = tf.expand_dims(pos_inds, axis=0)   
        if tf.shape(pos_inds)[0] < num_expected:
            return pos_inds
        else:
            return tf.random.shuffle(pos_inds)[:num_expected]
        
    def _sample_neg(self, gt_inds, num_expected, **kwargs):
        """Randomly sample some negative samples."""
        neg_inds = tf.where(gt_inds == 0)
        neg_inds = tf.squeeze(neg_inds)
        neg_inds = tf.random.shuffle(neg_inds)
        num_neg = tf.shape(neg_inds)[0]
        while num_neg < num_expected:
            neg_inds = tf.concat([neg_inds, tf.random.shuffle(neg_inds)], axis=0)
            num_neg = tf.shape(neg_inds)[0]
        return neg_inds[:num_expected]
