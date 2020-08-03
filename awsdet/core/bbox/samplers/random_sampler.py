import tensorflow as tf

from ..builder import BBOX_SAMPLERS
from .base_sampler import BaseSampler
from awsdet.core.bbox import demodata

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
        self.rng = demodata.ensure_rng(kwargs.get('rng', None))
    
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
    
    def _sample_pos(self, assign_result, num_expected, **kwargs):
        """Randomly sample some positive samples."""
        pos_inds = tf.where(assign_result.gt_inds > 0)
        