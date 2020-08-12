import numpy as np
import tensorflow as tf

from ..builder import BBOX_CODERS
from .base_bbox_coder import BaseBBoxCoder
from ..transforms import bbox2delta, delta2bbox, bbox_clip

@BBOX_CODERS.register_module()
class DeltaYXHWBBoxCoder(BaseBBoxCoder):
    """Delta XYWH BBox coder.
    Following the practice in `R-CNN <https://arxiv.org/abs/1311.2524>`_,
    this coder encodes bbox (x1, y1, x2, y2) into delta (dx, dy, dw, dh) and
    decodes delta (dx, dy, dw, dh) back to original bbox (x1, y1, x2, y2).
    Args:
        target_means (Sequence[float]): Denormalizing means of target for
            delta coordinates
        target_stds (Sequence[float]): Denormalizing standard deviation of
            target for delta coordinates
    """

    def __init__(self,
                 target_means=(0., 0., 0., 0.),
                 target_stds=(1., 1., 1., 1.)):
        super(BaseBBoxCoder, self).__init__()
        self.means = target_means
        self.stds = target_stds
        
    def encode(self, bboxes, gt_bboxes):
        """Get box regression transformation deltas that can be used to
        transform the ``bboxes`` into the ``gt_bboxes``.
        Args:
            bboxes (torch.Tensor): Source boxes, e.g., object proposals.
            gt_bboxes (torch.Tensor): Target of the transformation, e.g.,
                ground-truth boxes.
        Returns:
            torch.Tensor: Box transformation deltas
        """
        encoded_bboxes = bbox2delta(bboxes, gt_bboxes, self.means, self.stds)
        return encoded_bboxes
    
    def decode(self,
               bboxes,
               deltas,
               max_shape=None,
               wh_ratio_clip=np.abs(np.log(16.0/1000))):
        """Apply transformation `pred_bboxes` to `boxes`.
        Args:
            boxes (torch.Tensor): Basic boxes.
            pred_bboxes (torch.Tensor): Encoded boxes with shape
            max_shape (tuple[int], optional): Maximum shape of boxes.
                Defaults to None.
            wh_ratio_clip (float, optional): The allowed ratio between
                width and height.
        Returns:
            torch.Tensor: Decoded boxes.
        """
        decoded_bboxes = delta2bbox(bboxes, 
                                    deltas, 
                                    self.means, 
                                    self.stds, 
                                    wh_ratio_clip=wh_ratio_clip)
        if max_shape:
            decoded_bboxes = bbox_clip(bboxes, max_shape)

        return decoded_bboxes