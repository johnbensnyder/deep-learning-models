import tensorflow as tf

from .builder import IOU_CALCULATORS


@IOU_CALCULATORS.register_module()
class BboxOverlaps2D(object):
    """2D IoU Calculator."""

    def __call__(self, bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
        """Calculate IoU between 2D bboxes.
        Args:
            bboxes1 (Tensor): bboxes have shape (m, 4) in <x1, y1, x2, y2>
                format, or shape (m, 5) in <x1, y1, x2, y2, score> format.
            bboxes2 (Tensor): bboxes have shape (m, 4) in <x1, y1, x2, y2>
                format, shape (m, 5) in <x1, y1, x2, y2, score> format, or be
                empty. If is_aligned is ``True``, then m and n must be equal.
            mode (str): "iou" (intersection over union) or iof (intersection
                over foreground).
        Returns:
            ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)
        """
        return compute_overlaps(bboxes1, bboxes2, mode=mode, is_aligned=is_aligned, eps=eps)
        
    def __repr__(self):
        """str: a string describing the module"""
        repr_str = self.__class__.__name__ + '()'
        return repr_str
        
    
def compute_overlaps(boxes1, boxes2, mode='iou', is_aligned=False, eps=1e-6):
    '''Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    '''
    # 1. Tile boxes2 and repeat boxes1. This allows us to compare
    # every boxes1 against every boxes2 without loops.
    # TF doesn't have an equivalent to np.repeat() so simulate it
    # using tf.tile() and tf.reshape.

    num_boxes1 = tf.shape(boxes1)[0]
    num_boxes2 = tf.shape(boxes2)[0]
    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),
                            [1, 1, num_boxes2]), [-1, 4])
    b2 = tf.tile(boxes2, [num_boxes1, 1])
    # 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    # 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection
    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = tf.reshape(iou, [num_boxes1, num_boxes2])
    return overlaps