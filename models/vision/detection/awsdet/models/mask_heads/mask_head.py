import tensorflow as tf
from tensorflow.keras import layers
from ..registry import HEADS

@HEADS.register_module
class MaskHead(tf.keras.Model):
    def __init__(self, num_classes, depth=4):
        super().__init__()
        self.depth = depth
        for layer in range(self.depth):
            self.__dict__['conv_{}'.format(layer)] = tf.keras.layers.Conv2D(256, (3, 3), 
                                                    padding="same", activation='relu', 
                                                    name="mask_conv_{}".format(layer))
        self.deconv = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=2, 
                                                    activation="relu",
                                                    name="mask_deconv")
        self.masks = tf.keras.layers.Conv2D(num_classes, (1, 1), 
                                                strides=1, activation="sigmoid", name="mask")
        self.loss_func = tf.keras.losses.SparseCategoricalCrossentropy()
            
    @tf.function(experimental_relax_shapes=True)
    def call(self, mask_rois, training=True):
        mask_rois = tf.concat(mask_rois, axis=0)
        for layer in range(self.depth):
            mask_rois = self.__dict__['conv_{}'.format(layer)](mask_rois)
        mask_rois = self.deconv(mask_rois)
        return self.masks(mask_rois)
    
    @tf.function(experimental_relax_shapes=True)
    def gather_mask_predictions(self, rcnn_masks, rcnn_target_matchs):
        rcnn_masks = tf.transpose(rcnn_masks, [0, 3, 1, 2])
        target_length = tf.range(tf.shape(rcnn_target_matchs)[0])
        mask_idx = tf.transpose(tf.stack([target_length, rcnn_target_matchs]))
        masks = tf.gather_nd(rcnn_masks, mask_idx)
        masks = tf.expand_dims(masks, axis=[-1])
        return masks

    def compute_fg_offset(self, fg_assignments, gt_masks):
        fg_size = tf.shape(fg_assignments)
        roi_length = fg_size//tf.shape(gt_masks)[0]
        num_masks = tf.shape(gt_masks)[1]
        idx = tf.range(tf.shape(fg_assignments)[0])
        idx_multiplier = (idx//roi_length)*num_masks
        fg_adjusted = tf.cast(fg_assignments, tf.int32) + idx_multiplier
        return fg_adjusted

    def reshape_masks(self, gt_masks):
        batch_size = tf.shape(gt_masks)[0]
        num_masks = tf.shape(gt_masks)[1]
        return tf.reshape(gt_masks, [batch_size*num_masks,
                                     tf.shape(gt_masks)[2], 
                                     tf.shape(gt_masks)[3], 1])

    @tf.function(experimental_relax_shapes=True)
    def crop_masks(self, rois_list, fg_assignments, gt_masks, img_metas, size=(28, 28)):
        H = tf.reduce_mean(img_metas[...,6])
        W = tf.reduce_mean(img_metas[...,7])
        rois = tf.concat(rois_list, axis=0) / tf.stack([H, W, H ,W])
        gt_masks_reshape = self.reshape_masks(gt_masks)
        fg_adjusted = self.compute_fg_offset(fg_assignments, gt_masks)
        cropped_masks = tf.image.crop_and_resize(gt_masks_reshape,
                             rois,
                             fg_adjusted,
                             size,
                             method='nearest')
        return cropped_masks
    
    @tf.function(experimental_relax_shapes=True)
    def loss(self, masks_pred, rcnn_target_matchs, rois_list, 
                  fg_assignments, gt_masks, img_metas):
        masks_pred = self.gather_mask_predictions(masks_pred, rcnn_target_matchs)
        masks_true = self.crop_masks(rois_list, fg_assignments, gt_masks, img_metas)
        masks_pred = tf.boolean_mask(masks_pred, rcnn_target_matchs!=0)
        masks_true = tf.boolean_mask(masks_true, rcnn_target_matchs!=0)
        masks_pred = tf.reshape(masks_pred, [-1])
        masks_pred = tf.transpose(tf.stack([1 - masks_pred, masks_pred]))
        masks_true = tf.reshape(masks_true, [-1])
        loss = self.loss_func(masks_true, masks_pred)
        return loss
    
    @tf.function(experimental_relax_shapes=True)
    def mold_masks(self, masks, bboxes, img_meta, threshold=0.5):
        mask_array = tf.TensorArray(tf.int32, size=tf.shape(masks)[0])
        bboxes = tf.cast(bboxes, tf.int32)
        img_meta = tf.cast(img_meta, tf.int32)
        for idx in tf.range(100):
            mask_array = mask_array.write(idx, self._mold_single_mask(masks[idx], bboxes[idx], img_meta, threshold))
        mask_array = mask_array.stack()
        return mask_array
    
    @tf.function(experimental_relax_shapes=True)
    def _mold_single_mask(self, mask, bbox, img_meta, threshold=0.5):
        '''
        Resize a mask and paste to background for image
        '''
        y1 = bbox[0]
        x1 = bbox[1]
        y2 = bbox[2] 
        x2 = bbox[3]
        h = y2 - y1
        w = x2 - x1
        if tf.math.multiply(h, w)<=0:
            return tf.zeros((img_meta[6], img_meta[7], 1), dtype=tf.int32)
        mask_resize = tf.cast(tf.image.resize(mask, (h, w), method='nearest')>threshold, tf.int32)
        pad = [[y1, img_meta[6]-y2], [x1, img_meta[7]-x2], [0,0]]
        mask_resize = tf.pad(mask_resize, pad)
        return mask_resize