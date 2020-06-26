import tensorflow as tf
import tensorflow_addons as tfa
from awsdet.core.mask import transforms
from awsdet.models.losses import losses
from ..registry import HEADS

@HEADS.register_module
class MaskHead(tf.keras.Model):
    def __init__(self, num_classes,
                       weight_decay=1e-5, 
                       group_norm=False,
                       batch_norm=False):
        super().__init__()
        self.num_classes = num_classes
        self.weight_decay = weight_decay
        assert not (group_norm & batch_norm), "Cannot use both group and batch norm"
        self.group_norm = group_norm
        self.batch_norm = batch_norm
        self._conv_0 = tf.keras.layers.Conv2D(256, (3, 3),
                                             padding="same",
                                             kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0,
                                                                                                      mode='fan_out'),
                                             kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                             activation=tf.keras.activations.relu,
                                             name="mask_conv_0")
        if self.group_norm:
            self._conv_0_gn = tfa.layers.GroupNormalization()
        if self.batch_norm:
            self._conv_0_bn = tf.keras.layers.BatchNormalization()
        self._conv_1 = tf.keras.layers.Conv2D(256, (3, 3),
                                             padding="same",
                                             kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0,
                                                                                                      mode='fan_out'),
                                             kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                             activation=tf.keras.activations.relu,
                                             name="mask_conv_0")
        if self.group_norm:
            self._conv_1_gn = tfa.layers.GroupNormalization()
        if self.batch_norm:
            self._conv_1_bn = tf.keras.layers.BatchNormalization()
        self._conv_2 = tf.keras.layers.Conv2D(256, (3, 3),
                                             padding="same",
                                             kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0,
                                                                                                      mode='fan_out'),
                                             kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                             activation=tf.keras.activations.relu,
                                             name="mask_conv_0")
        if self.group_norm:
            self._conv_2_gn = tfa.layers.GroupNormalization()
        if self.batch_norm:
            self._conv_2_bn = tf.keras.layers.BatchNormalization()
        self._conv_3 = tf.keras.layers.Conv2D(256, (3, 3),
                                             padding="same",
                                             kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0,
                                                                                                      mode='fan_out'),
                                             kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                             activation=tf.keras.activations.relu,
                                             name="mask_conv_0")
        if self.group_norm:
            self._conv_3_gn = tfa.layers.GroupNormalization()
        if self.batch_norm:
            self._conv_3_bn = tf.keras.layers.BatchNormalization()
        self._deconv = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=2, 
                                                    kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0,
                                                                                                      mode='fan_out'),
                                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                                    activation=tf.keras.activations.relu,
                                                    name="mask_deconv")
        self._masks = tf.keras.layers.Conv2D(self.num_classes, (1, 1),
                                             kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.001),
                                             kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                             strides=1, name="mask")
        
    @tf.function(experimental_relax_shapes=True)
    def call(self, mask_rois_list, training=True):
        mask_list = []
        for mask_rois in mask_rois_list:
            mask_rois = self._conv_0(mask_rois)
            if self.group_norm:
                mask_rois = self._conv_0_gn(mask_rois)
            if self.batch_norm:
                mask_rois = self._conv_0_bn(mask_rois, training=training)
            mask_rois = self._conv_1(mask_rois)
            if self.group_norm:
                mask_rois = self._conv_1_gn(mask_rois)
            if self.batch_norm:
                mask_rois = self._conv_1_bn(mask_rois, training=training)
            mask_rois = self._conv_2(mask_rois)
            if self.group_norm:
                mask_rois = self._conv_2_gn(mask_rois)
            if self.batch_norm:
                mask_rois = self._conv_2_bn(mask_rois, training=training)
            mask_rois = self._conv_3(mask_rois)
            if self.group_norm:
                mask_rois = self._conv_3_gn(mask_rois)
            if self.batch_norm:
                mask_rois = self._conv_3_bn(mask_rois, training=training)
            mask_rois = self._deconv(mask_rois)
            mask_rois = self._masks(mask_rois)
            mask_rois = tf.transpose(mask_rois, [0, 3, 1, 2])
            mask_rois = tf.expand_dims(mask_rois, [-1])
            mask_list.append(mask_rois)
        return mask_list
        
    @tf.function(experimental_relax_shapes=True)
    def gather_mask_predictions(self, pred_mask, rcnn_target_matchs):
        pred_mask = tf.boolean_mask(pred_mask, rcnn_target_matchs!=0)
        mask_indices = tf.range(tf.shape(rcnn_target_matchs)[0])
        mask_indices = tf.transpose(tf.stack([mask_indices, rcnn_target_matchs-1]))
        mask_indices = tf.boolean_mask(mask_indices, rcnn_target_matchs!=0)
        pred_mask = tf.gather_nd(pred_mask, mask_indices)
        return pred_mask

    @tf.function(experimental_relax_shapes=True)
    def crop_masks(self, rois, fg_assignments, gt_masks, img_metas, size=(28, 28)):
        H = tf.reduce_mean(img_metas[...,6])
        W = tf.reduce_mean(img_metas[...,7])
        norm_rois = tf.concat(rois, axis=0) / tf.stack([H, W, H ,W])
        cropped_masks = tf.image.crop_and_resize(gt_masks,
                             norm_rois,
                             fg_assignments,
                             size,
                             method='nearest')
        return cropped_masks

    @tf.function(experimental_relax_shapes=True)
    def _mask_loss_single(self, masks_pred, rcnn_target_matchs, rois, 
                      fg_assignments, gt_masks, img_metas):
        masks_pred = self.gather_mask_predictions(masks_pred, rcnn_target_matchs)
        masks_true = self.crop_masks(rois, fg_assignments, gt_masks, img_metas)
        masks_true = tf.boolean_mask(masks_true, rcnn_target_matchs!=0)
        loss = tf.math.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=masks_true, 
                                                                           logits=masks_pred))
        return loss

    @tf.function(experimental_relax_shapes=True)
    def mask_loss(self, masks_pred_list, rcnn_target_matchs, rois_list, 
                      fg_assignments, gt_masks, img_metas):
        batch_size = tf.shape(img_metas)[0]
        num_rois = tf.shape(rois_list[0])[0]
        fg_assignments = tf.cast(tf.reshape(fg_assignments, [batch_size, num_rois]), tf.int32)
        rcnn_target_matchs = tf.reshape(rcnn_target_matchs, [batch_size, num_rois])
        gt_masks = tf.expand_dims(gt_masks, [-1])
        loss = 0.
        valid_losses = 0
        for i in range(img_metas.shape[0]):
            single_loss=self._mask_loss_single(masks_pred_list[i], rcnn_target_matchs[i],
                                    rois_list[i], fg_assignments[i], gt_masks[i],
                                    img_metas[i])
            # if no masks detected, don't add to loss
            if tf.math.is_nan(single_loss):
                continue
            valid_losses += 1
            loss += single_loss
        # adjust in case we got any nan value
        loss *= tf.cast(img_metas.shape[0]/valid_losses, tf.float32)
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
        mask = tf.math.sigmoid(mask)
        mask_resize = tf.cast(tf.image.resize(mask, (h, w), method='nearest')>threshold, tf.int32)
        pad = [[y1, img_meta[6]-y2], [x1, img_meta[7]-x2], [0,0]]
        mask_resize = tf.pad(mask_resize, pad)
        return mask_resize