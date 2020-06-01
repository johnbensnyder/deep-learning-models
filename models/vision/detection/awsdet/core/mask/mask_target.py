from awsdet.models.utils import misc
import tensorflow as tf

class MaskTarget:
    
    def __init__(self, cropped_target_size = (28, 28),
                       loss_func=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)):
        self.crop_size = tf.constant(cropped_target_size)
        self.loss_func = loss_func

    @tf.function(experimental_relax_shapes=True)
    def batch_indices(self, fg_assignments, rcnn_target_matchs, img_metas):
        batch_size = tf.shape(img_metas)[0]
        batch_length = tf.shape(fg_assignments)[0]//batch_size
        batch_indices = tf.gather(tf.cast(tf.repeat(tf.range(batch_size), batch_length), tf.int32) * 1, 
                                  tf.squeeze(tf.where(rcnn_target_matchs!=0)))
        gt_indices = tf.cast(tf.gather(fg_assignments * 1, tf.squeeze(tf.where(rcnn_target_matchs!=0))), tf.int32)
        gt_indices = tf.transpose(tf.stack([batch_indices, gt_indices]))
        if tf.rank(gt_indices) == 1:
            gt_indices = tf.expand_dims(gt_indices, axis=0)
        return gt_indices

    @tf.function(experimental_relax_shapes=True)
    def fg_masks(self, gt_masks, fg_assignments, rcnn_target_matchs, img_metas):
        return tf.expand_dims(tf.gather_nd(gt_masks * 1, self.batch_indices(fg_assignments, rcnn_target_matchs, img_metas)), axis=-1)

    @tf.function(experimental_relax_shapes=True)
    def fg_regions(self, mask_pooled_regions_list, rcnn_target_matchs):
        return tf.gather(tf.concat(mask_pooled_regions_list, axis=0) * 1, tf.squeeze(tf.where(rcnn_target_matchs!=0)))
    
    @tf.function(experimental_relax_shapes=True)
    def crop_and_resize(self, rois_list, gt_masks, fg_assignments, rcnn_target_matchs, img_metas):
        H = img_metas[0,6]
        W = img_metas[0,7]
        norm_rois = tf.gather(tf.concat(rois_list, axis=0) * 1, 
                                tf.squeeze(tf.where(rcnn_target_matchs!=0))) \
                                / tf.stack([H, W, H ,W])
        if tf.rank(norm_rois)==1:
            norm_rois = tf.expand_dims(norm_rois, axis=0)
        fg_masks = self.fg_masks(gt_masks, fg_assignments, rcnn_target_matchs, img_metas)
        cropped_targets = tf.image.crop_and_resize(fg_masks, 
                             norm_rois, 
                             tf.range(tf.shape(norm_rois)[0]), 
                             self.crop_size)
        cropped_targets = tf.cast(cropped_targets, tf.int32)
        return cropped_targets
    
    @tf.function(experimental_relax_shapes=True)
    def get_targets(self, mask_pooled_regions_list, rois_list, gt_masks, 
                    fg_assignments, rcnn_target_matchs, img_metas):
        targets = self.crop_and_resize(rois_list, gt_masks, fg_assignments, 
                                       rcnn_target_matchs, img_metas)
        regions = self.fg_regions(mask_pooled_regions_list, rcnn_target_matchs)
        regions = tf.reshape(regions, [-1, 14, 14, 256])
        return regions, targets
    
    @tf.function(experimental_relax_shapes=True)
    def loss(self, pred, target, rcnn_target_matchs):
        pred = tf.transpose(pred, perm=[0, 3, 1, 2])
        locations = tf.squeeze(tf.gather(rcnn_target_matchs * 1, tf.where(rcnn_target_matchs!=0)))
        if tf.rank(locations)==0:
            locations = tf.expand_dims(locations, axis=0)
        locations = tf.transpose(tf.stack([tf.range(tf.shape(locations)[0]), locations]))
        pred = tf.gather_nd(pred * 1, locations)
        target = tf.reshape(target, [-1])
        pred = tf.reshape(pred, [-1])
        pred = tf.transpose(tf.stack([1 - pred, pred]))
        return self.loss_func(target, pred)