from awsdet.models.utils import misc
import tensorflow as tf

class MaskTarget:
    
    def __init__(self):
        pass
    
    @tf.function(experimental_relax_shapes=True)
    def _extract_mask_single(self, roi, mask, 
                            gt_mask, fg_assignment, 
                            rcnn_target_match, H, W, 
                            mask_shape):
        #orig_shape=tf.shape(rcnn_target_match)[0]
        #fg_locs = tf.reshape(tf.where(rcnn_target_match!=0), [-1])
        #roi = tf.gather(roi, fg_locs)
        #gt_mask = tf.gather(gt_mask, fg_locs)
        #fg_assignment = tf.gather(fg_assignment, fg_locs)
        #rcnn_target_match = tf.gather(rcnn_target_match, fg_locs)
        norm_rois = tf.cast(roi, tf.float32) / tf.stack([H, W, H, W])
        '''print("\n\n\nrois")
        print(norm_rois)
        print("\n\n\nfg")
        print(fg_assignment)
        print("\n\n\ngt")
        print(gt_mask)
        print("\n\n\nmask_shape")
        print(mask_shape)'''
        cropped_targets = tf.image.crop_and_resize(tf.expand_dims(gt_mask, axis=-1), 
                             norm_rois, 
                             tf.cast(fg_assignment, tf.int32), 
                             mask_shape)
        rcnn_target_match = tf.stack([tf.range(tf.shape(rcnn_target_match)[0]), 
                                                    rcnn_target_match], axis=1)
        mask = tf.transpose(mask, perm=[0, 3, 1, 2])
        mask = tf.gather_nd(mask, rcnn_target_match)
        mask = tf.expand_dims(mask, axis=-1)
        #pad_shape = [[0, orig_shape-tf.shape(mask)[0]], [0,0], [0,0], [0,0]]
        #mask = tf.pad(mask, pad_shape)
        #cropped_targets = tf.pad(cropped_targets, pad_shape)
        return mask, cropped_targets

    @tf.function(experimental_relax_shapes=True) 
    def __call__(self, rois_list, masks, gt_masks, 
                   fg_assignments, rcnn_target_matchs, img_metas):
        batch_size = tf.shape(img_metas)[0]
        splits = tf.cast(tf.repeat(tf.shape(rcnn_target_matchs)[0]/batch_size, batch_size), tf.int32)
        rcnn_target_matchs_list = tf.stack(tf.split(rcnn_target_matchs, splits))
        fg_assignments_list = tf.stack(tf.split(fg_assignments, splits))
        rois = tf.stack(rois_list)
        masks = tf.stack(tf.split(masks, splits))
        shape = tf.cast(misc.calc_batch_padded_shape(img_metas), tf.float32)
        H = shape[0]
        W = shape[1]
        mask_shape = tf.shape(masks)[2:4]
        masks_list = tf.TensorArray(tf.float32, size = batch_size)
        cropped_targets_list = tf.TensorArray(tf.float32, size = batch_size)
        for i in range(batch_size):
            mask, cropped_targets = self._extract_mask_single(tf.gather(rois, i), 
                                                         tf.gather(masks, i), 
                                                         tf.gather(gt_masks, i), 
                                                         tf.gather(fg_assignments_list, i), 
                                                         tf.gather(rcnn_target_matchs_list, i), 
                                                         H, W, mask_shape)
            masks_list = masks_list.write(i, mask)
            cropped_targets_list = cropped_targets_list.write(i, cropped_targets)
        masks_list = masks_list.stack()
        cropped_targets_list = cropped_targets_list.stack()
        cropped_targets_list = tf.cast(cropped_targets_list>0.5, tf.int32)
        return masks_list, cropped_targets_list
    
    @tf.function
    def mask_loss(self, target_masks, pred_masks, rcnn_target_matchs, img_metas):
        batch_size = tf.shape(img_metas)[0]
        splits = tf.cast(tf.repeat(tf.shape(rcnn_target_matchs)[0]/batch_size, batch_size), tf.int32)
        mask_indices = tf.where(tf.stack(tf.split(rcnn_target_matchs, splits))!=0)
        targets = tf.gather_nd(target_masks, mask_indices)
        predictions = tf.gather_nd(pred_masks, mask_indices)
        targets = tf.keras.layers.Flatten()(targets)
        predictions = tf.keras.layers.Flatten()(predictions)
        targets = tf.reshape(targets, [-1])
        predictions = tf.reshape(predictions, [-1])
        predictions = tf.transpose(tf.stack([1-predictions, predictions]))
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False) \
                    (targets, predictions)
        return loss