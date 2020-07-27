import tensorflow as tf
from awsdet.core.bbox import bbox_target
from awsdet.core.bbox import transforms
from awsdet.models.losses import losses
from .. import builder
from ..registry import HEADS

@HEADS.register_module
class CascadeHead(tf.keras.Model):
    
    def __init__(self,
                 num_stages=3,
                 stage_loss_weights=[1, 0.5, 0.25],
                 iou_thresholds=[0.5, 0.6, 0.7],
                 class_agnostic=True,
                 num_classes=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 **kwargs):
        super(CascadeHead, self).__init__(**kwargs)
        assert len(stage_loss_weights) == num_stages
        assert len(bbox_head) == num_stages
        assert len(iou_thresholds) == num_stages
        assert class_agnostic or num_classes
        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights
        self.class_agnostic = class_agnostic
        self.num_classes = num_classes
        if bbox_roi_extractor is not None and bbox_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(bbox_roi_extractor)
            self.bbox_heads = [builder.build_head(head) for head in bbox_head]
            
        self.bbox_targets = []
        for iou, bbox_head in zip(iou_thresholds, self.bbox_heads):
            target = bbox_target.ProposalTarget(
                target_means=bbox_head.target_means,
                target_stds=bbox_head.target_stds, 
                num_rcnn_deltas=512,
                positive_fraction=0.25,
                pos_iou_thr=iou,
                neg_iou_thr=0.1,
                num_classes=1 if self.class_agnostic else self.num_classes)
            self.bbox_targets.append(target)
    
    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=True):
        if training:
            proposals_list, rcnn_feature_maps, gt_boxes, \
            gt_class_ids, img_metas = inputs
        else:
            proposals_list, rcnn_feature_maps, img_metas = inputs
        logits = []
        probs = []
        deltas = []
        target_matches = []
        target_deltas = []
        in_weights = []
        out_weights = []
        batch_size = img_metas.shape[0]
        loss_dict = {}
        rois_list = proposals_list
        for i in range(self.num_stages):
            if training:
                rois_list, rcnn_target_matches, rcnn_target_deltas, inside_weights, \
                    outside_weights = self.bbox_targets[i].build_targets( \
                    rois_list, gt_boxes, gt_class_ids, img_metas)
                target_matches.append(rcnn_target_matches)
                target_deltas.append(rcnn_target_deltas)
                in_weights.append(inside_weights)
                out_weights.append(outside_weights)    
            pooled_regions_list = self.bbox_roi_extractor(
                (rois_list, rcnn_feature_maps, img_metas), training=training)
            rcnn_class_logits, rcnn_probs, rcnn_deltas = self.bbox_heads[i](pooled_regions_list, training=training)
            if training:
                loss_dict['rcnn_class_loss_stage_{}'.format(i)] = losses.rcnn_class_loss(rcnn_class_logits, 
                                                                                         rcnn_target_matches)
                loss_dict['rcnn_box_loss_stage_{}'.format(i)] = losses.rcnn_bbox_loss(rcnn_deltas,
                                                                                      rcnn_target_deltas, 
                                                                                      inside_weights, 
                                                                                      outside_weights)
            logits.append(rcnn_class_logits)
            probs.append(rcnn_probs)
            deltas.append(rcnn_deltas)
            roi_shapes = [tf.shape(i)[0] for i in rois_list]
            refinements = tf.split(rcnn_deltas, roi_shapes)
            new_rois = []
            if i<(self.num_stages-1):
                for j in range(batch_size):
                    new_rois.append(transforms.delta2bbox(rois_list[j], refinements[j],
                                                   target_means=self.bbox_heads[i].target_means, \
                                                   target_stds=self.bbox_heads[i].target_stds))
                rois_list = new_rois
        if training:
            return loss_dict
        else:
            detections_list = self.bbox_heads[-1].get_bboxes(rcnn_probs,
                                                            rcnn_deltas,
                                                            rois_list,
                                                            img_metas)
            detections_dict = {
                    'bboxes': detections_list[0][0],
                    'labels': detections_list[0][1],
                    'scores': detections_list[0][2]
            }
            return detections_dict
