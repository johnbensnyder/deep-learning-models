import sys
import os
import time
from tqdm.notebook import tqdm
from time import sleep
sys.path.append('..')

import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import tensorflow_addons as tfa
from tqdm import tqdm
import numpy as np

from awsdet.utils.misc.config import Config
from awsdet.utils.logger import get_root_logger
from awsdet.utils.runner.dist_utils import init_dist
from awsdet.datasets.data_generator import DataGenerator
from awsdet.datasets import build_dataset
from awsdet.datasets.loader.build_loader import build_dataloader

import matplotlib.pyplot as plt
from awsdet.models.builder import build_backbone, build_neck, build_head

from awsdet.datasets.data_generator import DataGenerator

config_path = '../configs/faster_rcnn/EC2/faster_rcnn_r50_fpn_1x_coco.py'
cfg = Config.fromfile(config_path)
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
cfg.work_dir = os.path.join('./work_dirs',
                                os.path.splitext(os.path.basename(config_path))[0])
log_file = os.path.join(cfg.work_dir, f'{timestamp}.log')

init_dist()

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

coco = build_dataset(cfg.data.train)

train_tdf, train_size = build_dataloader(coco, 4)
train_tdf_iter = iter(train_tdf.prefetch(32).repeat())

class RPN(tf.keras.Model):
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg.model.backbone)
        self.neck = build_neck(cfg.model.neck)
        self.rpn_head = build_head(cfg.model.rpn_head)
        self.backbone.layers[0].load_weights(cfg.model.backbone.weights_path)
    
    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=True):
        if training:
            imgs, img_metas, gt_bboxes, gt_labels = inputs
        else:
            imgs, img_metas = inputs
        C2, C3, C4, C5 = self.backbone(imgs, training=training)
        P2, P3, P4, P5, P6 = self.neck((C2, C3, C4, C5), training=training)
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        rpn_class_logits, rpn_probs, rpn_deltas = self.rpn_head(rpn_feature_maps, training=training)
        if training:
            all_labels, all_label_weights, all_bbox_targets, \
            all_bbox_weights, inds_list, pos_flags_list = self.rpn_head.get_targets(rpn_class_logits,
                                                                                     gt_bboxes,
                                                                                     img_metas,
                                                                                     gt_labels)
            label_loss = self.cross_entropy(rpn_class_logits, all_labels, all_label_weights)
            box_loss = self.box_loss(rpn_deltas, all_bbox_targets, all_bbox_weights)
            return label_loss + box_loss
        else:
            return rpn_class_logits, rpn_probs, rpn_deltas
    
    def cross_entropy(self, rpn_class_logits, all_labels, all_label_weights):
        num_images = tf.shape(all_labels)[0]
        logits = tf.concat([tf.reshape(i, [num_images, -1, 2]) \
                            for i in rpn_class_logits], axis=1)[:,:,1]
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(tf.cast(all_labels, logits.dtype), 
                                                                logits) * all_label_weights
        loss = tf.reduce_sum(cross_entropy)/tf.reduce_sum(all_label_weights)
        return loss

    def smooth_l1_loss(self, pred, target, beta=1.0):
        diff = tf.math.abs(pred - target)
        loss = tf.keras.backend.switch(diff < beta, lambda: 0.5 * diff * diff / beta, lambda: diff - 0.5 * beta)
        return loss

    def box_loss(self, rpn_deltas, all_bbox_targets, all_bbox_weights):
        num_images = tf.shape(all_bbox_targets)[0]
        deltas = tf.concat([tf.reshape(i, [num_images, -1, 4]) \
                            for i in rpn_deltas], axis=1)
        loss = self.smooth_l1_loss(deltas, all_bbox_targets) * all_bbox_weights
        loss = tf.reduce_sum(loss)/tf.reduce_sum(all_bbox_weights)
        return loss
    
imgs, img_metas, gt_bboxes, gt_labels = next(train_tdf_iter)
img_metas = tf.cast(img_metas, tf.float16)
gt_bboxes = tf.cast(gt_bboxes, tf.float16)

model = RPN(cfg)
loss = model((imgs, img_metas, gt_bboxes, gt_labels), training=True)
opt = tfa.optimizers.SGDW(1e-4, 1e-2)
opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, loss_scale='dynamic')

@tf.function(experimental_relax_shapes=True)
def train_step(inputs):
    with tf.GradientTape() as tape:
        loss = model(inputs)
        scaled_loss = opt.get_scaled_loss(loss)
    scaled_grads = tape.gradient(scaled_loss, model.trainable_variables)
    grads = opt.get_unscaled_gradients(scaled_grads)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return loss

loss = train_step(next(train_tdf_iter))

progressbar = tqdm(range(50000))
loss_history = []
for i in progressbar:
    imgs, img_metas, gt_bboxes, gt_labels = next(train_tdf_iter)
    loss = train_step((imgs, img_metas, gt_bboxes, gt_labels))
    loss_history.append(loss.numpy())
    progressbar.set_description("Loss: {}".format(np.array(loss_history[-50:]).mean()))