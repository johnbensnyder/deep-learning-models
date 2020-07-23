# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
import numpy as np

class DataGenerator:

    def __init__(self, dataset, num_gpus=0, index=0, shuffle=False):
        self.dataset = dataset
        self.shuffle = shuffle
        self.gpu_index = index
        self.num_gpus = num_gpus
        if num_gpus > 0:
            self.num_examples = len(dataset) // num_gpus
        else:
            self.num_examples = len(dataset)
    
    def __call__(self):
        if self.num_gpus == 0:
            indices = np.arange(0, len(self.dataset))
        else:
            if self.dataset.train:
                indices = np.arange(0, len(self.dataset)) # ensure that each worker has a different seed
            else:
                indices = np.arange(self.gpu_index, len(self.dataset), self.num_gpus)
        while True:
            if self.shuffle:
                np.random.shuffle(indices)

            print('Starting new loop for GPU:', self.gpu_index)
            for img_idx in indices:
                if self.dataset.train:
                    img_instance = self.dataset[img_idx]
                    img_meta = self.build_img_meta(img_instance)
                    img = img_instance['img']
                    gt_boxes = img_instance['gt_bboxes']
                    yield img, img_meta, gt_boxes
                else:
                    img_instance = self.dataset[img_idx]
                    img_meta = self.build_img_meta(img_instance)
                    img = img_instance['img']
                    yield img, img_meta
    
    def build_img_meta(self, img_instance):
        meta = []
        meta.extend(img_instance['img_metas']['ori_shape'])
        meta.extend(img_instance['img_metas']['img_shape'])
        meta.extend(img_instance['img_metas']['pad_shape'])
        meta.append(img_instance['img_metas']['flip'])
        meta.extend(img_instance['img_metas']['scale_factor'])
        return np.array(meta)