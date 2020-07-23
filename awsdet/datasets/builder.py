# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-

from awsdet.utils import Registry, build_from_cfg

DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')

def build_dataset(cfg, default_args=None):
    #TODO: Handle cases with multiple datasets, etc.
    dataset = build_from_cfg(cfg, DATASETS, default_args)
    return dataset