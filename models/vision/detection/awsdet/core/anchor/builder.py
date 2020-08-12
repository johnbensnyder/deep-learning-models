# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-

from awsdet.utils import Registry, build_from_cfg

ANCHOR_GENERATORS = Registry('Anchor generator')


def build_anchor_generator(cfg, default_args=None):
    return build_from_cfg(cfg, ANCHOR_GENERATORS, default_args)
