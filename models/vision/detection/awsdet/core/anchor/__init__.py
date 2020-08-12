# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-

from .builder import build_anchor_generator, ANCHOR_GENERATORS
from .anchor_generator import AnchorGenerator

__all__ = ['build_anchor_generator', 'ANCHOR_GENERATORS',
           'AnchorGenerator']
