# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
from .base_assigner import BaseAssigner
from .max_iou_assigner import MaxIoUAssigner
from .assign_result import AssignResult

__all__ = [
    'BaseAssigner', 'MaxIoUAssigner', 'AssignResult'
]