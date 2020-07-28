# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env bash
TRAIN_CFG=$1

echo ""
echo "TRAIN_CFG: ${TRAIN_CFG}"
echo ""

cd /workspace/shared_workspace/deep-learning-models/models/vision/detection
export PYTHONPATH=${PYTHONPATH}:${PWD}

mpirun --allow-run-as-root \
        -x FI_PROVIDER="efa" \
        -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/efa/lib:/usr/local/lib:/nccl/build/lib:/aws-ofi-nccl/install/lib \
        -x NCCL_DEBUG=INFO \
        -x NCCL_TREE_THRESHOLD=0 \
        --hostfile /root/.ssh/hosts \
        --mca plm_rsh_no_tree_spawn 1 -bind-to none -map-by slot -mca pml ob1 \
        --mca btl_vader_single_copy_mechanism none \
        --mca oob_tcp_if_include ens5 \
        --mca btl_tcp_if_include ens5 \
        python tools/train.py \
        --configuration ${TRAIN_CFG}