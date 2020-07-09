cd /workspace/shared_workspace/deep-learning-models/models/vision/detection
export PYTHONPATH=${PYTHONPATH}:${PWD}

mpirun --hostfile /root/.ssh/hosts \
    -x FI_PROVIDER="efa" \
    -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/efa/lib:/usr/local/lib:/nccl/build/lib:/aws-ofi-nccl/install/lib \
    -x NCCL_DEBUG=INFO \
    -x NCCL_TREE_THRESHOLD=0 \
    --mca plm_rsh_no_tree_spawn 1 -bind-to none -map-by slot -mca pml ob1 -mca btl ^openib \
    --mca btl_vader_single_copy_mechanism none \
    --mca oob_tcp_if_include ens5 \
    --mca btl_tcp_if_include ens5 \
    --allow-run-as-root \
    python tools/train.py --config configs/mask_rcnn/EC2/mask_rcnn_r50_fpn_1x_coco_exp.py
    
    
    
    -mca btl_tcp_if_exclude lo,docker0 \
    -x PATH \
    -x NCCL_SOCKET_IFNAME=^docker0,lo \
    -x NCCL_MIN_NRINGS=8 \
    -x NCCL_DEBUG=INFO \
    -x TF_CUDNN_USE_AUTOTUNE=0 \
    -x HOROVOD_CYCLE_TIME=0.5 \
    -x HOROVOD_FUSION_THRESHOLD=67108864 \
    
mpirun --hostfile /root/.ssh/hosts \
    --allow-run-as-root \
    --mca plm_rsh_no_tree_spawn 1 -bind-to none -map-by slot -mca pml ob1 -mca btl ^openib \
    -mca btl_tcp_if_exclude lo,docker0 \
    -mca btl_vader_single_copy_mechanism none \
    -x LD_LIBRARY_PATH \
    -x PATH \
    -x NCCL_SOCKET_IFNAME=^docker0,lo \
    -x NCCL_MIN_NRINGS=8 \
    -x NCCL_DEBUG=INFO \
    -x TF_CUDNN_USE_AUTOTUNE=0 \
    -x HOROVOD_CYCLE_TIME=0.5 \
    -x HOROVOD_FUSION_THRESHOLD=67108864 \
    python tools/train.py --config configs/mask_rcnn/EC2/mask_rcnn_r50_fpn_1x_coco_exp.py
    
    
    
    
    
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
             python tools/train.py --config configs/mask_rcnn/EC2/mask_rcnn_r50_fpn_1x_coco_exp.py