mpirun -np 4 -H localhost:4 --allow-run-as-root \
    python tools/train_docker.py \
    --configuration configs/docker_mask_rcnn.py \
    --base_learning_rate 15e-3 \
    --batch_size_per_device 2 \
    --fp16 True \
    --schedule 1x \
    --warmup_init_lr_scale 3.0 \
    --warmup_steps 500 \
    --use_rcnn_bn False \
    --use_conv True \
    --ls 0.0 \
    --name test
    
    
    configs/docker_default_config.py
    
    docker_mask_rcnn.py