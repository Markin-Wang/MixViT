python3 -m torch.distributed.launch --nproc_per_node 2 train.py \
--name $LOG_PATH \
--dataset soybean_aging_R5 \
--model_type R50-ViT-B_16 \
--pretrained_dir ./pretrained_models/R50+ViT-B_16.npz \
--img_size 448 \
--resize_size 600 \
--learning_rate 5e-3 \
--num_steps  15500 \
--fp16 \
--eval_every 155 \
--decay_type linear \
--weight_decay 1e-4 \
--num_parts 5 \
--mask_weight 4 \
--mask_topk 0.2 \
--seed 3567 \
--mask 