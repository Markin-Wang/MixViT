python3 -m torch.distributed.launch --nproc_per_node 2 train.py \
--name cotton_mttrasnformer \
--dataset cotton \
--model_type R50-ViT-B_16 \
--pretrained_dir ./pretrained_models/R50+ViT-B_16.npz \
--img_size 448 \
--resize_size 600 \
--train_batch_size 16 \
--learning_rate 5e-3 \
--num_steps 3200 \
--fp16 \
--eval_every 80 \
--decay_type linear \
--weight_decay 1e-4 \
--eval_every 16 \
--num_parts 5 \
--mask_weight 4 \
--mask_topk 0.2 \
--mask

