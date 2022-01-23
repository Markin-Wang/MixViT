python3 -m torch.distributed.launch --nproc_per_node 2 train.py \
--name soyloc_base \
--dataset soyloc \
--model_type R50-ViT-B_16 \
--pretrained_dir ./pretrained_models/R50+ViT-B_16.npz \
--img_size 448 \
--resize_size 500 \
--train_batch_size 16 \
--learning_rate 1e-3 \
--num_steps 4000 \
--fp16 \
--eval_every 32 \
--mask \
--mask_weight 0.1 \
--mask_topk 0.02 \
--mask_cls_weight 1 \
#--decay_type linear \
#--weight_decay 1e-4