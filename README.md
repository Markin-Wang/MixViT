# Mix-ViT: Mixing Attentive Vision Transformer for Ultra-Fine-Grained Visual Categorization

Official PyTorch implementation of [Mix-ViT: Mixing Attentive Vision Transformer for Ultra-Fine-Grained Visual Categorization](https://www.sciencedirect.com/science/article/pii/S0031320322006112) accepted by Pattern Recognition. 

If you use the code in this repo for your work, please cite the following bib entries:

    @article{yu2023mix,
      title={Mix-ViT: Mixing attentive vision transformer for ultra-fine-grained visual categorization},
      author={Yu, Xiaohan and Wang, Jun and Zhao, Yang and Gao, Yongsheng},
      journal={Pattern Recognition},
      volume={135},
      pages={109131},
      year={2023},
      publisher={Elsevier}
    }


## Abstract
<div style="text-align:justify"> Ultra-fine-grained visual categorization (ultra-FGVC) moves down the taxonomy level to classify sub-granularity categories of fine-grained objects. This inevitably poses a challenge, i.e., classifying highly similar objects with limited samples, which impedes the performance of recent advanced vision transformer methods. To that end, this paper introduces Mix-ViT, a novel mixing attentive vision transformer to address the above challenge towards improved ultra-FGVC. The core design is a self-supervised module that mixes the high-level sample tokens and learns to predict whether a token has been substituted after attentively substituting tokens. This drives the model to understand the contextual discriminative details among inter-class samples. Via incorporating such a self-supervised module, the network gains more knowledge from the intrinsic structure of input data and thus improves generalization capability with limited training sample. The proposed Mix-ViT achieves competitive performance on seven publicly available datasets, demonstrating the potential of vision transformer compared to CNN for the first time in addressing the challenging ultra-FGVC tasks. </div>


<img src='architecture.jpeg' width='1280' height='350'>


## Prerequisites

The following packages are required to run the scripts:
- [Python >= 3.6]
- [PyTorch = 1.8]
- [Torchvision]
- [Apex]


## Download Google pre-trained ViT models

* [Get models in this link](https://console.cloud.google.com/storage/vit_models/): ViT-B_16, ViT-B_32...
```bash
wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz
```

## Dataset
You can download the datasets from the links below:

+ [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html).
+ [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/)
+ [Cotton and Soy.Loc](https://drive.google.com/drive/folders/1UkWRepieAvEVEn3Z8n1Zx04bASvvqL7G?usp=sharing).


## Training scripts for FFVT on Cotton dataset.
Train the model on the Cotton dataset. We run our experiments on 4x2080Ti/4x1080Ti with the batchsize of 8 for each card.

    $ python3 -m torch.distributed.launch --nproc_per_node 4 train.py --name {name} --dataset cotton --model_type ViT-B_16 --pretrained_dir {pretrained_model_dir} --img_size 384 --resize_size 500 --train_batch_size 8 --learning_rate 0.02 --num_steps 2000 --fp16 --eval_every 16 --feature_fusion

## Training scripts for FFVT on Soy.Loc dataset.
Train the model on the SoyLoc dataset. We run our experiments on 4x2080Ti/4x1080Ti with the batchsize of 8 for each card.

    $ python3 -m torch.distributed.launch --nproc_per_node 4 train.py --name {name} --dataset soyloc --model_type ViT-B_16 --pretrained_dir {pretrained_model_dir} --img_size 384 --resize_size 500 --train_batch_size 8 --learning_rate 0.02 --num_steps 4750 --fp16 --eval_every 50 --feature_fusion
    
## Training scripts for FFVT on CUB dataset.
Train the model on the CUB dataset. We run our experiments on 4x2080Ti/4x1080Ti with the batchsize of 8 for each card.

    $ python3 -m torch.distributed.launch --nproc_per_node 4 train.py --name {name} --dataset CUB --model_type ViT-B_16 --pretrained_dir {pretrained_model_dir} --img_size 448 --resize_size 600 --train_batch_size 8 --learning_rate 0.02 --num_steps 10000 --fp16 --eval_every 200 --feature_fusion
    
## Training scripts for FFVT on Dogs dataset.
Train the model on the Dog dataset. We run our experiments on 4x2080Ti/4x1080Ti with the batchsize of 4 for each card.

    $ python3 -m torch.distributed.launch --nproc_per_node 4 train.py --name {name} --dataset CUB --model_type ViT-B_16 --pretrained_dir {pretrained_model_dir} --img_size 448 --resize_size 600 --train_batch_size 4 --learning_rate 3e-3 --num_steps 30000 --fp16 --eval_every 300 --feature_fusion --decay_type linear --num_token 24
    
        
            
## Download  Models


[Trained model Google Drive](https://drive.google.com/drive/folders/1k1vqc0avk_zpCAVuLNZpVX-w-Q3xXf-5?usp=sharing)




## Acknowledgment
Thanks for the advice and guidance given by Dr.Xiaohan Yu and Prof. Yongsheng Gao.

Our project references the codes in the following repos. Thanks for thier works and sharing.
- [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)
- [TransFG](https://github.com/TACJu/TransFG)





