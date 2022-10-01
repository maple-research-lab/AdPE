# AdPE

<a href="https://github.com/marktext/marktext/releases/latest">
   <img src="https://img.shields.io/badge/AdPE-v1.0.0-green">
   <img src="https://img.shields.io/badge/platform-Linux%20%7C%20Mac%20-green">
   <img src="https://img.shields.io/badge/Language-python3-green">
   <img src="https://img.shields.io/badge/dependencies-tested-green">
   <img src="https://img.shields.io/badge/licence-MIT-green">
</a>   

AdPE is a MIM based self-supervised learning methods, which is submitted to ICLR.

Copyright (C) 2022 Xiao Wang, Ying Wang, Ziwei Xuan, Guo-Jun Qi

License: MIT for academic use.

Contact: Xiao Wang (wang3702@purdue.edu), Guo-Jun Qi (guojunq@gmail.com)

## Introduction
Unsupervised learning of vision transformers seeks to pretrain an encoder via pretext tasks without labels. Among them is the Masked Image Modeling (MIM) aligned with pretraining of language transformers by predicting masked patches as a pretext task. A criterion in unsupervised pretraining is the pretext task needs to be sufficiently hard to prevent the transformer encoder from learning trivial low-level features not generalizable well to downstream tasks. For this purpose, we propose an Adversarial Positional Embedding (AdPE) approach -- It distorts the local visual structures by perturbing the position encodings so that the learned transformer cannot simply use the locally correlated patches to predict the missing ones. We hypothesize that it forces the transformer encoder to learn more discriminative features in a global context with stronger generalizability to downstream tasks. We will consider both absolute and relative positional encodings, where adversarial positions can be imposed both in the embedding mode and the coordinate mode. We will also present a new MAE+ baseline that brings the performance of the MIM pretraining to a new level with the AdPE. The experiments demonstrate that our approach can improve the fine-tuning accuracy of MAE by 0.8% and 0.4% over 1600 epochs of pretraining ViT-B and ViT-L on Imagenet1K. For the transfer learning task, it outperforms the MAE with the ViT-B backbone by 2.6% in mIoU on ADE20K, and by 3.2% in AP<sup>box</sup> and 1.6% in AP<sup>mask</sup> on COCO, respectively. These results are obtained with the AdPE being a pure MIM approach that does not use any extra models or external datasets for pretraining.

## Installation  
CUDA version should be 10.1 or higher. 
### 1. [`Install git`](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) 
### 2. Clone the repository in your computer 
```
git clone git@github.com:maple-research-lab/AdPE.git && cd AdPE
```

### 3. Build dependencies.   
You have two options to install dependency on your computer:
#### 3.1 Install with pip and python(Ver 3.6.9).
##### 3.1.1[`install pip`](https://pip.pypa.io/en/stable/installing/).
##### 3.1.2  Install dependency in command line.
```
pip install -r requirements.txt --user
```
If you encounter any errors, you can install each library one by one:
```
pip install torch==1.8.1
pip install torchvision==0.9.1
pip install timm==0.3.2
pip install easydict
pip install opencv-python
pip install simplejson
pip install lvis
pip install tensorboard
pip install tensorboardX
```

#### 3.2 Install with anaconda
##### 3.2.1 [`install conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html). 
##### 3.2.2 Install dependency in command line
```
conda create -n AdPE python=3.7.1
conda activate AdPE
pip install -r requirements.txt 
```
Each time when you want to run my code, simply activate the environment by
```
conda activate AdPE
conda deactivate(If you want to exit) 
```

#### 4 Prepare the ImageNet dataset
##### 4.1 Download the [ImageNet2012 Dataset](http://image-net.org/challenges/LSVRC/2012/) under "./datasets/imagenet2012".
##### 4.2 Go to path "./datasets/imagenet2012/val"
##### 4.3 move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

## Usage

### 1. Absolute Position Embedding of MAE+

#### 1.1 MAE+ APE baseline
For batch-size of 4096 (gradient accumulation), we can run on a single machine of 8\*V100 32gb GPU with the following command:
```
python main.py  --pe_type=0 --adv_type=0 --norm=0  --batch_size 128 --accum_iter=4 \
 --input_size 112 --num_crop 4  --model adpe_vit_base_patch16  \
 --norm_pix_loss --mask_ratio 0.75  --epochs 400 --warmup_epochs 20 \
 --blr 1.5e-4 --weight_decay 0.05 --data_path [imagenet_path]  \
 --output_dir [output_dir] --log_dir [output_dir] --adv_lr=1.5e-4 \
 --world_size=1 --rank=0   --dist_url "tcp://localhost:10001"
```
Here [imagenet_path] is the imagenet directory, [output_dir] is the directory to save model weights and logs. 

To run 1600 epoch experiments, simply change "--epochs 1600 --warmup_epochs 40".

#### 1.2 MAE+ APE adversarial embedding
For batch-size of 4096 (gradient accumulation), we can run on a single machine of 8\*V100 32gb GPU with the following command:
```
python main.py  --pe_type=0 --adv_type=1 --norm=[norm_type]  --eps=[norm_eps] --batch_size 128 --accum_iter=4 \
 --input_size 112 --num_crop 4  --model adpe_vit_base_patch16  \
 --norm_pix_loss --mask_ratio 0.75  --epochs 400 --warmup_epochs 20 \
 --blr 1.5e-4 --weight_decay 0.05 --data_path [imagenet_path]  \
 --output_dir [output_dir] --log_dir [output_dir] --adv_lr=1.5e-4 \
 --world_size=1 --rank=0   --dist_url "tcp://localhost:10001"
```
Here [imagenet_path] is the imagenet directory, [output_dir] is the directory to save model weights and logs. 
[norm_type] specifies the norm type for PGD, 0 indicates no PGD, 1 indicates L-2 norm in PGD, 2 indicates L-inf norm in PGD.
[norm_eps] is the cutoff value. For the performance of different parameters, please refer to supplementary table A.2.

To run 1600 epoch experiments, simply change "--epochs 1600 --warmup_epochs 40". 

#### 1.3 MAE+ APE adversarial coordinate
For batch-size of 4096 (gradient accumulation), we can run on a single machine of 8\*V100 32gb GPU with the following command:
```
python main.py  --pe_type=0 --adv_type=2 --norm=[norm_type]  --eps=[norm_eps] --batch_size 128 --accum_iter=4 \
 --input_size 112 --num_crop 4  --model adpe_vit_base_patch16  \
 --norm_pix_loss --mask_ratio 0.75  --epochs 400 --warmup_epochs 20 \
 --blr 1.5e-4 --weight_decay 0.05 --data_path [imagenet_path]  \
 --output_dir [output_dir] --log_dir [output_dir] --adv_lr=1.5e-4 \
 --world_size=1 --rank=0   --dist_url "tcp://localhost:10001"
```
Here [imagenet_path] is the imagenet directory, [output_dir] is the directory to save model weights and logs. 
[norm_type] specifies the norm type for PGD, 0 indicates no PGD, 1 indicates L-2 norm in PGD, 2 indicates L-inf norm in PGD.
[norm_eps] is the cutoff value. For the performance of different parameters, please refer to supplementary table A.2.

To run 1600 epoch experiments, simply change "--epochs 1600 --warmup_epochs 40". 

## 2. Relative Position Embedding of MAE+
### 2.1 MAE+ RPE baseline
For batch-size of 4096 (gradient accumulation), we can run on a single machine of 8\*V100 32gb GPU with the following command:
```
python main.py  --pe_type=1 --adv_type=0 --norm=0  --batch_size 128 --accum_iter=4 \
 --input_size 112 --num_crop 4  --model adpe_vit_base_patch16  \
 --norm_pix_loss --mask_ratio 0.75  --epochs 400 --warmup_epochs 20 \
 --blr 1.5e-4 --weight_decay 0.05 --data_path [imagenet_path]  \
 --output_dir [output_dir] --log_dir [output_dir] --adv_lr=1.5e-4 \
 --world_size=1 --rank=0   --dist_url "tcp://localhost:10001"
```
Here [imagenet_path] is the imagenet directory, [output_dir] is the directory to save model weights and logs. 

To run 1600 epoch experiments, simply change "--epochs 1600 --warmup_epochs 40". 

#### 2.2 MAE+ RPE adversarial embedding
For batch-size of 4096 (gradient accumulation), we can run on a single machine of 8\*V100 32gb GPU with the following command:
```
python main.py  --pe_type=1 --adv_type=1 --norm=[norm_type]  --eps=[norm_eps] --batch_size 128 --accum_iter=4 \
 --input_size 112 --num_crop 4  --model adpe_vit_base_patch16  \
 --norm_pix_loss --mask_ratio 0.75  --epochs 400 --warmup_epochs 20 \
 --blr 1.5e-4 --weight_decay 0.05 --data_path [imagenet_path]  \
 --output_dir [output_dir] --log_dir [output_dir] --adv_lr=1.5e-4 \
 --world_size=1 --rank=0   --dist_url "tcp://localhost:10001"
```
Here [imagenet_path] is the imagenet directory, [output_dir] is the directory to save model weights and logs. 
[norm_type] specifies the norm type for PGD, 0 indicates no PGD, 1 indicates L-2 norm in PGD, 2 indicates L-inf norm in PGD.
[norm_eps] is the cutoff value. For the performance of different parameters, please refer to supplementary table A.2.

To run 1600 epoch experiments, simply change "--epochs 1600 --warmup_epochs 40". 

#### 2.3 MAE+ RPE adversarial coordinate
For batch-size of 4096 (gradient accumulation), we can run on a single machine of 8\*V100 32gb GPU with the following command:
```
python main.py  --pe_type=1 --adv_type=2 --norm=[norm_type]  --eps=[norm_eps] --batch_size 128 --accum_iter=4 \
 --input_size 112 --num_crop 4  --model adpe_vit_base_patch16  \
 --norm_pix_loss --mask_ratio 0.75  --epochs 400 --warmup_epochs 20 \
 --blr 1.5e-4 --weight_decay 0.05 --data_path [imagenet_path]  \
 --output_dir [output_dir] --log_dir [output_dir] --adv_lr=1.5e-4 \
 --world_size=1 --rank=0   --dist_url "tcp://localhost:10001"
```
Here [imagenet_path] is the imagenet directory, [output_dir] is the directory to save model weights and logs. 
[norm_type] specifies the norm type for PGD, 0 indicates no PGD, 1 indicates L-2 norm in PGD, 2 indicates L-inf norm in PGD.
[norm_eps] is the cutoff value. For the performance of different parameters, please refer to supplementary table A.2.

To run 1600 epoch experiments, simply change "--epochs 1600 --warmup_epochs 40". 

## 3 Finetune MAE+
### 3.1 Finetune APE of MAE+

### 3.2 Finetune RPE of MAE+
