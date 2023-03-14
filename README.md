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

#### 4 Install RPE acceleration.
Please see README in "rpe_ope" to install the acceleration for RPE calculation.

#### 5 Prepare the ImageNet dataset
##### 5.1 Download the [ImageNet2012 Dataset](http://image-net.org/challenges/LSVRC/2012/) under "./datasets/imagenet2012".
##### 5.2 Go to path "./datasets/imagenet2012/val"
##### 5.3 move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

## Usage

### 1. Absolute Position Embedding of MAE+
All APE should be finetuned via --pe_type=0 in finetune script.

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

RPE should be finetuned via --pe_type=1 in finetune script.

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

RPE should be finetuned via --pe_type=1 in finetune script.

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

Interpolated RPE should be finetuned via --pe_type=2 in finetune script.

To run 1600 epoch experiments, simply change "--epochs 1600 --warmup_epochs 40". 

## 3 Multi Machine Pre-Training
This can only be run with multiple machines. Limited by our computing resources, we run experiments with 4096 on 4 matchines of 8*V100 GPU.
Here we use MAE+ APE baseline as an example:

On the first node machine, run the following command:
```
python main.py  --pe_type=1 --adv_type=0 --norm=0  --batch_size 128 --accum_iter=4 \
 --input_size 112 --num_crop 4  --model adpe_vit_base_patch16  \
 --norm_pix_loss --mask_ratio 0.75  --epochs 400 --warmup_epochs 20 \
 --blr 1.5e-4 --weight_decay 0.05 --data_path [imagenet_path]  \
 --output_dir [output_dir] --log_dir [output_dir] --adv_lr=1.5e-4 \
 --world_size=4 --rank=0   --dist_url "tcp://localhost:10001"
```
Here [imagenet_path] is the imagenet directory, [output_dir] is the directory to save model weights and logs.

On the other 3 nodes, run the following command:
```
python main.py  --pe_type=1 --adv_type=0 --norm=0  --batch_size 128 --accum_iter=4 \
 --input_size 112 --num_crop 4  --model adpe_vit_base_patch16  \
 --norm_pix_loss --mask_ratio 0.75  --epochs 400 --warmup_epochs 20 \
 --blr 1.5e-4 --weight_decay 0.05 --data_path [imagenet_path]  \
 --output_dir [output_dir] --log_dir [output_dir] --adv_lr=1.5e-4 \
 --world_size=4 --rank=[rank_id]   --dist_url "tcp://[master_ip]:10001"
```
Here we should change [master_ip] to the IP of the 1st node, also we should adjust rank with 1, 2, and 3 for 3 additional nodes.

## 4 Finetune MAE+
Here we used vit_base_patch16 to illustrate the command. For finetuning vit-l and vit-h, please check MAE paper for detailed finetune parameter changes, we followed the same hyper-param change. For learning rate, please always sweep from 1e-4 to 1e-3 to find a better learning rate with better finetune results.

### 4.1 Finetune APE of MAE+
For batch-size of 1024 (gradient accumulation), we can run on a single machine of 8\*V100 32gb GPU with the following command:
```
python3 main_finetune.py --accum_iter 1 --pe_type 0  \
    --batch_size 128 --model vit_base_patch16 --finetune [model_path] \
    --output_dir [output_dir] --log_dir [output_dir] \
    --epochs 100 --blr [base_learning_rate] --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 \
    --reprob 0.25 --dist_eval --data_path [imagenet_path] \
    --world_size=1 --rank=0   --dist_url "tcp://localhost:10001"
```
Here [model_path] is the pre-trained model path,[imagenet_path] is the imagenet directory, [output_dir] is the directory to save model weights and logs,
[base_learning_rate] is the base learning rate for finetune, please try sweep lr from 1e-4 to 1e-3 to find a suitable lr for finetuning. 

For ViT-L and ViT-H, please check the hyper-parameter in MAE to make adjustment for parameters such as --layer_decay.

### 4.2 Finetune RPE of MAE+
For batch-size of 4096 (gradient accumulation), we can run on a single machine of 8\*V100 32gb GPU with the following command:
```
python3 main_finetune.py --accum_iter 4 --pe_type 1  \
    --batch_size 128 --model vit_base_patch16 --finetune [model_path] \
    --output_dir [output_dir] --log_dir [output_dir] \
    --epochs 100 --blr [base_learning_rate] --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 \
    --reprob 0.25 --dist_eval --data_path [imagenet_path] \
    --world_size=1 --rank=0   --dist_url "tcp://localhost:10001"
```
Here [model_path] is the pre-trained model path,[imagenet_path] is the imagenet directory, [output_dir] is the directory to save model weights and logs,
[base_learning_rate] is the base learning rate for finetune, please try sweep lr from 1e-4 to 1e-3 to find a suitable lr for finetuning. 

For ViT-L and ViT-H, please check the hyper-parameter in MAE to make adjustment for parameters such as --layer_decay.


### 4.3 Finetune RPE of MAE+
For batch-size of 4096 (gradient accumulation), we can run on a single machine of 8\*V100 32gb GPU with the following command:
```
python3 main_finetune.py --accum_iter 4 --pe_type 2  \
    --batch_size 128 --model vit_base_patch16 --finetune [model_path] \
    --output_dir [output_dir] --log_dir [output_dir] \
    --epochs 100 --blr [base_learning_rate] --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 \
    --reprob 0.25 --dist_eval --data_path [imagenet_path] \
    --world_size=1 --rank=0   --dist_url "tcp://localhost:10001"
```
Here [model_path] is the pre-trained model path,[imagenet_path] is the imagenet directory, [output_dir] is the directory to save model weights and logs,
[base_learning_rate] is the base learning rate for finetune, please try sweep lr from 1e-4 to 1e-3 to find a suitable lr for finetuning. 

For ViT-L and ViT-H, please check the hyper-parameter in MAE to make adjustment for parameters such as --layer_decay.

# Performance
### Vit-B performance
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Method </th>
<th valign="bottom">PE Type</th>
<th valign="bottom">Adv Type</th>
<th valign="bottom">FT Top-1</th>
<th valign="bottom">Model<br/>Link</th>
<!-- TABLE BODY -->
<tr><td align="left">MAE+</td>
<td align="center">APE</td>
<td align="center">-</td>
<td align="center">83.9</td>
<td align="center"><a href="https://purdue0-my.sharepoint.com/:u:/g/personal/wang3702_purdue_edu/EVyDG3ZhpqpEkgPLQOBDSvsBga7kPdlt_MuGNQmcIu-KDA?e=Pc5gdW">model</a></td>
</tr>
<tr><td align="left">AdPE</td>
<td align="center">APE</td>
<td align="center">Adv Embed</td>
<td align="center">84.1</td>
<td align="center"><a href="https://purdue0-my.sharepoint.com/:u:/g/personal/wang3702_purdue_edu/EUrmQsBkOAlGhDxA7Ft2_S0B4cFmDAdfK9tWhpTwT-AuYQ?e=TtHU0g">model</a></td>
</tr>
<tr><td align="left">AdPE</td>
<td align="center">APE</td>
<td align="center">Adv Coord</td>
<td align="center">84.2</td>
<td align="center"><a href="https://purdue0-my.sharepoint.com/:u:/g/personal/wang3702_purdue_edu/EfV8LVMq99BKgJSliu9dCcEB-8MRkiiWCwCqDgylpt-m2A?e=Ms6vzB">model</a></td>
</tr>
<tr><td align="left">MAE+</td>
<td align="center">RPE</td>
<td align="center">-</td>
<td align="center">84.2</td>
<td align="center"><a href="https://purdue0-my.sharepoint.com/:u:/g/personal/wang3702_purdue_edu/EYdJwg-IXmhKlVVlmAhZEGUBLtXfh8zYEty0JumTRe0uDQ?e=XDNESa">model</a></td>
</tr>
<tr><td align="left">AdPE</td>
<td align="center">RPE</td>
<td align="center">Adv Embed</td>
<td align="center">84.3</td>
<td align="center"><a href="https://purdue0-my.sharepoint.com/:u:/g/personal/wang3702_purdue_edu/ETOhQDvABH1DtraWqjSzbjgB9aua7urwABRSTcI9zDInsA?e=cEkKTN">model</a></td>
</tr>
<tr><td align="left">AdPE</td>
<td align="center">RPE</td>
<td align="center">Adv Coord</td>
<td align="center">84.4</td>
<td align="center"><a href="https://purdue0-my.sharepoint.com/:u:/g/personal/wang3702_purdue_edu/EVy4z40ivhRNn_cRHeJtUuoBw0ul_i5EswU-NQqjCn_GVA?e=73bdBd">model</a></td>
</tr>

</tbody></table>

### Final Performance
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">ViT-Base</th>
<th valign="bottom">ViT-Large</th>
<th valign="bottom">ViT-Huge</th>
<!-- TABLE BODY -->
<tr><td align="left">pre-trained checkpoint</td>
<td align="center"><a href="https://purdue0-my.sharepoint.com/:u:/g/personal/wang3702_purdue_edu/EVy4z40ivhRNn_cRHeJtUuoBw0ul_i5EswU-NQqjCn_GVA?e=73bdBd">download</a></td>
<td align="center"><a href="https://purdue0-my.sharepoint.com/:u:/g/personal/wang3702_purdue_edu/ERC011ZVEeNMt8jqPl5pOSoBSgI6IpZjwy9up92NWRlQHw?e=4T8b6w">download</a></td>
<td align="center"><a href="">download</a></td>
</tr>
<tr><td align="left">FT Top-1</td>
<td align="center"><tt>84.4</tt></td>
<td align="center"><tt>86.3</tt></td>
<td align="center"><tt>waiting</tt></td>
</tr>
</tbody></table>

## Downstream Tasks
### 1 ADE20K Segmentation 
#### 1.1 Please install [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) following their instructions.
#### 1.2 Please follow [instructions](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/mae) for ADE20K segmentation.
#### 1.3 Run segmentation with configurations.
```
bash tools/dist_train.sh \
[config_path] \
 $GPUS  --options optimizer.lr=[learning_rate] \
 --work-dir [output_dir] --deterministic
```
[config_path] specifies the configuration file  "segmentation/upernet_mae-base_fp16_512x512_160k_ade20k_ms.py" in (https://github.com/open-mmlab/mmsegmentation/tree/master/configs/mae).
[learning_rate] should be specified as 2e-4 and it should be searched in [5e-5, 3e-4].

### 2 COCO detection and instance segmentation
#### 2.1 Install [detectron2](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md) and follow its instructions in preparing datasets.
#### 2.2 Install additional dependencies.
```
pip install setuptools==59.5.0
pip install shapely
```
#### 2.3 Follow [instructions](https://github.com/facebookresearch/detectron2/tree/main/projects/ViTDet) for training the vit detection on COCO.
#### 2.4 Run detectron2 with our configuration in "detection" dir.
Run on a GPU machine with 8 A100 GPUs.
```
python tools/lazyconfig_train_net.py \
--config-file [config_path] --num-gpus=8 \
OUTPUT_DIR=[output_dir] train.init_checkpoint=[model_path] \
  dataloader.train.total_batch_size=32 \
   train.max_iter=400000
```
Here [config_path] specifies the configuration file, should be "detection/cascade_mask_rcnn_vitdet_b_100ep.py".
[output_dir] is the directory to save model weights and logs.
[mode_path] is the pre-trained model path.
For learning rate, please change lr_multiplier.scheduler.values in "detection/mask_rcnn_vitdet_b_100ep.py" for a more suitable learning rate.

## Citation:
[AdPE: Adversarial Positional Embeddings for Pretraining Vision Transformers via MAE+]().  
```
@article{wang2022adpe,
  title={AdPE: Adversarial Positional Embeddings for Pretraining Vision Transformers via MAE+},
  author={Wang, Xiao and Wang, Ying and Xuan, Ziwei and Qi, Guo-Jun},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence (submitted)},
  year={2023}
}
```
