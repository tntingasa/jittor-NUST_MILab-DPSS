# PASS的使用
[安装](#1)

[训练](#2)

[评估](#3)

<div id="1"></div>

# 安装
本模型的环境要求有两个：
# 1、进行crf操作的专属环境
* System: **Linux**(e.g. Ubuntu/CentOS/Arch), **macOS**, or **Windows Subsystem of Linux (WSL)**
* Python version == 3.6
* CPU compiler (require at least one of the following)
    * g++ (>=5.4.0)
    * clang (>=8.0)
## 安装依赖
```shell
python -m pip install tqdm
python -m pip install pillow
python -m pip install opencv-python
python -m pip install pydensecrf
python -m pip install scikit-image
python -m pip install multiprocess
```
# 2、进行除crf外操作的环境
* System: **Linux**(e.g. Ubuntu/CentOS/Arch), **macOS**, or **Windows Subsystem of Linux (WSL)**
* Python version >= 3.7
* CPU compiler (require at least one of the following)
    * g++ (>=5.4.0)
    * clang (>=8.0)

## 第一步: 安装计图
计图的安装可以参考以下文档[Jittor install](https://github.com/Jittor/jittor#install)

## 第二步: 安装依赖
```shell
python -m pip install scikit-learn
python -m pip install pandas
python -m pip install munkres
python -m pip install tqdm
python -m pip install pillow
python -m pip install opencv-python
python -m pip install faiss-gpu
python -m pip install scipy
python -m pip install matplotlib
pip3 install torch torchvision torchaudio
```

# 训练
我们提供了训练脚本[train.sh](./train.sh)，下文解释了训练脚本中每个部分的功能。

## 步骤1：无监督的表征学习
首先进行非对比像素到像素表示对齐和深度到浅层监督进行预训练，对比baseline更改了部分参数。
```shell
CUDA_VISIBLE_DEVICES=${CUDA} mpirun -np ${N_GPU} --allow-run-as-root python ./main_pretrain.py \
--arch ${ARCH} \
--data_path ${DATA}/train \
--dump_path ${DUMP_PATH} \
--nmb_crops 2 \
--size_crops 224  \
--min_scale_crops 0.08  \
--max_scale_crops 1. \
--crops_for_assign 0 1 \
--temperature 0.1 \
--epsilon 0.05 \
--sinkhorn_iterations 3 \
--feat_dim 128 \
--hidden_mlp ${HIDDEN_DIM} \
--nmb_prototypes ${NUM_PROTOTYPE} \
--queue_length ${QUEUE_LENGTH} \
--epoch_queue_starts 15 \
--epochs ${EPOCH} \
--batch_size ${BATCH} \
--base_lr 0.6 \
--final_lr 0.0006  \
--freeze_prototypes_niters ${FREEZE_PROTOTYPES} \
--wd 0.000001 \
--warmup_epochs 0 \
--workers 10 \
--seed 31 \
--shallow 3 \
--weights 1 1
```

## 步骤2：使用像素注意力生成像素标签
### 步骤2.1：微调像素注意力
在这一部分中，您应该将"--pretrained"设置为步骤1中获得的第400个预训练权重。
```shell

CUDA_VISIBLE_DEVICES=${CUDA} mpirun -np ${N_GPU} --allow-run-as-root python ./main_pixel_attention.py \
--arch ${ARCH} \
--data_path ${IMAGENETS}/train \
--dump_path ${DUMP_PATH_FINETUNE} \
--nmb_crops 2 \
--size_crops 224 \
--min_scale_crops 0.08 \
--max_scale_crops 1. \
--crops_for_assign 0 1 \
--temperature 0.1 \
--epsilon 0.05 \
--sinkhorn_iterations 3 \
--feat_dim 128 \
--hidden_mlp ${HIDDEN_DIM} \
--nmb_prototypes ${NUM_PROTOTYPE} \
--queue_length ${QUEUE_LENGTH_PIXELATT} \
--epoch_queue_starts 0 \
--epochs ${EPOCH_PIXELATT} \
--batch_size ${BATCH} \
--base_lr 0.6 \
--final_lr 0.0006  \
--freeze_prototypes_niters ${FREEZE_PROTOTYPES_PIXELATT} \
--wd 0.000001 \
--warmup_epochs 0 \
--workers 10 \
--seed 31 \
--pretrained ${DUMP_PATH}/checkpoints/ckp-400.pth.tar
```

### 步骤2.2：聚类
由于"--pretrained"和"--data_path"的改进，可以得到更好的聚类中心\
实验中一共用到三个聚类中心\
聚类中心1：\
将"--pretrained"设置为步骤2.1中获得的预训练权重。\
将"--data_path"设置为所有的训练图片。\
在本部分中，每个聚类的中心将生成并保存在`${DUMP_PATH_FINETUNE}/cluster/centroids.npy`中。
```shell
CUDA_VISIBLE_DEVICES=${CUDA} python cluster.py -a ${ARCH} \
--pretrained ${DUMP_PATH_FINETUNE}/checkpoint.pth.tar \
--data_path ${IMAGENETS}/train \
--dump_path ${DUMP_PATH_FINETUNE} \
-c 50 \
--seed 31
```
聚类中心2：\
将"--pretrained"设置为步骤3.1中获得的训练权重。\
将"--data_path"设置为所有的训练图片。\
在本部分中，每个聚类的中心将生成并保存在`${DUMP_PATH_FINETUNE_1}/cluster/centroids.npy`中。
```shell
CUDA_VISIBLE_DEVICES=${CUDA} python cluster.py -a ${ARCH} \
--pretrained ${DUMP_PATH_classification}/ckp-11.pth.tar \
--data_path ${IMAGENETS}/train \
--dump_path ${DUMP_PATH_FINETUNE_1} \
-c 50 \
--seed 31
```
聚类中心3:\
将"--pretrained"设置为步骤3.1中获得的训练权重。\
将"--data_path"设置为train_42865_0.866_ccy数据集,train_42865_0.866_ccy数据集是对train根据离聚类中心距离进行筛选并重采样得到的。\
train_42865_0.866_ccy文件夹由于文件大小限制，没有放入文件夹中，可以根据train_42865_0.866_ccy.txt得到数据集中的图片\
生成train_42865_0.866_ccy文件夹，运行createtrain.py\
修改createtrain.py中的"root"为训练集
```shell
python createtrain.py
```
在本部分中，每个聚类的中心将生成并保存在`${DUMP_PATH_FINETUNE_2}/cluster/centroids.npy`中。
```shell
CUDA_VISIBLE_DEVICES=${CUDA} python cluster.py -a ${ARCH} \
--pretrained ${DUMP_PATH_classification}/ckp-11.pth.tar \
--data_path ${DUMP_PATH_classification}/train_42865_0.866_ccy \
--dump_path ${DUMP_PATH_FINETUNE_2} \
 -c 50 \
 --seed 31
 ```
### 步骤2.3：选择生成伪标签的阈值。
“centroid”是一个保存聚类中心的npy文件。并且“pretrained”应该被设置为在步骤3.1中获得的训练权重。

在此步骤中，将显示不同阈值下的val mIoUs。
```shell
CUDA_VISIBLE_DEVICES=${CUDA} python ./inference_pixel_attention.py -a ${ARCH} \
--pretrained ${DUMP_PATH_classification}/ckp-11.pth.tar \
--data_path ${IMAGENETS} \
--dump_path ${DUMP_PATH_FINETUNE_2} \
-c ${NUM_CLASSES} \
--mode validation \
--test \
--centroid ${DUMP_PATH_FINETUNE_1}/cluster/centroids.npy

CUDA_VISIBLE_DEVICES=${CUDA} python ./evaluator.py \
--predict_path ${DUMP_PATH_FINETUNE_2} \
--data_path ${IMAGENETS} \
-c ${NUM_CLASSES} \
--mode validation \
--curve \
--min 30 \
--max 60 > result.txt
```

### 步骤2.4：为训练集生成伪标签
请将“t”设置为在步骤2.3中获得的最佳阈值。\
“pretrained”应该被设置为在步骤3.1中获得的训练权重。
```shell
CUDA_VISIBLE_DEVICES=${CUDA}  python ./inference_pixel_attention_1.py -a ${ARCH} \
--pretrained ${DUMP_PATH_classification}/ckp-11.pth.tar \
--data_path ${IMAGENETS} \
--dump_path ${DUMP_PATH_FINETUNE_1} \
--dump_path_1 ${DUMP_PATH_FINETUNE_1}/logit_max_1/ \
--testPoint_path ${DUMP_PATH_FINETUNE_1}/trainPoint.txt \
--testLabel_path ${DUMP_PATH_FINETUNE_1}/trainLabel.txt \
-c ${NUM_CLASSES} \
--mode train \
--centroid ${DUMP_PATH_FINETUNE_2}/cluster/centroids.npy \
-t 0.42
```
## 步骤3：分类去噪网络
### 步骤3.1：用于生成聚类的预训练权重 ${DUMP_PATH_classification}/ckp-11.pth.tar
train_clean_ccy_e36文件夹由于文件大小限制，可以根据clean_ccy_e36.txt得到\
生成train_42865_0.866_ccy文件夹，运行createtrain2.py\
修改createtrain2.py中的"root"为训练集
```shell
python createtrain2.py
```
train_clean_ccy_e36是先用所有图片训练分类网络，用第15个epoch得到train的标签并根据损失筛选出损失小的图片\
根据去噪筛选损失小的图片和相应标签再去训练分类网络，得到最好的权重./weights/pass50/pixel_classification/checkpoints/ckp-36.pth.tar,可以生成标签，\
并筛选损失小的图片标签得到train_clean_ccy_e36训练集和相应的标签文件
```shell
CUDA_VISIBLE_DEVICES=${CUDA} python classification_clu.py -a ${ARCH} \
--pretrained ${DUMP_PATH_FINETUNE}/checkpoint.pth.tar  \
--data_path ${DUMP_PATH_classification}/train_clean_ccy_e36 \
--dump_path ${DUMP_PATH_classification} \
--epochs 70 \
--batch_size 32 \
--base_lr 0.01 \
--final_lr 0.0006 \
--wd 0.000001 \
--warmup_epochs 0 \
--workers 8 \
--num_classes 50
```
### 步骤3.2：用于生成去噪后的标签
ccy_newclu_qzclu_43636_0.8867/train文件夹由于文件大小限制，可以根据ccy_newclu_qzclu_43636_0.8867/train.txt得到\
生成ccy_newclu_qzclu_43636_0.8867/train文件夹，运行createtrain3.py\
修改createtrain3.py中的"root"为训练集
```shell
python createtrain3.py
```
训练前需要修改./src/cl_da.py中的self.img_label为./weights/pass50/pixel_classification/ccy_newclu_qzclu_43636_0.8867/train_2.txt\
修改self.data_dir为./weights/pass50/pixel_classification/ccy_newclu_qzclu_43636_0.8867/train\
ccy_newclu_qzclu_43636_0.8867/train训练集是由聚类2根据与聚类中心距离筛选得到的
```shell
CUDA_VISIBLE_DEVICES=${CUDA} python classification.py -a ${ARCH} \
--pretrained ${DUMP_PATH_FINETUNE}/checkpoint.pth.tar  \
--data_path ${DUMP_PATH_classification}/ccy_newclu_qzclu_43636_0.8867/train \
--dump_path ${DUMP_PATH_classification} \
--epochs 70 \
--batch_size 32 \
--base_lr 0.01 \
--final_lr 0.0006 \
--wd 0.000001 \
--warmup_epochs 0 \
--workers 8 \
--num_classes 50
```
### 步骤3.3：根据去噪网络推出训练集标签
"--pretrained"被设置为由3.2得到的分类去噪网络权重
```shell
CUDA_VISIBLE_DEVICES=${CUDA} python inference_classification.py \
-a ${ARCH} \
--pretrained ${DUMP_PATH_classification}/checkpoints/ckp-58.pth.tar \
--data_path ${IMAGENETS}  \
--dump_path  ${DUMP_PATH_classification} \
-c 50 \
--mode train
```

## 步骤4：生成第一轮分割训练的伪标签
### 步骤4.1：选择生成伪标签的阈值
使用聚类中心2产生train跟validation
“centroid”是一个保存聚类中心的npy文件。并且“pretrained”应该被设置为在步骤3.1中获得的预训练权重。
在此步骤中，将显示不同阈值下的val mIoUs，并保存到./result.txt中。
```shell
CUDA_VISIBLE_DEVICES=${CUDA} python ./inference_pixel_attention.py -a ${ARCH} \
--pretrained ${DUMP_PATH_classification}/ckp-11.pth.tar \
--data_path ${IMAGENETS} \
--dump_path ${DUMP_PATH_FINETUNE_1} \
-c ${NUM_CLASSES} \
--mode validation \
--test \
--centroid ${DUMP_PATH_FINETUNE_1}/cluster/centroids.npy

CUDA_VISIBLE_DEVICES=${CUDA} python ./evaluator.py \
--predict_path ${DUMP_PATH_FINETUNE_1} \
--data_path ${IMAGENETS} \
-c ${NUM_CLASSES} \
--mode validation \
--curve \
--min 30 \
--max 60 > result.txt
```
### 步骤4.2：为训练集生成伪标签
请将“t”设置为在步骤4.1中获得的最佳阈值，logit_max_1中将存放标记最亮点的图片，trainPoint.txt上记录了train的最亮点坐标。
```shell
CUDA_VISIBLE_DEVICES=${CUDA}  python ./inference_pixel_attention_1.py -a ${ARCH} \
--pretrained ${DUMP_PATH_classification}/ckp-11.pth.tar \
--data_path ${IMAGENETS} \
--dump_path ${DUMP_PATH_FINETUNE_1} \
--dump_path_1 ${DUMP_PATH_FINETUNE_1}/logit_max_1/ \
--testPoint_path ${DUMP_PATH_FINETUNE_1}/trainPoint.txt \
--testLabel_path ${DUMP_PATH_FINETUNE_1}/trainLabel.txt \
-c ${NUM_CLASSES} \
--mode train \
--centroid ${DUMP_PATH_FINETUNE_2}/cluster/centroids.npy \
-t 0.42
```
### 步骤4.3：对伪标签进行后处理操作
```shell
python ./crf_pe.py \
--data_path ${IMAGENETS} \
--predict_path ${DUMP_PATH_FINETUNE_1} \
--dump_path ${DUMP_PATH_FINETUNE_1} \
--mode train

python ./SAM_ReFine/main.py \
--pseudo_path ${DUMP_PATH_FINETUNE_1}/train_crf \
--sam_path ./sam/train
```
### 步骤4.4：利用步骤4.2产生的trainPoint.txt跟步骤4.3产生的train生成sam-mask
```shell
CUDA_VISIBLE_DEVICES=${CUDA} python ./SAM-jittor/amg_box_point.py \
--checkpoint ./SAM-jittor/checkpoint/sam_vit_b_01ec64.pth \
--model vit_b \
--input  ${IMAGENETS}/train \
--output ${DUMP_PATH_FINETUNE_1}/sam-train  \
--mask ${DUMP_PATH_FINETUNE_1}/train \
--txt1_path ${DUMP_PATH_FINETUNE_1}/trainPoint.txt
```
### 步骤4.5：利用聚类中心2产生的去噪后的train_label赋值步骤4.4产生的sam-mask
```shell
python fuzhi.py
```
## 步骤5：进行一轮分割训练
### 步骤5.1
请将“pseudo_path”设置为步骤4.5中生成的伪标签的路径。
```shell
CUDA_VISIBLE_DEVICES=${CUDA} python ./main_pixel_finetuning.py \
--arch ${ARCH} \
--data_path ${DATA}/train \
--dump_path ${DUMP_PATH_SEG} \
--epochs ${EPOCH_SEG} \
--batch_size ${BATCH} \
--base_lr 0.6 \
--final_lr 0.0006 \
--wd 0.000001 \
--warmup_epochs 0 \
--workers 8 \
--num_classes ${NUM_CLASSES} \
--pseudo_path  ${DUMP_PATH_FINETUNE_1}/sam-train \
--pretrained ${DUMP_PATH}/checkpoints/ckp-400.pth.tar \
--seed 31
```
### 步骤5.2：挑选出最佳权重
```shell
CUDA_VISIBLE_DEVICES=${CUDA} python ./inference.py -a ${ARCH} \
--pretrained ${DUMP_PATH_SEG}/checkpoint.pth.tar \
--data_path ${IMAGENETS} \
--dump_path ${DUMP_PATH_SEG} \
-c ${NUM_CLASSES} \
--mode validation \
--match_file ${DUMP_PATH_SEG}/validation/match.json

CUDA_VISIBLE_DEVICES=${CUDA} python ./evaluator.py \
--predict_path ${DUMP_PATH_SEG} \
--data_path ${IMAGENETS} \
-c ${NUM_CLASSES} \
--mode validation
```
### 步骤5.3：使用步骤5.3中的最佳权重推理得到train
```shell
CUDA_VISIBLE_DEVICES=${CUDA} python ./inference.py -a ${ARCH} \
--pretrained ${DUMP_PATH_SEG}/checkpoints/ckp-36.pth.tar \
--data_path ${IMAGENETS} \
--dump_path ${DUMP_PATH_SEG} \
-c ${NUM_CLASSES} \
--mode train \
--match_file ${DUMP_PATH_SEG}/validation/match.json
```
### 步骤5.4：利用步骤4.2产生的trainPoint.txt跟步骤5.3产生的train生成sam-mask
```shell
CUDA_VISIBLE_DEVICES=${CUDA} python ./SAM-jittor/amg_box_point_3.py \
--checkpoint ./SAM-jittor/checkpoint/sam_vit_b_01ec64.pth \
--model vit_b \
--input  ${IMAGENETS}/train \
--output ${DUMP_PATH_SEG}/sam-train  \
--mask ${DUMP_PATH_SEG}/train \
--txt1_path ${DUMP_PATH_FINETUNE_1}/trainPoint.txt
```
### 步骤5.5：
对步骤5.4产生的sam-mask选取最佳mask
利用聚类中心3产生的去噪后的train_label赋值步骤5.4产生的sam-mask
```shell
python fenzu.py
python fuzhi_2.py
```

## 步骤6：进行二轮分割训练
### 步骤6.1：
请将“pseudo_path”设置为步骤5.5中生成的伪标签的路径。
```shell
CUDA_VISIBLE_DEVICES=${CUDA} python ./main_pixel_finetuning.py \
--arch ${ARCH} \
--data_path ${DATA}/train \
--dump_path ${DUMP_PATH_SEG_2} \
--epochs ${EPOCH_SEG} \
--batch_size ${BATCH} \
--base_lr 0.6 \
--final_lr 0.0006 \
--wd 0.000001 \
--warmup_epochs 0 \
--workers 8 \
--num_classes ${NUM_CLASSES} \
--pseudo_path  ${DUMP_PATH_SEG}/sam-train-3/train_1 \
--pretrained ${DUMP_PATH}/checkpoints/ckp-400.pth.tar \
--seed 31
```
### 步骤6.2：推理
如果想评估测试集的性能，请将“mode”设置为“test”，从而生成对应的标签。
```shell
CUDA_VISIBLE_DEVICES=${CUDA} python ./inference.py -a ${ARCH} \
--pretrained ${DUMP_PATH_SEG_2}/checkpoints/ckp-42.pth.tar \
--data_path ${IMAGENETS} \
--dump_path ${DUMP_PATH_SEG_2} \
-c ${NUM_CLASSES} \
--mode validation \
--match_file ${DUMP_PATH_SEG_2}/validation/match.json
```
<div id="2"></div>

# 评估
## Fully unsupervised protocol
```shell
CUDA_VISIBLE_DEVICES=${CUDA} python ./evaluator.py \
--predict_path ${DUMP_PATH_SEG_2} \
--data_path ${IMAGENETS} \
-c ${NUM_CLASSES} \
--mode validation
```
