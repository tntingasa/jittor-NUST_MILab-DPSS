CUDA='GPU-023e2b32-30d5-c1f7-2f06-fc2517b6deb9'
N_GPU=1
BATCH=128
DATA=/farm/litingting/jittor/ImageNetS50/
IMAGENETS=/farm/litingting/jittor/ImageNetS50
DUMP_PATH=./weights/pass50
DUMP_PATH_FINETUNE=${DUMP_PATH}/pixel_attention
DUMP_PATH_FINETUNE_1=${DUMP_PATH}/pixel_attention_1
DUMP_PATH_FINETUNE_2=${DUMP_PATH}/pixel_attention_2
DUMP_PATH_classification=${DUMP_PATH}/pixel_classification
DUMP_PATH_SEG=${DUMP_PATH}/pixel_finetuning_1
DUMP_PATH_SEG_2=${DUMP_PATH}/pixel_finetuning_2
QUEUE_LENGTH=2048
QUEUE_LENGTH_PIXELATT=3840
HIDDEN_DIM=512
NUM_PROTOTYPE=500
ARCH=resnet18
NUM_CLASSES=50
EPOCH=1000
EPOCH_PIXELATT=20
EPOCH_SEG=50
FREEZE_PROTOTYPES=1001
FREEZE_PROTOTYPES_PIXELATT=0

mkdir -p ${DUMP_PATH_FINETUNE}
mkdir -p ${DUMP_PATH_FINETUNE_1}
mkdir -p ${DUMP_PATH_FINETUNE_2}
mkdir -p ${DUMP_PATH_classification}
mkdir -p ${DUMP_PATH_SEG}
mkdir -p ${DUMP_PATH_SEG_2}
export DISABLE_MULTIPROCESSING=1

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

#一共用到三个聚类中心
#聚类中心1
CUDA_VISIBLE_DEVICES=${CUDA} python cluster.py -a ${ARCH} \
--pretrained ${DUMP_PATH_FINETUNE}/checkpoint.pth.tar \
--data_path ${IMAGENETS}/train \
--dump_path ${DUMP_PATH_FINETUNE} \
-c 50 \
--seed 31

#聚类中心2
#聚类预训练权重是根据分类网络去噪得到的
CUDA_VISIBLE_DEVICES=${CUDA} python cluster.py -a ${ARCH} \
--pretrained ${DUMP_PATH_classification}/ckp-11.pth.tar \
--data_path ${IMAGENETS}/train \
--dump_path ${DUMP_PATH_FINETUNE_1} \
-c 50 \
--seed 31

#聚类中心3
#train_42865_0.866_ccy数据集是对train根据离聚类中心距离进行筛选并重采样得到的
CUDA_VISIBLE_DEVICES=${CUDA} python cluster.py -a ${ARCH} \
--pretrained ${DUMP_PATH_classification}/ckp-11.pth.tar \
--data_path ${DUMP_PATH_classification}/train_42865_0.866_ccy \
--dump_path ${DUMP_PATH_FINETUNE_2} \
 -c 50 \
 --seed 31

#分类去噪网络  用于生成聚类的预训练权重
#train_clean_ccy_e36是根据去噪筛选损失小的图片和相应标签
CUDA_VISIBLE_DEVICES=${CUDA} python classification_clu.py --arch ${ARCH} \
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

#分类去噪网络  用于生成去噪后的标签
#训练前需要修改./src/cl_da.py中的self.img_label为./weights/pass50/pixel_classification/ccy_newclu_qzclu_43636_0.8867/train_2.txt
#修改self.data_dir为./weights/pass50/pixel_classification/ccy_newclu_qzclu_43636_0.8867/train
CUDA_VISIBLE_DEVICES=${CUDA} python classification.py --arch ${ARCH} \
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

#根据去噪网络推出训练集标签
CUDA_VISIBLE_DEVICES=${CUDA} python inference_classification.py \
--a ${ARCH} \
--pretrained ${DUMP_PATH_classification}/checkpoints/ckp-58.pth.tar \
--data_path ${IMAGENETS}  \
--dump_path  ${DUMP_PATH_classification} \
-c 50 \
--mode train

#使用聚类中心2产生train跟validation
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

CUDA_VISIBLE_DEVICES=${CUDA}  python ./inference_pixel_attention_1.py -a ${ARCH} \
--pretrained ${DUMP_PATH_classification}/ckp-11.pth.tar \
--data_path ${IMAGENETS} \
--dump_path ${DUMP_PATH_FINETUNE_1} \
--dump_path_1 ${DUMP_PATH_FINETUNE_1}/logit_max_1/ \
--testPoint_path ${DUMP_PATH_FINETUNE_1}/trainPoint.txt \
--testLabel_path ${DUMP_PATH_FINETUNE_1}/trainLabel.txt \
-c ${NUM_CLASSES} \
--mode train \
--centroid ${DUMP_PATH_FINETUNE_1}/cluster/centroids.npy \
-t 0.42


python ./crf_pe.py \
--data_path ${IMAGENETS} \
--predict_path ${DUMP_PATH_FINETUNE_1} \
--dump_path ${DUMP_PATH_FINETUNE_1} \
--mode train

python ./SAM_ReFine/main.py \
--pseudo_path ${DUMP_PATH_FINETUNE_1}/train_crf \
--sam_path ./sam/train

CUDA_VISIBLE_DEVICES=${CUDA} python ./SAM-jittor/amg_box_point.py \
--checkpoint ./SAM-jittor/checkpoint/sam_vit_b_01ec64.pth \
--model vit_b \
--input  ${IMAGENETS}/train \
--output ${DUMP_PATH_FINETUNE_1}/sam-train  \
--mask ${DUMP_PATH_FINETUNE_1}/train \
--txt1_path ${DUMP_PATH_FINETUNE_1}/trainPoint.txt

python fuzhi.py

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

#挑选出最佳权重生成train
CUDA_VISIBLE_DEVICES=${CUDA} python ./inference.py -a ${ARCH} \
--pretrained ${DUMP_PATH_SEG}/checkpoints/ckp-36.pth.tar \
--data_path ${IMAGENETS} \
--dump_path ${DUMP_PATH_SEG} \
-c ${NUM_CLASSES} \
--mode train \
--match_file ${DUMP_PATH_SEG}/validation/match.json

CUDA_VISIBLE_DEVICES=${CUDA} python ./SAM-jittor/amg_box_point_3.py \
--checkpoint ./SAM-jittor/checkpoint/sam_vit_b_01ec64.pth \
--model vit_b \
--input  ${IMAGENETS}/train \
--output ${DUMP_PATH_SEG}/sam-train  \
--mask ${DUMP_PATH_SEG}/train \
--txt1_path ${DUMP_PATH_FINETUNE_1}/trainPoint.txt

python fenzu.py

python fuzhi_2.py

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

CUDA_VISIBLE_DEVICES=${CUDA} python ./inference.py -a ${ARCH} \
--pretrained ${DUMP_PATH_SEG_2}/checkpoints/ckp-42.pth.tar \
--data_path ${IMAGENETS} \
--dump_path ${DUMP_PATH_SEG_2} \
-c ${NUM_CLASSES} \
--mode validation \
--match_file ${DUMP_PATH_SEG_2}/validation/match.json

CUDA_VISIBLE_DEVICES=${CUDA} python ./evaluator.py \
--predict_path ${DUMP_PATH_SEG_2} \
--data_path ${IMAGENETS} \
-c ${NUM_CLASSES} \
--mode validation
