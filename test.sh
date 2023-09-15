CUDA='GPU-023e2b32-30d5-c1f7-2f06-fc2517b6deb9'
N_GPU=1
BATCH=128
DUMP_PATH=./weights/pass50
DATA=/farm/litingting/jittor/ImageNetS50/
DUMP_PATH_SEG_2=${DUMP_PATH}/pixel_finetuning_2
DUMP_PATH_SEG_384=${DUMP_PATH}/pixel_finetuning_384
ARCH=resnet18
#由下游训练权重推出测试集result并生成label.txt标签和label_fen.txt标签
CUDA_VISIBLE_DEVICES=${CUDA} python inference1.py \
-a ${ARCH} \
--pretrained ${DUMP_PATH_SEG_2}/checkpoints/ckp-42.pth.tar \
--data_path ${DATA} \
--dump_path ./ \
-c 50 \
--mode test \
--match_file ${DUMP_PATH_SEG_2}/match.json

#根据label.txt统一图片像素标签
python gai.py

#对result进行crf后处理，结果放到result_crf中
python crf.py

#对result_crf进行sam细化边缘操作，结果放到result中
python ./SAM_ReFine/main.py --pseudo_path ./result_crf --sam_path ./sam/test

#下面是由另一个下游训练权重推出一个差不多分的测试集result2并生成label2.txt标签和label2_fen.txt标签，接下来根据label1_fen.txt和label2_fen.txt的分数融合result和result2
CUDA_VISIBLE_DEVICES=${CUDA} python inference2.py \
-a ${ARCH} \
--pretrained ${DUMP_PATH_SEG_384}/checkpoints/ckp-42.pth.tar \
--data_path ${DATA} \
--dump_path ./ \
-c 50 \
--mode test \
--match_file ${DUMP_PATH_SEG_384}/match.json

python tongji_test.py
python gai2.py
python crf2.py
python ./SAM_ReFine/main.py --pseudo_path ./result_crf2 --sam_path ./sam/test
python gai_clean.py
python gai3.py


