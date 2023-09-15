import os
import jittor as jt
import jittor.nn as nn
import argparse
import shutil
from tqdm import tqdm
import numpy as np
from cluster.kmeans import Kmeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from cluster.hungarian import reAssignSingle
import json
from logging import getLogger
import src.resnet as resnet_model
from src.utils import bool_flag
import jittor.transform as transforms
from src.cl_da import claDataset
import src.pseudo_transforms as custom_transforms
import cv2
import math
import time
jt.flags.use_cuda = 1

from src.utils import (
    initialize_exp,
    restart_from_checkpoint,
    fix_random_seeds,
    AverageMeter,
    distributed_sinkhorn
)
from src.multicropdataset import MultiCropDataset
import src.resnet as resnet_models
from options import getOption

logger = getLogger()
parser = getOption()
def main():
    global args
    args = parser.parse_args()
    fix_random_seeds(args.seed)
    logger, training_stats = initialize_exp(args, "epoch", "loss")
    model = resnet_models.__dict__[args.arch](
        normalize=True,
        hidden_mlp=0,
        output_dim=0,
        nmb_prototypes=0,
        train_mode='pixelattn'
    )
    if jt.in_mpi:
        for n, p in model.named_parameters():
            p.assign(p.mpi_broadcast())
    # build data
    normalize = transforms.ImageNormalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
    train_dataset = claDataset(
        args.data_path,
        # custom_transforms.Compose([
        #     custom_transforms.RandomResizedCropSemantic(224),
        #     custom_transforms.RandomHorizontalFlipSemantic(),
        #     custom_transforms.ToTensorSemantic(),
        #     normalize,
        # ]),
        transform=transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        pseudo_path='./weights/pass50/pixel_classification/train_clean_ccy_e36.txt',
    )
    train_loader = train_dataset.set_attrs(
        batch_size=args.batch_size,
        num_workers=8,
        drop_last=True,
        shuffle=True
    )
    print('Building data done with {} images loaded.',len(train_dataset))

    # for pixel attention, only finetuning the attention head and prototypes
    # for name, param in model.named_parameters():
    #     if 'fc' not in name:
    #         param.requires_grad = False
    #     else:
    #         print(name)
    # # loading pretrained weights
    for name, param in model.named_parameters():
        print(name,param.requires_grad)
    checkpoint = jt.load(args.pretrained)["state_dict"]
    for k in list(checkpoint.keys()):
        if k not in model.state_dict().keys():
            del checkpoint[k]
    model.load_state_dict(checkpoint)
    print("=> loaded model '{}'".format(args.pretrained))
    # copy model to GPU
    if jt.rank == 0:
        logger.info(model)
    logger.info("Building model done.")

    # build optimizer
    optimizer = jt.optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=args.wd,
    )
    warmup_lr_schedule = np.linspace(args.start_warmup, args.base_lr, len(train_loader) * args.warmup_epochs)
    iters = np.arange(len(train_loader) * (args.epochs - args.warmup_epochs))
    cosine_lr_schedule = np.array([args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (1 + \
                   math.cos(math.pi * t /(len(train_loader)* (args.epochs - args.warmup_epochs)))) for t in iters])
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
    logger.info("Building optimizer done.")
    # optionally resume from a checkpoint
    to_restore = {"epoch": 0}
    restart_from_checkpoint(
        os.path.join(args.dump_path, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=model,
        optimizer=optimizer,
    )
    start_epoch = to_restore["epoch"]
    for epoch in range(start_epoch, args.epochs):
        # train the network for one epoch
        logger.info('============ Starting epoch %i ... ============' % epoch)
        # train the network
        scores = train(train_loader, model, optimizer, epoch,lr_schedule)
        training_stats.update(scores)

        # save checkpoints
        if jt.rank == 0:
            save_dict = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            jt.save(
                save_dict,
                os.path.join(args.dump_path, 'checkpoint.pth.tar'),
            )
            # if epoch % args.checkpoint_freq == 0 or epoch == args.epochs - 1:
            if epoch % 1 == 0 or epoch == args.epochs - 1:
                shutil.copyfile(
                    os.path.join(args.dump_path, 'checkpoint.pth.tar'),
                    os.path.join(args.dump_checkpoints,
                                 'ckp-' + str(epoch) + '.pth.tar'),
                )
        jt.sync_all()

def train(train_loader, model, optimizer, epoch, lr_schedule):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    end = time.time()
    # if epoch == 10:
    #     for name, param in model.named_parameters():
    #         if "fbg" in name:
    #             param.requires_grad = True
    #         if 'fc' in name:
    #             param.requires_grad =False

    for it, (inputs, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # update learning rate
        iteration = epoch * len(train_loader) + it
        for param_group in optimizer.param_groups:
            if 'lr_scale' in param_group:
                param_group['lr'] = lr_schedule[iteration] * param_group['lr_scale']
            else:
                param_group['lr'] = lr_schedule[iteration]
        # print('inputs',inputs.shape)
        # print('labels',labels)
        emb = model(inputs, mode="clssification_clu")
        # print('emb',emb)
        # print('labels',labels)
        loss_1 = nn.cross_entropy_loss(emb, labels,reduction=False)
        # print('loss_1',loss_1)
        for i in range(len(loss_1)):
            min_idx = i
            for j in range(i + 1, len(loss_1)):
                if loss_1[min_idx] > loss_1[j]:
                    min_idx = j

            loss_1[i], loss_1[min_idx] = loss_1[min_idx], loss_1[i]
        if epoch<1000:
            forget_rate=0
        else:
            forget_rate=0.2
        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(loss_1))
        loss_2=loss_1[:num_remember]
        loss = loss_2.sum()
        loss = loss / num_remember

        # loss3=0
        # for i in range(len(loss_1)):
        #     if i<num_remember:
        #         loss3=loss3+loss_1[i]
        #     else:
        #         loss3 = loss3 + loss_1[i]*0.5
        # loss=loss3/len(loss_1)


        # print('loss_2',loss_2)

        # print('loss',loss)

        # ============ backward and optim step ... ============
        optimizer.step(loss)
        # ============ misc ... ============
        losses.update(loss.item(), inputs[0].size(0))
        # acc.update(acc1.item(), inputs[0].size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if jt.rank == 0 and it % 50 == 0:
            logger.info('Epoch: [{0}][{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Lr: {lr:.8f}'.format(
                epoch,
                it,
                batch_time=batch_time,
                data_time=data_time,
                loss=losses,
                lr=optimizer.param_groups[-1]['lr'],
            ))
    return (epoch, losses.avg)



if __name__ == "__main__":
    fix_random_seeds()
    main()
