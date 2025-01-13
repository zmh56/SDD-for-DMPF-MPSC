# -*- coding: utf-8 -*-
# ref in https://github.com/YuanGongND/ast
# @Time    : 13/1/25
# @Author  : Minghui Zhao
# @Affiliation  : Southeast university


import argparse
import os
import ast
import pickle
import sys
import time
import torch
from torch.utils.data import WeightedRandomSampler
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
import dataloader
import models
import numpy as np
from traintest import train, validate,train_daic

fold = 5
fold = fold
acc_sum_snt_list = []
epoch_best = []
epoch_get = []
for i in range(fold):
    acc_sum_snt_list.append(1)
    epoch_best.append(0)
    epoch_get.append([])

dict_my = {'acc_sum_snt_list':acc_sum_snt_list, 'epoch_get':epoch_get, 'fold_i': 0, 'best_epoch':epoch_best, 'fold_all':fold}

for fold_i in range(fold):
    print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))

    train_list = f'train_{fold_i}.scp'
    val_list = f'val_{fold_i}.scp'
    exp_dir = f'./exp/fold{fold_i}'
    dict_my['fold_i'] = fold_i

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-train", type=str, default=train_list, help="training data json")
    parser.add_argument("--data-val", type=str, default=val_list, help="validation data json")
    parser.add_argument("--data-eval", type=str, default='', help="evaluation data json")
    parser.add_argument("--label-csv", type=str, default='', help="csv with class labels")
    parser.add_argument("--n_class", type=int, default=2, help="number of classes") #527
    parser.add_argument("--model", type=str, default='ast', help="the model used")
    parser.add_argument("--dataset", type=str, default="", help="the dataset used")

    parser.add_argument("--exp-dir", type=str, default=exp_dir, help="directory to dump experiments")
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, metavar='LR', help='initial learning rate') #0.001
    parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
    parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N', help='mini-batch size') #12
    parser.add_argument('-w', '--num-workers', default=0, type=int, metavar='NW', help='# of workers for dataloading (default: 32)')
    parser.add_argument("--n-epochs", type=int, default=20, help="number of maximum training epochs") #1
    # not used in the formal experiments
    parser.add_argument("--lr_patience", type=int, default=2, help="how many epoch to wait to reduce lr if mAP doesn't improve")

    parser.add_argument("--n-print-steps", type=int, default=100, help="number of steps to print statistics")
    parser.add_argument('--save_model', default=False, help='save the model or not', type=ast.literal_eval)

    parser.add_argument('--freqm', help='frequency mask max length', type=int, default=24) #0
    parser.add_argument('--timem', help='time mask max length', type=int, default=96)#0
    parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
    parser.add_argument("--bal", type=str, default=None, help="use balanced sampling or not")
    # the stride used in patch spliting, e.g., for patch size 16*16, a stride of 16 means no overlapping, a stride of 10 means overlap of 6.
    parser.add_argument("--fstride", type=int, default=10, help="soft split freq stride, overlap=patch_size-stride")
    parser.add_argument("--tstride", type=int, default=10, help="soft split time stride, overlap=patch_size-stride")
    parser.add_argument('--imagenet_pretrain', help='if use ImageNet pretrained audio spectrogram transformer model', type=ast.literal_eval, default='True')
    parser.add_argument('--audioset_pretrain', help='if use ImageNet and audioset pretrained audio spectrogram transformer model', type=ast.literal_eval, default='True')

    parser.add_argument("--dataset_mean", type=float, default=-6.6268077, help="the dataset spectrogram mean")#-4.2677393
    parser.add_argument("--dataset_std", type=float, default=5.358466, help="the dataset spectrogram std")#4.5689974
    parser.add_argument("--audio_length", type=int, default=512, help="the dataset spectrogram std")#1024
    parser.add_argument('--noise', help='if augment noise', type=ast.literal_eval, default='False')

    parser.add_argument("--metrics", type=str, default="acc", help="evaluation metrics", choices=["acc", "mAP"])#None
    parser.add_argument("--loss", type=str, default="CE", help="loss function", choices=["BCE", "CE"])#None
    parser.add_argument('--warmup', help='if warmup the learning rate', type=ast.literal_eval, default='False')
    parser.add_argument("--lrscheduler_start", type=int, default=5, help="which epoch to start reducing the learning rate")#2
    parser.add_argument("--lrscheduler_step", type=int, default=1, help="how many epochs as step to reduce the learning rate")
    parser.add_argument("--lrscheduler_decay", type=float, default=0.85, help="the learning rate decay rate at each step")#0.5

    parser.add_argument('--wa', help='if weight averaging', type=ast.literal_eval, default='False')
    parser.add_argument('--wa_start', type=int, default=1, help="which epoch to start weight averaging the checkpoint model")
    parser.add_argument('--wa_end', type=int, default=5, help="which epoch to end weight averaging the checkpoint model")

    args = parser.parse_args()

    # transformer based model
    if args.model == 'ast':
        print('now train a audio spectrogram transformer model')

        audio_conf = {'num_mel_bins': 128, 'target_length': args.audio_length, 'freqm': args.freqm, 'timem': args.timem, 'mixup': args.mixup, 'dataset': args.dataset, 'mode':'train', 'mean':args.dataset_mean, 'std':args.dataset_std,
                      'noise':args.noise}
        val_audio_conf = {'num_mel_bins': 128, 'target_length': args.audio_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset, 'mode':'evaluation', 'mean':args.dataset_mean, 'std':args.dataset_std, 'noise':False}

        if args.bal == 'bal':
            print('balanced sampler is being used')
            samples_weight = np.loadtxt(args.data_train[:-5]+'_weight.csv', delimiter=',')
            sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

            train_loader = torch.utils.data.DataLoader(
                dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
                batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True)
        else:
            print('balanced sampler is not used')
            train_loader = torch.utils.data.DataLoader(
                dataloader.AudiosetDataset(dataset_json_file=args.data_train, dataset_name=args.dataset,label_csv=args.label_csv, audio_conf=audio_conf, audio_class=args.n_class),
                batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            dataloader.AudiosetDataset(args.data_val, dataset_name=args.dataset, label_csv=args.label_csv, audio_conf=val_audio_conf, audio_class=args.n_class),
            batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        audio_model = models.ASTModel(label_dim=args.n_class, fstride=args.fstride, tstride=args.tstride, input_fdim=128,
                                      input_tdim=args.audio_length, imagenet_pretrain=args.imagenet_pretrain,
                                      audioset_pretrain=args.audioset_pretrain, model_size='base384')

    print("\nCreating experiment directory: %s" % args.exp_dir)
    if not os.path.exists(f"{args.exp_dir}"):
        os.makedirs(f"{args.exp_dir}")

    print('Now starting training for {:d} epochs'.format(args.n_epochs))
    train_daic(audio_model, train_loader, val_loader, args, dict_my)



tem_num = float('inf')
sum_err = []
for list_tem in dict_my['epoch_get']:
    if len(list_tem) < tem_num:
        tem_num = len(list_tem)
for i in range(tem_num):
    aaa_tem = 0
    for list_tem in dict_my['epoch_get']:
        aaa_tem = aaa_tem + list_tem[i]
    sum_err.append(aaa_tem/fold)

with open(args.exp_dir+"/result.csv", "a") as res_file:
    arrray_err = np.array(dict_my['acc_sum_snt_list']).mean()
    res_file.write(f"last-fold={dict_my['acc_sum_snt_list']}---{arrray_err}\n")
    print(sum_err)
    res_file.write(f"epoch err ={sum_err}\n")

