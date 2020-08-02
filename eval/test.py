import argparse
import os
import re
import sys
import time

from tensorboardX import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, '../utils')
sys.path.insert(0, '../backbone')
from dataset_3d_lc import UCF101_3d, HMDB51_3d
from model_3d_lc import *
from resnet_2d3d import neq_load_customized
from augmentation import *
from utils import AverageMeter, ConfusionMeter, save_checkpoint, write_log, calc_topk_accuracy, calc_accuracy, \
    write_out_images
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils import data
import torch.nn as nn
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument('--net', default='resnet18', type=str)
parser.add_argument('--model', default='lc_cont', type=str)
parser.add_argument('--dataset', default='hmdb51', type=str)
parser.add_argument('--split', default=1, type=int)
parser.add_argument('--seq_len', default=30, type=int)
parser.add_argument('--num_seq', default=1, type=int)
parser.add_argument('--num_class', default=51, type=int)
parser.add_argument('--dropout', default=0.5, type=float)
parser.add_argument('--ds', default=1, type=int)
parser.add_argument('--representation_size', default=128, type=int)
parser.add_argument('--batch_size', default=10, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--wd', default=1e-3, type=float, help='weight decay')
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--pretrain',
                    default='/home/david/workspaces/cvhci/DPC/dpc/training_logs/2020-07-30_21-35-11_training_exp-000/model/model_min_val_loss_ep235.pth.tar',
                    type=str)
parser.add_argument('--test', default='', type=str)
parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--gpu', default=[0], type=int, nargs='+')
parser.add_argument('--print_freq', default=5, type=int)
parser.add_argument('--reset_lr', action='store_true', help='Reset learning rate when resume training?')
parser.add_argument('--train_what', default='last_only', type=str, help='Train what parameters?')
parser.add_argument('--prefix', default='tmp_cont', type=str)
parser.add_argument('--img_dim', default=128, type=int)
parser.add_argument('--num_workers', default=16, type=int)

global start_time
global stop_time


def main():
    global args;
    args = parser.parse_args()

    print("Startup parameters:")
    print(args)
    print()

    os.environ[
        "CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # NVIDIA-SMI uses PCI_BUS_ID device order, but CUDA orders graphics devices by speed by default (fastest first).
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(id) for id in args.gpu])

    print('Cuda visible devices: {}'.format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print('Available device count: {}'.format(torch.cuda.device_count()))

    args.gpu = list(range(
        torch.cuda.device_count()))  # Really weird: In Pytorch 1.2, the device ids start from 0 on the visible devices.

    print(
        "Note: At least in Pytorch 1.2, device ids are reindexed on the visible devices and not the same as in nvidia-smi.")

    for i in args.gpu:
        print("Using Cuda device {}: {}".format(i, torch.cuda.get_device_name(i)))
    print("Cuda is available: {}".format(torch.cuda.is_available()))

    global cuda
    cuda = torch.device('cuda')

    if args.dataset == 'ucf101':
        args.num_class = 101
    elif args.dataset == 'hmdb51':
        args.num_class = 51

    print("Using dataset {}".format(args.dataset))

    ### classifier model ###
    if args.model == 'lc':
        model = LC(sample_size=args.img_dim,
                   num_seq=args.num_seq,
                   seq_len=args.seq_len,
                   network=args.net,
                   num_class=args.num_class,
                   dropout=args.dropout)
    elif args.model == "lc_cont":
        model = Resnet18Classifier(sample_size=args.img_dim,
                                   num_seq=args.num_seq,
                                   seq_len=args.seq_len,
                                   network=args.net,
                                   num_class=args.num_class,
                                   dropout=args.dropout,
                                   crossm_vector_length=args.representation_size
                                   )
    else:
        raise ValueError('wrong model!')

    print("Using model {}".format(model.__class__))
    model = nn.DataParallel(model)
    model = model.to(cuda)
    global criterion;
    criterion = nn.CrossEntropyLoss()

    ### optimizer ### 
    params = None
    if args.train_what == 'ft':
        print('=> finetune backbone with smaller lr')
        params = []
        for name, param in model.module.named_parameters():
            if ('resnet' in name) or ('rnn' in name):
                params.append({'params': param, 'lr': args.lr / 10})
            else:
                params.append({'params': param})
    elif args.train_what == 'last_only':
        print('=> only train last layers')
        params = []
        print("=======Only training the following parameters:=======")
        for name, param in model.module.named_parameters():
            if ('resnet' in name) or ('rnn' in name):
                pass
            else:
                params.append({'params': param})
                print(name)

    else:
        pass  # train all layers

    print('\n===========Check Grad============')
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    print('=================================\n')

    if params is None: params = model.parameters()

    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    if args.dataset == 'hmdb51':
        lr_lambda = lambda ep: MultiStepLR_Restart_Multiplier(ep, gamma=0.1, step=[150, 250, 300], repeat=1)
    elif args.dataset == 'ucf101':
        if args.img_dim == 224:
            lr_lambda = lambda ep: MultiStepLR_Restart_Multiplier(ep, gamma=0.1, step=[300, 400, 500], repeat=1)
        else:
            lr_lambda = lambda ep: MultiStepLR_Restart_Multiplier(ep, gamma=0.1, step=[60, 80, 100], repeat=1)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    args.old_lr = None
    best_acc = 0
    global iteration;
    iteration = 0

    ### restart training ###
    if args.test:
        if os.path.isfile(args.test):
            print("=> loading testing checkpoint '{}'".format(args.test))
            checkpoint = torch.load(args.test)
            try:
                model.load_state_dict(checkpoint['state_dict'])
            except:
                print('=> [Warning]: weight structure is not equal to test model; Use non-equal load ==')
                model = neq_load_customized(model, checkpoint['state_dict'])
            print("=> loaded testing checkpoint '{}' (epoch {})".format(args.test, checkpoint['epoch']))
            global num_epoch;
            num_epoch = checkpoint['epoch']
        elif args.test == 'random':
            print("=> [Warning] loaded random weights")
        else:
            raise ValueError()

        transform = transforms.Compose([
            RandomSizedCrop(consistent=True, size=224, p=0.0),
            Scale(size=(args.img_dim, args.img_dim)),
            ToTensor(),
            Normalize()
            ])
        test_loader = get_data(transform, 'test')
        test_loss, test_acc = test(test_loader, model)
        sys.exit()
    else:  # not test
        pass
        # torch.backends.cudnn.benchmark = True

    if args.resume:
        if os.path.isfile(args.resume):
            args.old_lr = float(re.search('_lr(.+?)_', args.resume).group(1))
            print("=> loading resumed checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            if not args.reset_lr:  # if didn't reset lr, load old optimizer
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                print('==== Change lr from %f to %f ====' % (args.old_lr, args.lr))
            iteration = checkpoint['iteration']
            print("=> loaded resumed checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if (not args.resume) and args.pretrain:
        if args.pretrain == 'random':
            print('=> using random weights')
        elif os.path.isfile(args.pretrain):
            print("=> loading pretrained checkpoint '{}'".format(args.pretrain))
            checkpoint = torch.load(args.pretrain, map_location=torch.device('cpu'))
            model.module.load_weights_state_dict(checkpoint['state_dict'], model=model)
            print("=> loaded pretrained checkpoint '{}' (epoch {})".format(args.pretrain, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrain))

    ### load data ###
    transform = transforms.Compose([
        RandomSizedCrop(consistent=True, size=224, p=1.0),
        Scale(size=(args.img_dim, args.img_dim)),
        RandomHorizontalFlip(consistent=True),
        ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=0.3, consistent=True),
        ToTensor(),
        Normalize()
        ])
    val_transform = transforms.Compose([
        RandomSizedCrop(consistent=True, size=224, p=0.3),
        Scale(size=(args.img_dim, args.img_dim)),
        RandomHorizontalFlip(consistent=True),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3, consistent=True),
        ToTensor(),
        Normalize()
        ])

    train_loader = get_data(transform, 'train')
    val_loader = get_data(val_transform, 'val')

    # setup tools
    global de_normalize;
    global img_path;
    img_path, model_path = set_path(args)
    global writer_train
    try:  # old version
        writer_val = SummaryWriter(log_dir=os.path.join(img_path, 'val'))
        writer_train = SummaryWriter(log_dir=os.path.join(img_path, 'train'))
    except:  # v1.7
        writer_val = SummaryWriter(logdir=os.path.join(img_path, 'val'))
        writer_train = SummaryWriter(logdir=os.path.join(img_path, 'train'))

    ### main loop ###
    for epoch in range(args.start_epoch, args.epochs):
        train_loss, train_acc = train(train_loader, model, optimizer, epoch)
        val_loss, val_acc = validate(val_loader, model)
        scheduler.step(epoch)

        writer_train.add_scalar('global/loss', train_loss, epoch)
        writer_train.add_scalar('global/accuracy', train_acc, epoch)
        writer_val.add_scalar('global/loss', val_loss, epoch)
        writer_val.add_scalar('global/accuracy', val_acc, epoch)

        # save check_point
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        save_checkpoint({
            'epoch':      epoch + 1,
            'net':        args.net,
            'state_dict': model.state_dict(),
            'best_acc':   best_acc,
            'optimizer':  optimizer.state_dict(),
            'iteration':  iteration
            }, is_best, model_path=model_path)

    print('Training from ep %d to ep %d finished' % (args.start_epoch, args.epochs))


def train(data_loader, model, optimizer, epoch):
    data_loading_times = []
    cuda_transfer_times = []
    calculation_times = []
    losses = AverageMeter()
    accuracy = AverageMeter()
    model.train()
    global iteration
    global start_time
    global stop_time

    start_time = time.perf_counter()

    for idx, (vid_seq, target) in enumerate(data_loader):
        tic = time.time()

        stop_time = time.perf_counter()  # Timing data loading
        data_loading_times.append(stop_time - start_time)

        start_time = time.perf_counter()  # Timing cuda transfer

        vid_seq = vid_seq.to(cuda)

        target = target.to(cuda)

        stop_time = time.perf_counter()

        cuda_transfer_times.append(stop_time - start_time)

        start_time = time.perf_counter()  # Timing calculation

        B = vid_seq.size(0)
        output = model(vid_seq)

        # Visualize images for tensorboard.
        if iteration == 0:
            write_out_images(vid_seq, writer_train, iteration, img_dim=args.img_dim)
        del vid_seq

        [_, N, D] = output.size()
        output = output.view(B * N, D)
        target = target.repeat(1, N).view(-1)

        loss = criterion(output, target)
        acc = calc_accuracy(output, target)

        losses.update(loss.item(), B)
        accuracy.update(acc.item(), B)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del target, loss

        stop_time = time.perf_counter()

        calculation_times.append(stop_time - start_time)

        if idx % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.local_avg:.4f})\t'
                  'Acc: {acc.val:.4f} ({acc.local_avg:.4f}) T:{3:.2f}\t'.format(
                epoch, idx, len(data_loader), time.time() - tic,
                loss=losses, acc=accuracy))

            total_weight = 0.0
            decay_weight = 0.0
            for m in model.parameters():
                if m.requires_grad: decay_weight += m.norm(2).data
                total_weight += m.norm(2).data
            print('Decay weight / Total weight: %.3f/%.3f' % (decay_weight, total_weight))

            writer_train.add_scalar('local/loss', losses.val, iteration)
            writer_train.add_scalar('local/accuracy', accuracy.val, iteration)

            iteration += 1

            start_time = time.perf_counter()

    print("Avg t input loading: {:.4f}; Avg t input to cuda: {:.4f}; Avg t calculation: {:.4f}".format(
        sum(data_loading_times) / len(data_loading_times), sum(cuda_transfer_times) / len(cuda_transfer_times),
        sum(calculation_times) / len(calculation_times)
        ))

    return losses.local_avg, accuracy.local_avg


def validate(data_loader, model):
    losses = AverageMeter()
    accuracy = AverageMeter()
    model.eval()
    with torch.no_grad():
        for idx, (input_seq, target) in tqdm(enumerate(data_loader), total=len(data_loader)):
            input_seq = input_seq.to(cuda)
            target = target.to(cuda)
            B = input_seq.size(0)
            output = model(input_seq)

            [_, N, D] = output.size()
            output = output.view(B * N, D)
            target = target.repeat(1, N).view(-1)

            loss = criterion(output, target)
            acc = calc_accuracy(output, target)

            losses.update(loss.item(), B)
            accuracy.update(acc.item(), B)

    print('Loss {loss.avg:.4f}\t'
          'Acc: {acc.avg:.4f} \t'.format(loss=losses, acc=accuracy))
    return losses.avg, accuracy.avg


def test(data_loader, model):
    losses = AverageMeter()
    acc_top1 = AverageMeter()
    acc_top5 = AverageMeter()
    confusion_mat = ConfusionMeter(args.num_class)
    model.eval()
    with torch.no_grad():
        for idx, (input_seq, target) in tqdm(enumerate(data_loader), total=len(data_loader)):
            input_seq = input_seq.to(cuda)
            target = target.to(cuda)
            B = input_seq.size(0)
            input_seq = input_seq.squeeze(0)  # squeeze the '1' batch dim
            output = model(input_seq)
            del input_seq
            top1, top5 = calc_topk_accuracy(torch.mean(
                torch.mean(
                    nn.functional.softmax(output, 2),
                    0), 0, keepdim=True),
                target, (1, 5))

            acc_top1.update(top1.item(), B)
            acc_top5.update(top5.item(), B)
            del top1, top5

            output = torch.mean(torch.mean(output, 0), 0, keepdim=True)
            loss = criterion(output, target.squeeze(-1))

            losses.update(loss.item(), B)
            del loss

            _, pred = torch.max(output, 1)
            confusion_mat.update(pred, target.view(-1).byte())

    print('Loss {loss.avg:.4f}\t'
          'Acc top1: {top1.avg:.4f} Acc top5: {top5.avg:.4f} \t'.format(loss=losses, top1=acc_top1, top5=acc_top5))
    confusion_mat.plot_mat(args.test + '_confm_test_split_{}.svg'.format(args.split))
    write_log(content='Loss {loss.avg:.4f}\t Acc top1: {top1.avg:.4f} Acc top5: {top5.avg:.4f} \t'.format(loss=losses,
                                                                                                          top1=acc_top1,
                                                                                                          top5=acc_top5,
                                                                                                          args=args),
              epoch=num_epoch,
              filename=args.test + '_log_test_split_{}.md'.format(args.split))
    return losses.avg, [acc_top1.avg, acc_top5.avg]


def get_data(transform, mode='train'):
    print('Loading data for "%s" ...' % mode)
    global dataset
    if args.dataset == 'ucf101':
        dataset = UCF101_3d(mode=mode,
                            transform=transform,
                            seq_len=args.seq_len,
                            num_seq=args.num_seq,
                            downsample=args.ds,
                            which_split=args.split)
    elif args.dataset == 'hmdb51':
        dataset = HMDB51_3d(mode=mode,
                            transform=transform,
                            seq_len=args.seq_len,
                            num_seq=args.num_seq,
                            downsample=args.ds,
                            which_split=args.split)
    else:
        raise ValueError('dataset not supported')
    my_sampler = data.RandomSampler(dataset)
    if mode == 'train':
        data_loader = data.DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      sampler=my_sampler,
                                      shuffle=False,
                                      num_workers=args.num_workers,
                                      pin_memory=True,
                                      drop_last=True)
    elif mode == 'val':
        data_loader = data.DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      sampler=my_sampler,
                                      shuffle=False,
                                      num_workers=args.num_workers,
                                      pin_memory=True,
                                      drop_last=True)
    elif mode == 'test':
        data_loader = data.DataLoader(dataset,
                                      batch_size=1,
                                      sampler=my_sampler,
                                      shuffle=False,
                                      num_workers=args.num_workers,
                                      pin_memory=True)
    print('"%s" dataset size: %d' % (mode, len(dataset)))
    return data_loader


def set_path(args):
    if args.resume:
        exp_path = os.path.dirname(os.path.dirname(args.resume))
    else:
        exp_path = 'transfer_logs/{time}_log_{args.prefix}/{args.dataset}-{args.img_dim}-\
sp{args.split}_{0}_{args.model}_bs{args.batch_size}_\
lr{1}_wd{args.wd}_ds{args.ds}_seq{args.num_seq}_len{args.seq_len}_\
dp{args.dropout}_train-{args.train_what}{2}'.format(
            'r%s' % args.net[6::], \
            args.old_lr if args.old_lr is not None else args.lr, \
            '_pt=' + 'pretrained_net' if args.pretrain != "random" else 'untrained_net', \
            args=args, time=datetime.now().strftime("%Y%m%d%H%M%S"))
    img_path = os.path.join(exp_path, 'img')
    model_path = os.path.join(exp_path, 'model')
    if not os.path.exists(img_path): os.makedirs(img_path)
    if not os.path.exists(model_path): os.makedirs(model_path)
    return img_path, model_path


def MultiStepLR_Restart_Multiplier(epoch, gamma=0.1, step=[10, 15, 20], repeat=3):
    '''return the multipier for LambdaLR, 
    0  <= ep < 10: gamma^0
    10 <= ep < 15: gamma^1 
    15 <= ep < 20: gamma^2
    20 <= ep < 30: gamma^0 ... repeat 3 cycles and then keep gamma^2'''
    max_step = max(step)
    effective_epoch = epoch % max_step
    if epoch // max_step >= repeat:
        exp = len(step) - 1
    else:
        exp = len([i for i in step if effective_epoch >= i])
    return gamma ** exp


if __name__ == '__main__':
    main()
