import argparse
import os
import re
import sys
import time
from datetime import datetime

import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils import data
from tqdm import tqdm

sys.path.insert(0, '../utils')
sys.path.insert(0, '../backbone')
sys.path.insert(0, '../datasets')

from augmentation import *
from dataset_hmdb51 import HMDB51Dataset
from dataset_ucf101 import UCF101Dataset
from model_3d_lc import *
from resnet_2d3d import neq_load_customized
from utils import AverageMeter, ConfusionMeter, save_checkpoint, write_log, calc_topk_accuracy, calc_accuracy, \
    write_out_images

parser = argparse.ArgumentParser()
parser.add_argument('--prefix', default='tmp_r21d', type=str)

parser.add_argument('--gpu', default=[0], type=int, nargs='+')
parser.add_argument('--num_workers', default=16, type=int)

parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
parser.add_argument('--batch_size', default=1, type=int)

parser.add_argument('--dataset', default='hmdb51', type=str)
parser.add_argument('--split', default=1, type=int)
parser.add_argument('--sampling_shift', default=None, type=int, help='Limit for subsamples from available samples.')
parser.add_argument('--max_samples', default=None, type=int, help='Limit for samples.')

parser.add_argument('--seq_len', default=30, type=int)
parser.add_argument('--ds', default=2, type=int)
parser.add_argument('--img_dim', default=224, type=int)

parser.add_argument('--model', default='lc_r2+1d', type=str, choices=["lc_cont", "lc_r2+1d"])
parser.add_argument('--net', default='r2+1d18', type=str)
parser.add_argument('--dropout', default=0.5, type=float)
parser.add_argument('--representation_size', default=512, type=int)
parser.add_argument('--num_class', default=51, type=int)

parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--wd', default=1e-3, type=float, help='weight decay')
parser.add_argument('--train_what', default='last_only', type=str, help='Train what parameters?')

parser.add_argument('--print_freq', default=5, type=int)

parser.add_argument('--pretrain',
                    default='/home/david/temp/training_logs/test_temp/2020-08-01_02-46-13_training_exp-300-r21dbc/model/model_best_ep13.pth.tar',
                    type=str)

parser.add_argument('--resume', default='', type=str)
parser.add_argument('--reset_lr', action='store_true', help='Reset learning rate when resume training?')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')

parser.add_argument('--test', default='', type=str)

global start_time
global stop_time


def main():
    global args
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
    elif args.model == "lc_r2+1d":
        model = R2plus1DClassifier(sample_size=args.img_dim,
                                   seq_len=args.seq_len,
                                   backbone=args.net,
                                   num_class=args.num_class,
                                   dropout=args.dropout,
                                   representation_size=args.representation_size
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
    global de_normalize
    global img_path
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
        train_loss, train_acc = train(train_loader, model, optimizer, epoch, args)
        val_loss, val_acc = validate(val_loader, model, epoch, args)
        scheduler.step(epoch)

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


def train(data_loader, model, optimizer, epoch, args):
    tr_stats = {"time_data_loading":  AverageMeter(locality=args.print_freq),
                "time_cuda_transfer": AverageMeter(locality=args.print_freq),
                "time_forward":       AverageMeter(locality=args.print_freq),
                "time_backward":      AverageMeter(locality=args.print_freq),
                "time_all":           AverageMeter(locality=args.print_freq),

                "cv_rep_sim_m":       AverageMeter(locality=args.print_freq),
                "cv_rep_sim_s":       AverageMeter(locality=args.print_freq),
                "cv_rand_sim_m":      AverageMeter(locality=args.print_freq),
                "cv_rand_sim_s":      AverageMeter(locality=args.print_freq),

                "total_loss":         AverageMeter(locality=args.print_freq),

                "accuracy":           AverageMeter(locality=args.print_freq)
                }

    model.train()
    global iteration
    global start_time
    global stop_time

    start_time = time.perf_counter()
    time_all = time.perf_counter()

    for idx, (vid_seq, target) in enumerate(data_loader):
        (batch_size, C, seq_len, H, W) = vid_seq.shape

        stop_time = time.perf_counter()  # Timing data loading
        tr_stats["time_data_loading"].update(stop_time - start_time)

        start_time = time.perf_counter()  # Timing cuda transfer

        vid_seq = vid_seq.to(cuda)
        target = target.to(cuda)

        stop_time = time.perf_counter()

        tr_stats["time_cuda_transfer"].update(stop_time - start_time)

        # Visualize images for tensorboard.
        if iteration == 0:
            write_out_images(vid_seq, writer_train, iteration, img_dim=args.img_dim)

        start_time = time.perf_counter()  # Timing calculation

        output = model(vid_seq)

        del vid_seq

        loss = criterion(output, target)
        acc = calc_accuracy(output, target)

        stop_time = time.perf_counter()

        tr_stats["time_forward"].update(stop_time - start_time)

        tr_stats["total_loss"].update(loss.item(), batch_size)
        tr_stats["accuracy"].update(acc.item(), batch_size)

        start_time = time.perf_counter()  # Timing calculation

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del target, loss

        stop_time = time.perf_counter()

        tr_stats["time_backward"].update(stop_time - start_time)

        stop_time = time.perf_counter()
        tr_stats["time_all"].update(stop_time - time_all)

        if idx % args.print_freq == 0:
            print(f'Epoch: [{epoch}][{idx}/{len(data_loader)}]\t'
                  f'Loss {tr_stats["total_loss"].local_avg:.4f}\t'
                  f'Acc: {tr_stats["accuracy"].local_avg:.4f} T:{tr_stats["time_all"].local_avg:.2f}\t')

            write_stats_transfer_iteration(tr_stats, writer_train, iteration)

        iteration += 1

        start_time = time.perf_counter()
        time_all = time.perf_counter()

    print_tr_stats_timings_avg(tr_stats)
    write_stats_transfer_epoch(tr_stats, writer_train, epoch, mode="Train")

    return tr_stats["total_loss"].avg, tr_stats["accuracy"].avg


def validate(data_loader, model, epoch, args):
    tr_stats = {"total_loss": AverageMeter(locality=args.print_freq),
                "accuracy":   AverageMeter(locality=args.print_freq)
                }

    model.eval()
    with torch.no_grad():
        for idx, (input_seq, target) in tqdm(enumerate(data_loader), total=len(data_loader)):
            input_seq = input_seq.to(cuda)
            target = target.to(cuda)
            batch_size = input_seq.size(0)
            output = model(input_seq)

            loss = criterion(output, target)
            acc = calc_accuracy(output, target)

            tr_stats["total_loss"].update(loss.item(), batch_size)
            tr_stats["accuracy"].update(acc.item(), batch_size)

    write_stats_transfer_epoch(tr_stats, writer_train, epoch, mode="Val")

    print(f'Epoch: [{epoch}] \t'
          f'Avg Val Loss {tr_stats["total_loss"].avg:.4f}\t'
          f'Avg Val Acc: {tr_stats["accuracy"].avg:.4f}\t')

    return tr_stats["total_loss"].avg, tr_stats["accuracy"].avg


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
                    nn.functional.softmax(output, 2), 0), 0, keepdim=True),
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


def write_stats_transfer_epoch(stats, writer, epoch, mode="Train"):
    writer.add_scalars('ep/Losses', {f'{mode} Loss': stats["total_loss"].avg}, epoch)

    writer.add_scalars('ep/Accuracy', {f'{mode} Acc': stats["accuracy"].avg}, epoch)


def write_stats_transfer_iteration(stats, writer, iteration):
    writer.add_scalars('it/Train_Loss', {'loss': stats["total_loss"].local_avg}, iteration)

    writer.add_scalars('it/Train_Accuracy', {'Train Acc': stats["accuracy"].local_avg},
                       iteration)

    all_calced_timings = sum([stats[tms].local_avg for tms in ["time_data_loading",
                                                               "time_cuda_transfer",
                                                               "time_forward",
                                                               "time_backward",
                                                               ]])
    timing_dict = {'Loading Data':       stats["time_data_loading"].local_avg,
                   'Cuda Transfer':      stats["time_cuda_transfer"].local_avg,
                   'Forward Pass':       stats["time_forward"].local_avg,
                   'Backward Pass':      stats["time_backward"].local_avg,
                   'Loading + Transfer + '
                   'Forward + Backward': all_calced_timings,
                   'All':                stats["time_all"].local_avg
                   }

    writer.add_scalars('it/Batch-Wise_Timings', timing_dict, iteration)


def print_tr_stats_timings_avg(tr_stats):
    print('Batch-wise Timings:\n'
          f'Data Loading: {tr_stats["time_data_loading"].avg:.4f}s | '
          f'Cuda Transfer: {tr_stats["time_cuda_transfer"].avg:.4f}s | '
          f'Forward: {tr_stats["time_forward"].avg:.4f}s | '
          f'Backward: {tr_stats["time_backward"].avg:.4f}s | '
          f'All: {tr_stats["time_all"].avg:.4f}s\n')


def get_data(transform, mode='train'):
    print('Loading data for "%s" ...' % mode)
    global dataset
    if args.dataset == 'ucf101':
        dataset = UCF101Dataset(mode=mode,
                                transform=transform,
                                seq_len=args.seq_len,
                                downsample_vid=args.ds,
                                which_split=args.split,
                                max_samples=args.max_samples)
    elif args.dataset == 'hmdb51':
        dataset = HMDB51Dataset(mode=mode,
                                transform=transform,
                                seq_len=args.seq_len,
                                downsample_vid=args.ds,
                                which_split=args.split,
                                max_samples=args.max_samples)
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


def set_path(args, mode="training"):
    if args.resume:
        exp_path = os.path.dirname(os.path.dirname(args.resume))
    else:
        tm = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        exp_path = f'transfer_logs/{tm}_{mode}_{args.prefix}'

    img_path = os.path.join(exp_path, 'img')
    model_path = os.path.join(exp_path, 'model')
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

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
