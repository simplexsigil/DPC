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
from utils import AverageMeter, ConfusionMeter, write_log, calc_topk_accuracy, calc_accuracy, \
    write_out_images, write_out_checkpoint

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
parser.add_argument('--ds_vid', default=2, type=int)
parser.add_argument('--img_dim', default=224, type=int)

parser.add_argument('--model', default='r2+1d', type=str, choices=["resnet", "dpc-resnet", "r2+1d"])
parser.add_argument('--vid_backbone', default='r2+1d18', type=str, choices=['r2+1d18', 'resnet18'])
parser.add_argument('--backbone_naming', default='vid_backbone', type=str, choices=['vid_backbone', 'backbone'])
parser.add_argument('--dropout', default=0.5, type=float)
parser.add_argument('--representation_size', default=512, type=int)
parser.add_argument('--hidden_width', default=512, type=int)
parser.add_argument('--num_class', default=51, type=int)
parser.add_argument('--class_tapping', default=-1, type=int,
                    help="How many fully connected layers to go back to attach the classification layer.")

parser.add_argument('--optimizer', default="Adam", choices=["Adam", "SGD"], type=str)
parser.add_argument('--scheduler_steps', default=[50, 80, 100, 120, 180], type=int, nargs="+")
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--wd', default=1e-3, type=float, help='weight decay')
parser.add_argument('--fine_tuning', default=0.1, type=float, help='A ratio which determines the learning rate '
                                                                   'for the backbone in relation to lr.'
                                                                   'Backbone will be frozen if 0.')

parser.add_argument('--print_freq', default=5, type=int)

parser.add_argument('--save_best_val_loss', type=bool, default=False, help='Save model with best Val Loss.')
parser.add_argument('--save_best_val_acc', type=bool, default=True, help='Save model with best Val Accuracy.')
parser.add_argument('--save_best_train_loss', type=bool, default=False, help='Save model with best Train Loss.')
parser.add_argument('--save_best_train_acc', type=bool, default=True, help='Save model with best Train Accuracy.')

parser.add_argument('--pretrain',
                    default='/home/david/workspaces/cvhci/experiment_results/pretraining_batch_contrast_kinetics_r21d/2020-08-07_01-44-34_training_exp-000/model/model_best.pth.tar',
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

    cuda_dev = prepare_cuda(args)
    print()

    if args.dataset == 'ucf101':
        args.num_class = 101
    elif args.dataset == 'hmdb51':
        args.num_class = 51

    ### classifier model ###
    model = select_model(args)

    print("Using model {}".format(model.__class__))
    print()

    model = nn.DataParallel(model)
    model = model.to(cuda_dev)

    global criterion
    criterion = nn.CrossEntropyLoss()

    if args.test:
        test_only(model, args)
        sys.exit()

    optimizer, scheduler = prepare_optimizer(model, args)

    args.old_lr = None
    best_acc = 0
    global iteration
    iteration = 0

    if args.resume:
        prepare_resume(model, optimizer, args)
    else:
        args.best_train_loss = None
        args.best_train_acc = None
        args.best_val_loss = None
        args.best_val_acc = None

    if (not args.resume) and args.pretrain:
        if args.pretrain == 'random':
            print('=> using random weights')
        elif os.path.isfile(args.pretrain):
            checkpoint = torch.load(args.pretrain, map_location=torch.device('cpu'))

            model.module.load_weights_state_dict(checkpoint['state_dict'], model=model)
            print("=> loaded pretrained checkpoint '{}' (epoch {})".format(args.pretrain, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrain))

    ### load data ###
    transform = transforms.Compose([
        RandomSizedCrop(consistent=True, size=224, p=1.0, crop_area=(0.15, 1.0)),
        Scale(size=(args.img_dim, args.img_dim)),
        RandomHorizontalFlip(consistent=True),
        ColorJitter(brightness=1, contrast=1, saturation=1, hue=0.20, p=0.3, consistent=True),
        ToTensor(),
        Normalize()
        ])
    val_transform = transforms.Compose([
        RandomSizedCrop(consistent=True, size=224, p=0.3),
        Scale(size=(args.img_dim, args.img_dim)),
        ToTensor(),
        Normalize()
        ])

    train_loader = get_data(transform, args, 'train')
    val_loader = get_data(val_transform, args, 'val')

    # setup tools
    global de_normalize
    global img_path
    img_path, model_path = set_path(args)
    args.img_path, args.model_path = img_path, model_path
    global writer_train
    try:  # old version
        writer_val = SummaryWriter(log_dir=os.path.join(img_path, 'val'))
        writer_train = SummaryWriter(log_dir=os.path.join(img_path, 'train'))
    except:  # v1.7
        writer_val = SummaryWriter(logdir=os.path.join(img_path, 'val'))
        writer_train = SummaryWriter(logdir=os.path.join(img_path, 'train'))

    best_val_acc = args.best_val_acc
    best_val_loss = args.best_val_loss
    best_train_acc = args.best_train_acc
    best_train_loss = args.best_train_loss

    ### main loop ###
    for epoch in range(args.start_epoch, args.epochs):
        train_loss, train_acc = train(train_loader, model, optimizer, epoch, args)
        val_loss, val_acc = validate(val_loader, model, epoch, args)
        scheduler.step()

        best_val_acc = val_acc if best_val_acc is None or val_acc > best_val_acc else best_val_acc
        best_val_loss = val_loss if best_val_loss is None or val_loss < best_val_loss else best_val_loss

        best_train_acc = train_acc if best_train_acc is None or train_acc > best_train_acc else best_train_acc
        best_train_loss = train_loss if best_train_loss is None or train_loss < best_train_loss else best_train_loss

        write_out_checkpoint(epoch, iteration, model, optimizer, args,
                             train_loss, train_acc, val_loss, val_acc,
                             best_train_loss, best_train_acc, best_val_loss, best_val_acc)

    print('Training from ep %d to ep %d finished' % (args.start_epoch, args.epochs))

    args.test = os.path.join(args.model_path, "model_best.pth.tar")
    test_only(model, args)


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

        vid_seq = vid_seq.to(cuda_dev)
        target = target.to(cuda_dev)

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
            input_seq = input_seq.to(cuda_dev)
            target = target.to(cuda_dev)
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


def test(data_loader, model, num_epoch):
    losses = AverageMeter()
    acc_top1 = AverageMeter()
    acc_top5 = AverageMeter()
    confusion_mat = ConfusionMeter(args.num_class)
    model.eval()
    with torch.no_grad():
        for idx, (input_seq, target) in tqdm(enumerate(data_loader), total=len(data_loader)):
            input_seq = input_seq.to(cuda_dev)
            target = target.to(cuda_dev)
            (batch_size, C, T, W, H) = input_seq.shape

            seq_len_ds = args.seq_len // args.ds_vid + (1 if args.seq_len % args.ds_vid != 0 else 0)

            input_seq = input_seq.reshape((batch_size, C, -1, seq_len_ds, W, H))
            input_seq = input_seq.transpose(1, 2)
            (batch_size, num_seq, C, T, W, H) = input_seq.shape
            input_seq = input_seq.reshape((batch_size * num_seq, C, seq_len_ds, W, H))

            scores = model(input_seq)
            del input_seq

            soft_scores = nn.functional.softmax(scores, 1)
            mean_score = torch.mean(soft_scores, 0).view((batch_size, -1))

            top1, top5 = calc_topk_accuracy(mean_score, target, (1, 5))

            _, pred = torch.max(mean_score, 1)
            confusion_mat.update(pred, target.view(-1).byte())

            acc_top1.update(top1.item(), batch_size)
            acc_top5.update(top5.item(), batch_size)
            del top1, top5

            loss = criterion(scores, target.repeat(num_seq))

            losses.update(loss.item(), batch_size)
            del loss

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


def get_data(transform, args, mode='train'):
    print('Loading data for "%s" ...' % mode)
    global dataset
    if args.dataset == 'ucf101':
        dataset = UCF101Dataset(mode=mode,
                                transform=transform,
                                seq_len=args.seq_len,
                                downsample_vid=args.ds_vid,
                                which_split=args.split,
                                max_samples=args.max_samples)
    elif args.dataset == 'hmdb51':
        dataset = HMDB51Dataset(mode=mode,
                                transform=transform,
                                seq_len=args.seq_len,
                                downsample_vid=args.ds_vid,
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


def prepare_resume(model, optimizer, args):
    if os.path.isfile(args.resume):
        args.old_lr = float(re.search('_lr(.+?)_', args.resume).group(1))
        print("=> loading resumed checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
        args.start_epoch = checkpoint['epoch']
        args.start_iteration = checkpoint['iteration']

        args.best_train_loss = checkpoint['best_train_loss']
        args.best_train_acc = checkpoint['best_train_acc']
        args.best_val_loss = checkpoint['best_val_loss']
        args.best_val_acc = checkpoint['best_val_acc']

        model.load_state_dict(checkpoint['state_dict'])
        if not args.reset_lr:  # if didn't reset lr, load old optimizer
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print('==== Change lr from %f to %f ====' % (args.old_lr, args.lr))
        iteration = checkpoint['iteration']
        print("=> loaded resumed checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))


def prepare_optimizer(model, args):
    ### optimizer ###
    if args.fine_tuning == 1.0:
        params = model.parameters()

        print('\n==========Training All===========')
        print(f"{'Name':<42} {'Requires Grad':<6} Learning Rate")
        for name, param in model.named_parameters():
            print(f"{name:<50} {str(param.requires_grad):<6} {args.lr}")
        print('=================================\n')

    elif args.fine_tuning == 0.0:
        print("=======Training Last Only========")
        print("Only training the following parameters:")
        print(f"{'Name':<42} {'Requires Grad':<6} Learning Rate")

        params = []
        for name, param in model.module.named_parameters():
            if ("backbone" in name) or ("vid_fc" in name):
                pass
            else:
                params.append({'params': param})
                print(f"{name:<50} {str(param.requires_grad):<6} {args.lr}")
    elif args.fine_tuning < 0.0:
        raise ValueError
    else:
        print("====Training with Fine Tuning====")
        print(f'The backbone network is finetuned with a learning rate of {args.fine_tuning} x main learning rate.')
        if args.fine_tuning > 1.0:
            print("WARNING: A fine tuning learning rate ratio larger than 1.0 does not make sense.")

        print(f"{'Name':<42} {'Requires Grad':<6} Learning Rate")
        params = []
        for name, param in model.module.named_parameters():
            if ("backbone" in name) or ("vid_fc" in name):
                params.append({'params': param, 'lr': args.lr * args.fine_tuning})
                print(f"{name:<50} {str(param.requires_grad):<6} {args.lr * args.fine_tuning}")
            else:
                params.append({'params': param})
                print(f"{name:<50} {str(param.requires_grad):<6} {args.lr}")

    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)

    if args.dataset == 'hmdb51':
        lr_lambda = lambda ep: MultiStepLR_Restart_Multiplier(ep, gamma=0.1, step=args.scheduler_steps, repeat=1)
    elif args.dataset == 'ucf101':
        if args.img_dim == 224:
            lr_lambda = lambda ep: MultiStepLR_Restart_Multiplier(ep, gamma=0.1, step=args.scheduler_steps, repeat=1)
        else:
            lr_lambda = lambda ep: MultiStepLR_Restart_Multiplier(ep, gamma=0.1, step=args.scheduler_steps, repeat=1)
    else:
        raise ValueError

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    return optimizer, scheduler


def select_model(args):
    if args.model == "dpc-resnet":
        model = DPCResnetClassifier(img_dim=args.img_dim,
                                    seq_len=args.seq_len,
                                    downsampling=args.ds_vid,
                                    vid_backbone=args.vid_backbone,
                                    num_class=args.num_class,
                                    dropout=args.dropout,
                                    representation_size=args.representation_size,
                                    hidden_width=args.hidden_width,
                                    classification_tapping=args.class_tapping,
                                    backbone_naming=args.backbone_naming
                                    )
    elif args.model == "r2+1d":
        model = R2plus1DClassifier(backbone=args.vid_backbone,
                                   num_class=args.num_class,
                                   dropout=args.dropout,
                                   representation_size=args.representation_size,
                                   hidden_fc_width=args.hidden_width,
                                   classification_tapping=args.class_tapping
                                   )
    else:
        raise ValueError('wrong model!')

    return model


def prepare_cuda(args):
    os.environ[
        "CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # NVIDIA-SMI uses PCI_BUS_ID device order, but CUDA orders graphics devices by speed by default (fastest first).
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(id) for id in args.gpu])

    print('Cuda visible devices: {}'.format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print('Available device count: {}'.format(torch.cuda.device_count()))

    args.gpu = list(range(
        torch.cuda.device_count()))  # Really weird: In Pytorch 1.2, the device ids start from 0 on the visible devices.

    print("Note: Device ids are reindexed on the visible devices and not the same as in nvidia-smi.")

    for i in args.gpu:
        print("Using Cuda device {}: {}".format(i, torch.cuda.get_device_name(i)))
    print("Cuda is available: {}".format(torch.cuda.is_available()))

    global cuda_dev
    cuda_dev = torch.device('cuda')

    return cuda_dev


def test_only(model, args):
    if os.path.isfile(args.test):
        print('\n==========Testing Model===========')
        print("Loading testing checkpoint '{}'".format(args.test))
        checkpoint = torch.load(args.test)

        model.load_state_dict(checkpoint['state_dict'])
        print(f"Successfully loaded testing checkpoint '{args.test}' (epoch {checkpoint['epoch']})")
        num_epoch = checkpoint['epoch']
    elif args.test == 'random':
        num_epoch = 0
        print("=> [Warning] loaded random weights")
    else:
        raise ValueError

    transform = transforms.Compose([
        Scale(size=args.img_dim),
        RandomSizedCrop(consistent=True, size=args.img_dim, p=0.0),
        ToTensor(),
        Normalize()
        ])
    test_loader = get_data(transform, args, 'test')
    test_loss, test_acc = test(test_loader, model, num_epoch)


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
