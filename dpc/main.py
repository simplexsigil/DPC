import argparse
import sys
import os

import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

plt.switch_backend('agg')

sys.path.insert(0, '../utils')  # If that is the way to include paths for this project, then why not also for 'backbone'?
sys.path.insert(0, '../eval')
sys.path.insert(0, '../backbone')

from dataset_3d import *
from model_3d import *
from resnet_2d3d import neq_load_customized
from augmentation import *
from utils import AverageMeter, save_checkpoint, denorm, calc_topk_accuracy

import torch
import torch.optim as optim
from torch.utils import data
from torchvision import transforms
import torchvision.utils as vutils

# This way, cuda optimizes for the hardware available, if input size is always equal.
# torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--net', default='resnet18', type=str)
parser.add_argument('--model', default='dpc-rnn', type=str)
parser.add_argument('--dataset', default='nturgbd', type=str)
parser.add_argument('--seq_len', default=5, type=int, help='number of frames in each video block')
parser.add_argument('--num_seq', default=6, type=int, help='number of video blocks')
parser.add_argument('--pred_step', default=2, type=int)
parser.add_argument('--ds', default=1, type=int, help='frame downsampling rate')
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--wd', default=1e-5, type=float, help='weight decay')
parser.add_argument('--resume', default='', type=str, help='path of model to resume')
parser.add_argument('--pretrain', default='', type=str, help='path of pretrained model')
parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--gpu', default=[0], type=int, nargs='+')
parser.add_argument('--print_freq', default=5, type=int, help='frequency of printing output during training')
parser.add_argument('--reset_lr', action='store_true', help='Reset learning rate when resume training?')
parser.add_argument('--prefix', default='tmp', type=str, help='prefix of checkpoint filename')
parser.add_argument('--train_what', default='all', type=str)
parser.add_argument('--img_dim', default=128, type=int)
parser.add_argument('--train_csv',
                    default=os.path.expanduser("~/datasets/nturgbd/project_specific/dpc_converted/train_set.csv"),
                    type=str)
parser.add_argument('--test_csv',
                    default=os.path.expanduser("~/datasets/nturgbd/project_specific/dpc_converted/test_set.csv"),
                    type=str)
parser.add_argument('--nturgbd-video-info',
                    default=os.path.expanduser("~/datasets/nturgbd/project_specific/dpc_converted/video_info.csv"),
                    type=str)
parser.add_argument('--nturgbd-skele-motion', default=os.path.expanduser("~/datasets/nturgbd/skele-motion"),
                    type=str)
parser.add_argument('--split-mode', default="all", type=str)


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    global args;
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # NVIDIA-SMI uses PCI_BUS_ID device order, but CUDA orders graphics devices by speed by default (fastest first).
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(id) for id in args.gpu])
    
    
    print ('Cuda visible devices: {}'.format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print ('Available device count: {}'.format(torch.cuda.device_count()))

    args.gpu = list(range(torch.cuda.device_count()))  # Really weird: In Pytorch 1.2, the device ids start from 0 on the visible devices.

    print("Note: At least in Pytorch 1.2, device ids are reindexed on the visible devices and not the same as in nvidia-smi.")

    for i in args.gpu:
        print("Using Cuda device {}: {}".format(i, torch.cuda.get_device_name(i)))
    print("Cuda is available: {}".format(torch.cuda.is_available()))
    global cuda;
    cuda = torch.device('cuda')
    
    ### dpc model ###
    if args.model == 'dpc-rnn':
        model = DPC_RNN(sample_size=args.img_dim,
                        num_seq=args.num_seq,
                        seq_len=args.seq_len,
                        network=args.net,
                        pred_step=args.pred_step)
    else: raise ValueError('wrong model!')

    # Data Parallel uses a master device (default gpu 0) and performs scatter gather operations on batches and resulting gradients.
    model = nn.DataParallel(model)  # Distributes batches on mutiple devices to train model in parallel automatically.
    model = model.to(cuda)  # Sends model to device 0, other gpus are used automatically.
    global criterion
    criterion = nn.CrossEntropyLoss()  # Contrastive loss is basically CrossEntropyLoss with vector similarity and temperature.

    ### optimizer ###
    if args.train_what == 'last':
        for name, param in model.module.resnet.named_parameters():
            param.requires_grad = False
    else: pass  # train all layers

    print('\n===========Check Grad============')
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    print('=================================\n')

    params = model.parameters()
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    args.old_lr = None

    best_acc = 0
    global iteration
    iteration = 0

    ### restart training ###
    if args.resume:
        if os.path.isfile(args.resume):
            args.old_lr = float(re.search('_lr(.+?)_', args.resume).group(1))
            print("=> loading resumed checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch']
            iteration = checkpoint['iteration']
            best_acc = checkpoint['best_acc']
            # I assume this copies the *cpu located* parameters to the CUDA model automatically?
            model.load_state_dict(checkpoint['state_dict'])
            if not args.reset_lr:  # if didn't reset lr, load old optimizer
                optimizer.load_state_dict(checkpoint['optimizer'])
            else: print('==== Change lr from %f to %f ====' % (args.old_lr, args.lr))
            print("=> loaded resumed checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("[Warning] no checkpoint found at '{}'".format(args.resume))

    if args.pretrain:
        if os.path.isfile(args.pretrain):
            print("=> loading pretrained checkpoint '{}'".format(args.pretrain))
            checkpoint = torch.load(args.pretrain, map_location=torch.device('cpu'))
            model = neq_load_customized(model, checkpoint['state_dict'])
            print("=> loaded pretrained checkpoint '{}' (epoch {})"
                  .format(args.pretrain, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrain))

    ### load data ###
    if args.dataset == 'ucf101':  # designed for ucf101, short size=256, rand crop to 224x224 then scale to 128x128
        transform = transforms.Compose([
            RandomHorizontalFlip(consistent=True),
            RandomCrop(size=224, consistent=True),
            Scale(size=(args.img_dim, args.img_dim)),
            RandomGray(consistent=False, p=0.5),
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=1.0),
            ToTensor(),
            Normalize()
        ])
    elif args.dataset == 'k400':  # designed for kinetics400, short size=150, rand crop to 128x128
        transform = transforms.Compose([
            RandomSizedCrop(size=args.img_dim, consistent=True, p=1.0),
            RandomHorizontalFlip(consistent=True),
            RandomGray(consistent=False, p=0.5),
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=1.0),
            ToTensor(),
            Normalize()
        ])
    elif args.dataset == 'nturgbd':  # designed for nturgbd, short size=150, rand crop to 128x128
        transform = transforms.Compose([
            RandomSizedCrop(size=args.img_dim, consistent=True, p=1.0),
            RandomHorizontalFlip(consistent=True),
            RandomGray(consistent=False, p=0.5),
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=1.0),
            ToTensor(),
            Normalize()
        ])


    train_loader = get_data(transform, 'train')
    val_loader = get_data(transform, 'val')

    # setup tools
    global de_normalize;
    de_normalize = denorm()
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
        train_loss, train_acc, train_accuracy_list = train(train_loader, model, optimizer, epoch)
        val_loss, val_acc, val_accuracy_list = validate(val_loader, model, epoch)

        # save curve
        writer_train.add_scalar('global/loss', train_loss, epoch)
        writer_train.add_scalar('global/accuracy', train_acc, epoch)
        writer_val.add_scalar('global/loss', val_loss, epoch)
        writer_val.add_scalar('global/accuracy', val_acc, epoch)
        writer_train.add_scalar('accuracy/top1', train_accuracy_list[0], epoch)
        writer_train.add_scalar('accuracy/top3', train_accuracy_list[1], epoch)
        writer_train.add_scalar('accuracy/top5', train_accuracy_list[2], epoch)
        writer_val.add_scalar('accuracy/top1', val_accuracy_list[0], epoch)
        writer_val.add_scalar('accuracy/top3', val_accuracy_list[1], epoch)
        writer_val.add_scalar('accuracy/top5', val_accuracy_list[2], epoch)

        # save check_point
        is_best = val_acc > best_acc;
        best_acc = max(val_acc, best_acc)
        save_checkpoint({'epoch': epoch + 1,
                         'net': args.net,
                         'state_dict': model.state_dict(),
                         'best_acc': best_acc,
                         'optimizer': optimizer.state_dict(),
                         'iteration': iteration},
                        is_best, filename=os.path.join(model_path, 'epoch%s.pth.tar' % str(epoch + 1)), keep_all=False)

    print('Training from ep %d to ep %d finished' % (args.start_epoch, args.epochs))


def process_output(mask):
    '''task mask as input, compute the target for contrastive loss'''
    # dot product is computed in parallel gpus, so get less easy neg, bounded by batch size in each gpu'''
    # mask meaning: -2: omit, -1: temporal neg (hard), 0: easy neg, 1: pos, -3: spatial neg
    (B, NP, SQ, B2, NS, _) = mask.size()  # [B, P, SQ, B, N, SQ]
    target = mask == 1
    target.requires_grad = False
    return target, (B, B2, NS, NP, SQ)


def train(data_loader, model, optimizer, epoch):
    losses = AverageMeter()
    accuracy = AverageMeter()
    accuracy_list = [AverageMeter(), AverageMeter(), AverageMeter()]
    model.train()
    global iteration

    for idx, input_seq in enumerate(data_loader):
        tic = time.time()
        input_seq = input_seq.to(cuda)
        B = input_seq.size(0)
        [score_, mask_] = model(input_seq)
        # visualize
        if (iteration == 0) or (iteration == args.print_freq):  # I suppose this is a bug, since it does not write out images on print frequency, but only the first and second time.
            if B > 2: input_seq = input_seq[0:2, :]
            writer_train.add_image('input_seq',
                                   de_normalize(vutils.make_grid(
                                       input_seq.transpose(2, 3).contiguous().view(-1, 3, args.img_dim, args.img_dim),
                                       nrow=args.num_seq * args.seq_len)),
                                   iteration)
        del input_seq

        if idx == 0:
            target_, (_, B2, NS, NP, SQ) = process_output(mask_)

        # TODO: adapt logic for two stream network.
        # score is a 6d tensor: [B, P, SQ, B, N, SQ]
        score_flattened = score_.view(B * NP * SQ, B2 * NS * SQ)
        target_flattened = target_.view(B * NP * SQ, B2 * NS * SQ)
        target_flattened = target_flattened.double()
        target_flattened = target_flattened.argmax(dim=1)

        loss = criterion(score_flattened, target_flattened)
        top1, top3, top5 = calc_topk_accuracy(score_flattened, target_flattened, (1, 3, 5))

        accuracy_list[0].update(top1.item(), B)
        accuracy_list[1].update(top3.item(), B)
        accuracy_list[2].update(top5.item(), B)

        losses.update(loss.item(), B)
        accuracy.update(top1.item(), B)

        del score_

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del loss

        if idx % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.6f} ({loss.local_avg:.4f})\t'
                  'Acc: top1 {3:.4f}; top3 {4:.4f}; top5 {5:.4f} T:{6:.2f}\t'.format(
                epoch, idx, len(data_loader), top1, top3, top5, time.time() - tic, loss=losses))

            writer_train.add_scalar('local/loss', losses.val, iteration)
            writer_train.add_scalar('local/accuracy', accuracy.val, iteration)

            iteration += 1

    return losses.local_avg, accuracy.local_avg, [i.local_avg for i in accuracy_list]


def validate(data_loader, model, epoch):
    losses = AverageMeter()
    accuracy = AverageMeter()
    accuracy_list = [AverageMeter(), AverageMeter(), AverageMeter()]
    model.eval()

    with torch.no_grad():
        for idx, input_seq in tqdm(enumerate(data_loader), total=len(data_loader)):
            input_seq = input_seq.to(cuda)
            B = input_seq.size(0)
            [score_, mask_] = model(input_seq)
            del input_seq

            if idx == 0: target_, (_, B2, NS, NP, SQ) = process_output(mask_)

            # [B, P, SQ, B, N, SQ]
            score_flattened = score_.view(B * NP * SQ, B2 * NS * SQ)
            target_flattened = target_.view(B * NP * SQ, B2 * NS * SQ)
            target_flattened = target_flattened.double()
            target_flattened = target_flattened.argmax(dim=1)

            loss = criterion(score_flattened, target_flattened)
            top1, top3, top5 = calc_topk_accuracy(score_flattened, target_flattened, (1, 3, 5))

            losses.update(loss.item(), B)
            accuracy.update(top1.item(), B)

            accuracy_list[0].update(top1.item(), B)
            accuracy_list[1].update(top3.item(), B)
            accuracy_list[2].update(top5.item(), B)

    print('[{0}/{1}] Loss {loss.local_avg:.4f}\t'
          'Acc: top1 {2:.4f}; top3 {3:.4f}; top5 {4:.4f} \t'.format(
        epoch, args.epochs, *[i.avg for i in accuracy_list], loss=losses))
    return losses.local_avg, accuracy.local_avg, [i.local_avg for i in accuracy_list]


def get_data(transform, mode='train'):
    print('Loading data for "%s" ...' % mode)
    if args.dataset == 'k400':
        use_big_K400 = args.img_dim > 140
        dataset = Kinetics400_full_3d(mode=mode,
                                      transform=transform,
                                      seq_len=args.seq_len,
                                      num_seq=args.num_seq,
                                      downsample=5,
                                      big=use_big_K400)
    elif args.dataset == 'ucf101':
        dataset = UCF101_3d(mode=mode,
                            transform=transform,
                            seq_len=args.seq_len,
                            num_seq=args.num_seq,
                            downsample=args.ds)
    elif args.dataset == 'nturgbd':
        dataset = NTURGBD_3D(mode=mode,
                             transform=transform,
                             seq_len=args.seq_len,
                             num_seq=args.num_seq,
                             downsample=args.ds,
                             nturgbd_video_info=args.nturgbd_video_info,
                             skele_motion_root=args.nturgbd_skele_motion,
                             split_mode=args.split_mode)
    else:
        raise ValueError('dataset not supported')

    sampler = data.RandomSampler(dataset)

    if mode == 'train':
        data_loader = data.DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      sampler=sampler,
                                      shuffle=False,
                                      num_workers=32,
                                      pin_memory=True,
                                      drop_last=True)
    elif mode == 'val':
        data_loader = data.DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      sampler=sampler,
                                      shuffle=False,
                                      num_workers=32,
                                      pin_memory=True,
                                      drop_last=True)
    print('"%s" dataset size: %d' % (mode, len(dataset)))
    return data_loader


def set_path(args):
    if args.resume: exp_path = os.path.dirname(os.path.dirname(args.resume))
    else:
        exp_path = 'log_{args.prefix}/{args.dataset}-{args.img_dim}_{0}_{args.model}_\
bs{args.batch_size}_lr{1}_seq{args.num_seq}_pred{args.pred_step}_len{args.seq_len}_ds{args.ds}_\
train-{args.train_what}{2}'.format(
            'r%s' % args.net[6::], \
            args.old_lr if args.old_lr is not None else args.lr, \
            '_pt=%s' % args.pretrain.replace('/', '-') if args.pretrain else '', \
            args=args)
    img_path = os.path.join(exp_path, 'img')
    model_path = os.path.join(exp_path, 'model')
    if not os.path.exists(img_path): os.makedirs(img_path)
    if not os.path.exists(model_path): os.makedirs(model_path)
    return img_path, model_path


if __name__ == '__main__':
    main()
