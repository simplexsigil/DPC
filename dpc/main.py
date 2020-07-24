import argparse
import sys

import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

plt.switch_backend('agg')

sys.path.insert(0, '../utils')
sys.path.insert(0, '../eval')
sys.path.insert(0, '../backbone')

from dataset_3d import *
from dataset_nturgbd import *
from dataset_nturgbd_dali import *
from model_3d import *
from resnet_2d3d import neq_load_customized
from augmentation import *
from utils import AverageMeter, save_checkpoint, denorm, calc_topk_accuracy
from datetime import datetime
import re

import torch
import torch.optim as optim
from torch.utils import data
from torchvision import transforms
import torchvision.utils as vutils

# This way, cuda optimizes for the hardware available, if input size is always equal.
torch.backends.cudnn.benchmark = True

global start_time
global stop_time
global cuda_device
global criterion
global iteration
global de_normalize
global img_path
global writer_train
global args

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=[0], type=int, nargs='+')
parser.add_argument('--epochs', default=300, type=int, help='number of total epochs to run')
parser.add_argument('--dataset', default='kinetics400', type=str)
parser.add_argument('--model', default='skelcont', type=str)
parser.add_argument('--rgb_net', default='resnet18', type=str)
parser.add_argument('--img_dim', default=128, type=int)
parser.add_argument('--seq_len', default=30, type=int, help='number of frames in a video block')
parser.add_argument('--max_samples', default=None, type=int, help='Maximum number of samples loaded by dataloader.')
parser.add_argument('--ds', default=1, type=int, help='frame downsampling rate')
parser.add_argument('--representation_size', default=512, type=int)
parser.add_argument('--score_function', default='cos-nt-xent', type=str)
parser.add_argument('--temperature', default=0.01, type=float, help='Termperature value used for score functions.')
parser.add_argument('--batch_size', default=14, type=int)
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--wd', default=1e-5, type=float, help='weight decay')
parser.add_argument('--resume', default=None, type=str, help='path of model to resume')
parser.add_argument('--pretrain', default=None, type=str, help='path of pretrained model')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--print_freq', default=5, type=int, help='frequency of printing output during training')
parser.add_argument('--reset_lr', action='store_true', help='Reset learning rate when resume training?')
parser.add_argument('--use_dali', action='store_true', default=False, help='Reset learning rate when resume training?')
parser.add_argument('--memory_contrast', default=None, type=int, help='Number of contrast vectors. '
                                                                      'Batch contrast is used if not applied.')
parser.add_argument('--no_cache', action='store_true', default=False, help='Avoid using cached data.')
parser.add_argument('--prefix', default='exp-000', type=str, help='prefix of checkpoint filename')
parser.add_argument('--training_focus', default='all', type=str, help='Defines which parameters are trained.')
parser.add_argument('--loader_workers', default=16, type=int,
                    help='Number of data loader workers to pre load batch data. Main thread used if 0.')
parser.add_argument('--dali_workers', default=16, type=int,
                    help='Number of dali workers to pre load batch data. At least 1 worker is necessary.')
parser.add_argument('--dali_prefetch_queue', default=2, type=int,
                    help='Number of samples to prefetch in GPU memory.')
parser.add_argument('--train_csv',
                    default=os.path.expanduser("~/datasets/nturgbd/project_specific/dpc_converted/train_set.csv"),
                    type=str)
parser.add_argument('--test_csv',
                    default=os.path.expanduser("~/datasets/nturgbd/project_specific/dpc_converted/test_set.csv"),
                    type=str)

# TODO: Correct train json.
parser.add_argument('--train_json_kinetics',
                    default=os.path.expanduser("~/datasets/kinetics/kinetics400-skeleton/kinetics_val_label.json"),
                    type=str)
parser.add_argument('--test_json_kinetics',
                    default=os.path.expanduser("~/datasets/kinetics/kinetics400-skeleton/kinetics_val_label.json"),
                    type=str)
parser.add_argument('--nturgbd-video-info',
                    default=os.path.expanduser("~/datasets/nturgbd/project_specific/dpc_converted/video_info.csv"),
                    type=str)
parser.add_argument('--nturgbd-skele-motion', default=os.path.expanduser("~/datasets/nturgbd/skele-motion"),
                    type=str)
parser.add_argument('--kinetics-video-info',
                    default=os.path.expanduser("~/datasets/kinetics/kinetics400/video_info.csv"),
                    type=str)
parser.add_argument('--kinetics-skele-motion',
                    default=os.path.expanduser("~/datasets/kinetics/kinetics400-skeleton/skele-motion"),
                    type=str)
parser.add_argument('--split-mode', default="perc", type=str)
parser.add_argument('--split-test-frac', default=0.2, type=float)


def argument_checks(args):
    """
    This function performs non-obvious checks on the arguments provided. Most of these problems would also become
    apparent when starting training, but depending on the dataset size, this might take some time. Fail fast.
    """
    assert not (args.resume and args.pretrain), "Use of pretrained model and resuming training makes no sense."
    calc_gpus = len(args.gpu)
    calc_gpus = max(1, calc_gpus - 1) if args.use_dali else calc_gpus  # One GPU only used for DALI
    assert args.batch_size % calc_gpus == 0, "Batch size has to be divisible by GPU count. DALI reduces GPUs by one."
    assert args.loader_workers >= 0
    assert args.dali_workers >= 1, "Minimum 1"

    if not args.use_dali:  # For a cleaner printout of settings.
        args.dali_prefetch_queue = None
        args.dali_workers = None

    return args


def main():
    # Todo: Get rid of all these global variables.
    global args
    global cuda_device
    global criterion
    global iteration
    global de_normalize
    global img_path
    global writer_train

    # TODO: Set with arguments.
    augmentation_settings = {
        "rot_range":      (-30, 30),
        "hue_range":      (-180, 180),
        "sat_range":      (0.0, 1.5),
        "val_range":      (0.5, 1.5),
        "hue_prob":       1.,
        "crop_arr_range": (0.15, 1.)
        }

    best_acc = 0
    iteration = 0

    torch.manual_seed(0)
    np.random.seed(0)

    args = parser.parse_args()

    args = argument_checks(args)

    # setup tools
    de_normalize = denorm()
    img_path, model_path, exp_path = set_path(args)

    # Setup cuda
    cuda_device, device_ids = check_and_prepare_cuda(args.gpu)

    # Prepare model
    model = select_and_prepare_model(args.model)

    # Data Parallel uses a master device (default gpu 0)
    # and performs scatter gather operations on batches and resulting gradients.
    # Distributes batches on mutiple devices to train model in parallel automatically.
    # If we use dali, the last device is used for data-loading only.
    model = nn.DataParallel(model, device_ids=args.gpu if len(args.gpu) < 2 or not args.use_dali else args.gpu[0:-1])
    model = model.to(cuda_device)  # Sends model to device 0, other gpus are used automatically.

    check_and_prepare_parameters(model, args.training_focus)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=False)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    # Prepare Loss
    # Contrastive loss can be implemented with CrossEntropyLoss with vector similarity.
    criterion = nn.CrossEntropyLoss()  # Be aware that this includes a final Softmax.

    # Handle resuming and pretrained network.
    if args.resume:  # Resume a training which was interrupted.
        model, optimizer, start_epoch, iteration, best_acc, lr = prepare_on_resume(model, optimizer,
                                                                                   None if args.reset_lr else args.lr,
                                                                                   args.resume)
        args.lr = lr

    elif args.pretrain:  # Load a pretrained model
        # The difference to resuming: We do not expect the same model.
        # In this case, only some of the pretrained weights are used.
        model = prepare_on_pretrain(model, args.pretrain)
    else:
        pass  # Normal case, no resuming, not pretraining.

    transform = prepare_augmentations(augmentation_settings, args)

    writer_train, writer_val = get_summary_writers(args)

    write_settings_file(args, exp_path)

    train_loader, train_len = get_data(transform, 'train', augmentation_settings, use_dali=args.use_dali)
    val_loader, val_len = get_data(transform, 'val', augmentation_settings, use_dali=False)

    training_loop(model, optimizer, train_loader, val_loader, writer_train, writer_val, model_path, img_path,
                  args.memory_contrast, best_acc)


def training_loop(model, optimizer, train_loader, val_loader, writer_train, writer_val, model_path, img_path,
                  memory_contrast_size=None, best_acc=0.0, best_epoch=0, representation_size=512):
    memories = None
    if memory_contrast_size is not None:
        # torch.autograd.set_detect_anomaly(True)

        memories = {"skeleton": None, "video": None}
        memories["skeleton"] = torch.rand((len(train_loader.dataset), representation_size),
                                          requires_grad=False) - 0.5

        sk_norms = torch.norm(memories["skeleton"], dim=1, keepdim=True)

        memories["skeleton"] = memories["skeleton"] / sk_norms

        memories["video"] = torch.rand((len(train_loader.dataset), representation_size), requires_grad=False) - 0.5

        vid_norms = torch.norm(memories["video"], dim=1, keepdim=True)
        memories["video"] = memories["video"] / vid_norms

    ### main loop ###
    for epoch in range(args.start_epoch, args.epochs):
        plot_angle_distribution(memories["skeleton"], memories["video"], epoch, img_path)

        train_loss, train_acc, train_accuracy_list = train_two_stream_contrastive(train_loader, model, optimizer, epoch,
                                                                                  memories, memory_contrast_size, args)

        val_loss, val_acc, val_accuracy_list = validate(val_loader, model, epoch)

        if args.use_dali:
            train_loader.reset()
            val_loader.reset()

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
        is_best = val_acc > best_acc
        if is_best:
            best_epoch = epoch

        best_acc = max(val_acc, best_acc)
        save_checkpoint({'epoch':      epoch + 1,
                         'net':        args.rgb_net,
                         'state_dict': model.state_dict(),
                         'best_acc':   best_acc,
                         'optimizer':  optimizer.state_dict(),
                         'iteration':  iteration},
                        is_best,
                        model_path=model_path)

        with open(os.path.join(model_path, "training_state.log"), 'a') as f:
            f.write(
                "Epoch: {:4} | Acc Train: {:1.4f} | Acc Val: {:1.4f} | Best Acc Val: {:1.4f} | "
                "Best Epoch: {:4}\n".format(epoch + 1, train_acc, val_acc, best_acc, best_epoch + 1))

    print('Training from ep %d to ep %d finished' % (args.start_epoch, args.epochs))


def train_two_stream_contrastive(data_loader, model, optimizer, epoch, memories=None, memory_contrast_size=None,
                                 args=None):
    data_loading_times = []
    cuda_transfer_times = []
    memory_selection_times = []
    memory_update_times = []
    calculation_times = []
    losses = AverageMeter()
    accuracy = AverageMeter()
    accuracy_list = [AverageMeter(), AverageMeter(), AverageMeter()]
    model.train()
    global iteration
    global start_time
    global stop_time

    start_time = time.perf_counter()
    tic = time.time()

    for idx, out in enumerate(data_loader):
        stop_time = time.perf_counter()  # Timing data loading
        data_loading_times.append(stop_time - start_time)

        bat_idxs, input_seq, sk_seq = out

        batch_size = input_seq.size(0)

        start_time = time.perf_counter()  # Timing calculation

        if memories is not None:
            perm = list(range(memories["video"].shape[0]))
            bat_idxs_lst = sorted(bat_idxs.tolist(), reverse=True)
            for bat_idx in bat_idxs_lst:
                perm.pop(bat_idx)

            random.shuffle(perm)

            rand_idxs = torch.tensor(perm[:memory_contrast_size * len(args.gpu) - batch_size])

            rand_idxs = torch.cat((bat_idxs, rand_idxs), dim=0)

            mem_rgb = memories["video"][rand_idxs]
            mem_sk = memories["skeleton"][rand_idxs]

        else:
            mem_rgb = None
            mem_sk = None

        stop_time = time.perf_counter()

        memory_selection_times.append(stop_time - start_time)

        start_time = time.perf_counter()  # Timing cuda transfer

        input_seq = input_seq.to(cuda_device)
        sk_seq = sk_seq.to(cuda_device)

        if mem_rgb is not None:
            mem_rgb.to(cuda_device)
        if mem_sk is not None:
            mem_sk.to(cuda_device)

        stop_time = time.perf_counter()

        cuda_transfer_times.append(stop_time - start_time)

        start_time = time.perf_counter()

        score, targets, mem_sk, mem_rgb = model(input_seq, sk_seq, mem_rgb, mem_sk)

        # visualize
        if (iteration == 0) or (
                iteration == args.print_freq):  # I suppose this is a bug, since it does not write out images on print frequency, but only the first and second time.
            if batch_size > 2: input_seq = input_seq[0:2, :]
            writer_train.add_image('input_seq',
                                   de_normalize(vutils.make_grid(
                                       input_seq.transpose(2, 3).contiguous().view(-1, 3, args.img_dim, args.img_dim),
                                       nrow=args.seq_len)),
                                   iteration)
        # del input_seq, sk_seq
        targets_detached = targets.detach()  # It's the diagonals.

        loss = criterion(score, targets_detached)

        top1, top3, top5 = calc_topk_accuracy(score, targets_detached, (1, 3, 5))

        accuracy_list[0].update(top1.item(), batch_size)
        accuracy_list[1].update(top3.item(), batch_size)
        accuracy_list[2].update(top5.item(), batch_size)

        losses.update(loss.item(), batch_size)
        accuracy.update(top1.item(), batch_size)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        # del score, targets_detached, loss

        stop_time = time.perf_counter()

        calculation_times.append(stop_time - start_time)

        start_time = time.perf_counter()

        if mem_sk is not None:
            a = 0.1
            mem_sk_new = a * mem_sk.clone().detach().cpu() + (1. - a) * memories["skeleton"][bat_idxs]
            mem_sk_new.requires_grad = False

            mem_rgb_new = a * mem_rgb.clone().detach().cpu() + (1. - a) * memories["video"][bat_idxs]
            mem_rgb_new.requires_grad = False

            memories["skeleton"][bat_idxs] = mem_sk_new
            memories["video"][bat_idxs] = mem_rgb_new

            # del mem_sk, mem_rgb

        stop_time = time.perf_counter()

        memory_update_times.append(stop_time - start_time)

        if idx % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.6f} ({loss.local_avg:.4f})\t'
                  'Acc: top1 {3:.4f}; top3 {4:.4f}; top5 {5:.4f} T:{6:.2f}\t'.format(
                epoch, idx, len(data_loader), top1, top3, top5, time.time() - tic, loss=losses))

            writer_train.add_scalar('local/loss', losses.val, iteration)
            writer_train.add_scalar('local/accuracy', accuracy.val, iteration)

            iteration += 1

        start_time = time.perf_counter()
        tic = time.time()

    print(("Data Loading: {:.4f}s | Cuda Transfer: {:.4f}s | Memory Selection: {:.4f}s | " +
           "Avg t calculation: {:.4f}s | Memory Update: {:.4f}s").format(
        sum(data_loading_times) / len(data_loading_times), sum(cuda_transfer_times) / len(cuda_transfer_times),
        sum(memory_selection_times) / len(memory_selection_times), sum(calculation_times) / len(calculation_times),
        sum(memory_update_times) / len(memory_update_times)))
    return losses.local_avg, accuracy.local_avg, [i.local_avg for i in accuracy_list]


def validate(data_loader, model, epoch):
    losses = AverageMeter()
    accuracy = AverageMeter()
    accuracy_list = [AverageMeter(), AverageMeter(), AverageMeter()]
    model.eval()

    with torch.no_grad():
        for idx, out in enumerate(data_loader):
            bat_idx, input_seq, sk_seq = out

            input_seq = input_seq.to(cuda_device)
            B = input_seq.size(0)

            score, targets, _, _ = model(input_seq, sk_seq)

            del input_seq, sk_seq

            target_flattened = targets.detach()  # It's the diagonals.

            loss = criterion(score, target_flattened)

            top1, top3, top5 = calc_topk_accuracy(score, target_flattened, (1, 3, 5))

            losses.update(loss.item(), B)
            accuracy.update(top1.item(), B)

            del score, target_flattened, loss

            accuracy_list[0].update(top1.item(), B)
            accuracy_list[1].update(top3.item(), B)
            accuracy_list[2].update(top5.item(), B)

    print('[{0}/{1}] Loss {loss.local_avg:.4f}\t'
          'Acc: top1 {2:.4f}; top3 {3:.4f}; top5 {4:.4f} \t'.format(
        epoch, args.epochs, *[i.avg for i in accuracy_list], loss=losses))
    return losses.local_avg, accuracy.local_avg, [i.local_avg for i in accuracy_list]


def check_and_prepare_cuda(device_ids):
    # NVIDIA-SMI uses PCI_BUS_ID device order, but CUDA orders graphics devices by speed by default (fastest first).
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(id) for id in device_ids])

    print('Cuda visible devices: {}'.format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print('Available device count: {}'.format(torch.cuda.device_count()))

    device_ids = list(range(torch.cuda.device_count()))  # The device ids restart from 0 on the visible devices.

    print("Note: Device ids are reindexed on the visible devices and not the same as in nvidia-smi.")

    for i in device_ids:
        print("Using Cuda device {}: {}".format(i, torch.cuda.get_device_name(i)))

    print("Cuda is available: {}".format(torch.cuda.is_available()))

    cudev = torch.device('cuda')

    return cudev, device_ids


def select_and_prepare_model(model_name):
    if model_name == 'skelcont':
        model = SkeleContrast(img_dim=args.img_dim,
                              seq_len=args.seq_len,
                              network=args.rgb_net,
                              representation_size=args.representation_size,
                              score_function=args.score_function)
    else:
        raise ValueError('wrong model!')

    return model


def check_and_prepare_parameters(model, training_focus):
    if training_focus == 'except_resnet':
        for name, param in model.module.resnet.named_parameters():
            param.requires_grad = False

    elif training_focus == 'all':
        pass
    else:
        raise ValueError

    print('\n===========Check Grad============')
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    print('=================================\n')


def prepare_on_resume(model, optimizer, lr, resume_file):
    if not os.path.isfile(resume_file):
        print("####\n[Warning] no checkpoint found at '{}'\n####".format(args.resume))
        raise FileNotFoundError
    else:
        old_lr = float(re.search('_lr(.+?)_', resume_file).group(1))

        print("=> loading resumed checkpoint '{}'".format(resume_file))

        checkpoint = torch.load(resume_file, map_location=torch.device('cpu'))

        epoch = checkpoint['epoch']
        iteration = checkpoint['iteration']
        best_acc = checkpoint['best_acc']

        # I assume this copies the CPU located parameters automatically to cuda.
        model.load_state_dict(checkpoint['state_dict'])

        if not lr:  # If not explicitly reset, load old optimizer.
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr = old_lr
        else:
            print('==== Resuming with lr {:.1e} (Last lr: {:.1e}) ===='.format(lr, old_lr))

        print("=> loaded resumed checkpoint (epoch {}, lr {}) '{}' ".format(epoch, lr, resume_file))

        return model, optimizer, epoch, iteration, best_acc, lr


def prepare_on_pretrain(model, pretrain_file):
    if not os.path.isfile(pretrain_file):
        print("=> no checkpoint found at '{}'".format(args.pretrain))
        raise FileNotFoundError
    else:
        print("=> loading pretrained checkpoint '{}'".format(pretrain_file))

        checkpoint = torch.load(pretrain_file, map_location=torch.device('cpu'))
        model = neq_load_customized(model, checkpoint['state_dict'])

        print("=> loaded pretrained checkpoint '{}' (epoch {})".format(args.pretrain, checkpoint['epoch']))

        return model


def prepare_augmentations(augmentation_settings, args):
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
    elif args.dataset == 'nturgbd' or args.dataset == 'kinetics400':  # designed for nturgbd, short size=150, rand crop to 128x128
        transform = transforms.Compose([
            RandomRotation(degree=augmentation_settings["rot_range"]),
            RandomSizedCrop(size=args.img_dim, consistent=True),
            ColorJitter(brightness=augmentation_settings["val_range"], contrast=0,
                        saturation=augmentation_settings["sat_range"],
                        hue=[val / 360. for val in augmentation_settings["hue_range"]]),
            ToTensor(),
            Normalize()
            ])
    else:
        raise NotImplementedError

    return transform


def get_data(transform, mode='train', augmentation_settings=None, use_dali=False):
    if not use_dali:
        if args.dataset == 'kinetics400':
            dataset = Kinetics400_full_3d(split=mode,
                                          transform=transform,
                                          seq_len=args.seq_len,
                                          downsample=args.ds,
                                          video_info=args.kinetics_video_info,
                                          skele_motion_root=args.kinetics_skele_motion,
                                          split_mode=args.split_mode,
                                          sample_limit=args.max_samples,
                                          use_cache=not args.no_cache)
        elif args.dataset == 'ucf101':
            dataset = UCF101_3d(mode=mode,
                                transform=transform,
                                seq_len=args.seq_len,
                                num_seq=args.num_seq,
                                downsample=args.ds)
        elif args.dataset == 'nturgbd':
            dataset = NTURGBD_3D(split=mode,
                                 transform=transform,
                                 seq_len=args.seq_len,
                                 downsample=args.ds,
                                 nturgbd_video_info=args.nturgbd_video_info,
                                 skele_motion_root=args.nturgbd_skele_motion,
                                 split_mode=args.split_mode,
                                 sample_limit=args.max_samples)
        else:
            raise ValueError('dataset not supported')

        sampler = data.RandomSampler(dataset)

        data_loader = data.DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      sampler=sampler,
                                      shuffle=False,
                                      num_workers=args.loader_workers,
                                      pin_memory=True,
                                      drop_last=True)

        return data_loader, len(data_loader)

    else:
        if not args.dataset == 'nturgbd':
            raise NotImplementedError

        data_loader = NTURGBD3DDali(
            batch_size=args.batch_size,
            split=mode,
            seq_len=args.seq_len,
            downsample=args.ds,
            nturgbd_video_info=args.nturgbd_video_info,
            skele_motion_root=args.nturgbd_skele_motion,
            split_mode=args.split_mode,
            sample_limit=args.max_samples,
            num_workers_loader=args.loader_workers,
            num_workers_dali=args.dali_workers,
            dali_prefetch_queue_depth=args.dali_prefetch_queue,
            dali_devices=[args.gpu[-1]],
            aug_settings=augmentation_settings
            )

        return data_loader, len(data_loader)


def set_path(args, mode="training"):
    if args.resume:
        exp_path = os.path.dirname(os.path.dirname(args.resume))
    else:
        tm = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        exp_path = 'training_logs/{time}_{mode}_{args.prefix}'.format(time=tm, mode=mode, args=args)

    img_path = os.path.join(exp_path, 'img')
    model_path = os.path.join(exp_path, 'model')
    if not os.path.exists(img_path): os.makedirs(img_path)
    if not os.path.exists(model_path): os.makedirs(model_path)

    return img_path, model_path, exp_path


def get_summary_writers(args):
    time_str = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    tboard_str = '{time}-{mode}-{args.prefix}-m={args.model}-bs={args.batch_size}-sl={args.seq_len}-img={args.img_dim}-' \
                 'ds={args.ds}-rgb_net={args.rgb_net}-rs={args.representation_size}'
    val_name = tboard_str.format(args=args, mode="val", time=time_str)
    train_name = tboard_str.format(args=args, mode="train", time=time_str)

    try:  # old version
        writer_val = SummaryWriter(log_dir=os.path.join(img_path, val_name))
        writer_train = SummaryWriter(log_dir=os.path.join(img_path, train_name))
    except:  # v1.7
        writer_val = SummaryWriter(logdir=os.path.join(img_path, val_name))
        writer_train = SummaryWriter(logdir=os.path.join(img_path, train_name))

    return writer_train, writer_val


def write_settings_file(args, exp_path):
    args_d = vars(args)
    training_description = ["{}: {}".format(key, args_d[key]) for key in sorted(args_d.keys()) if
                            args_d[key] is not None]
    training_description = "\n".join(training_description)

    with open(os.path.join(exp_path, "training_description.txt"), 'w') as f:
        import subprocess
        label = subprocess.check_output(["git", "describe", "--always"]).decode("utf-8").strip()

        f.write("Git describe of repo: {}".format(label))

        f.write("\n\n")

        f.write(training_description)


def rand_self_cos_similarities(x, samples=1000):
    samples = min(len(x), samples)

    idxs = list(range(samples))
    random.shuffle(idxs)

    x = x[:samples]
    y = x[idxs]

    angles = torch.sum(torch.mul(x, y), dim=1)
    angles = angles.reshape((-1,))

    return angles


def plot_angle_distribution(memories_sk: torch.Tensor, memories_rgb: torch.Tensor, epoch: int, base_path="."):
    # matplotlib.use('module://backend_interagg')
    import matplotlib.pyplot as plt

    img_path = os.path.join(base_path, "angle_dists")

    if not os.path.exists(img_path):
        os.makedirs(img_path)

    mem_sk_beg_angles = rand_self_cos_similarities(memories_sk[:1000])
    mem_sk_end_angles = rand_self_cos_similarities(memories_sk[:-1000])
    mem_rgb_start_angles = rand_self_cos_similarities(memories_rgb[:1000])
    mem_rgb_end_angles = rand_self_cos_similarities(memories_rgb[:-1000])

    angle_dict = {"Skeleton Memory Head Angles": mem_sk_beg_angles,
                  "RGB Memory Head Angles":      mem_rgb_start_angles,
                  "Skeleton Memory Tail Angles": mem_sk_end_angles,
                  "RGB Memory Tail Angles":      mem_rgb_end_angles,
                  }

    fig, axs = plt.subplots(2, 2, sharex='col', figsize=(15, 15))

    for idx, (name, angles) in enumerate(angle_dict.items()):
        idx_1 = idx // 2
        idx_2 = idx % 2
        ax = axs[idx_1, idx_2]
        ax.hist(angles, 100, facecolor='b', density=True, alpha=0.75)

        ax.set_title(name)

        if idx_1 == 1:
            ax.set_xlabel('Cosine Similarity')

        ax.set_ylabel('Count')
        ax.set_xlim(-1.1, 1.1)
        ax.grid(True)

    fig.savefig(os.path.join(img_path, 'angle_distribution_ep{}.png'.format(epoch)))


if __name__ == '__main__':
    main()
