import os
import random
import time
from collections import deque

import torch

import train_batch_contrastive as tbc
from utils import AverageMeter, write_out_images, calc_topk_accuracy, save_checkpoint


def training_loop_mem_contrast(model, optimizer, criterion, train_loader, val_loader, writer_train, writer_val,
                               model_path, img_path, args, cuda_device, best_acc=0.0, best_epoch=0):
    memories, mem_queue = initialize_memories(len(train_loader.dataset), args.representation_size, args.memory_contrast,
                                              len(args.gpu), args.batch_size)

    iteration = args.start_epoch * len(train_loader)  # In case we are resuming.

    ### main loop ###
    for epoch in range(args.start_epoch, args.epochs):
        # plot_angle_distribution(memories["skeleton"], memories["video"], epoch, img_path)

        iteration, train_acc = train_skvid_mem_contrast(train_loader, memories, mem_queue, model, optimizer, criterion,
                                                        epoch, iteration, args, writer_train, cuda_device)

        val_loss, val_acc, val_accuracy_list = tbc.validate_batch_contrast(val_loader, model, epoch)

        if args.use_dali:
            train_loader.reset()
            val_loader.reset()

        # save curve
        writer_val.add_scalar('ep/val_loss', val_loss, epoch)

        writer_val.add_scalars('ep/val_accuracy', {"top1": val_accuracy_list[0],
                                                   "top3": val_accuracy_list[1],
                                                   "top5": val_accuracy_list[2]
                                                   }, epoch)

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


def initialize_memories(mem_count, representation_size, contrast_size, gpu_count, batch_size, model=None):
    if model is not None:
        # TODO: Initialize memories by forwarding every datasample once.
        raise NotImplementedError
    else:
        memories = {"skeleton": None, "video": None}
        memories["skeleton"] = torch.rand((mem_count, representation_size), requires_grad=False) - 0.5  # -0.5 to 0.5

        sk_norms = torch.norm(memories["skeleton"], dim=1, keepdim=True)

        memories["skeleton"] = memories["skeleton"] / sk_norms

        memories["video"] = torch.rand((mem_count, representation_size), requires_grad=False) - 0.5  # -0.5 to 0.5

        vid_norms = torch.norm(memories["video"], dim=1, keepdim=True)
        memories["video"] = memories["video"] / vid_norms

        mem_queue = deque(maxlen=contrast_size * gpu_count - batch_size)

    return memories, mem_queue


def train_skvid_mem_contrast(data_loader, memories, mem_queue, model, optimizer, criterion, epoch, iteration, args,
                             stat_writer, cuda_device):
    tr_stats = {"time_data_loading":     AverageMeter(locality=args.print_freq),
                "time_cuda_transfer":    AverageMeter(locality=args.print_freq),
                "time_memory_selection": AverageMeter(locality=args.print_freq),
                "time_memory_update":    AverageMeter(locality=args.print_freq),
                "time_forward":          AverageMeter(locality=args.print_freq),
                "time_backward":         AverageMeter(locality=args.print_freq),

                "total_loss":            AverageMeter(locality=args.print_freq),
                "sk_loss":               AverageMeter(locality=args.print_freq),
                "vid_loss":              AverageMeter(locality=args.print_freq),

                "cv_rep_sim_m":          AverageMeter(locality=args.print_freq),
                "cv_rep_sim_s":          AverageMeter(locality=args.print_freq),
                "cv_rand_sim_m":         AverageMeter(locality=args.print_freq),
                "cv_rand_sim_s":         AverageMeter(locality=args.print_freq),

                "cv_cont_sim_vid_m":     AverageMeter(locality=args.print_freq),
                "cv_cont_sim_vid_s":     AverageMeter(locality=args.print_freq),
                "cv_cont_sim_sk_m":      AverageMeter(locality=args.print_freq),
                "cv_cont_sim_sk_s":      AverageMeter(locality=args.print_freq),

                "sk_prox_reg_sim":       AverageMeter(locality=args.print_freq),
                "vid_prox_reg_sim":      AverageMeter(locality=args.print_freq),

                "prox_reg_loss":         AverageMeter(locality=args.print_freq),
                "accuracy_sk":           {"top1": AverageMeter(locality=args.print_freq),
                                          "top3": AverageMeter(locality=args.print_freq),
                                          "top5": AverageMeter(locality=args.print_freq)},
                "accuracy_rgb":          {"top1": AverageMeter(locality=args.print_freq),
                                          "top3": AverageMeter(locality=args.print_freq),
                                          "top5": AverageMeter(locality=args.print_freq)},
                }

    alph = args.memory_update_rate
    delta = args.prox_reg_multiplier

    model.train()

    tic = time.perf_counter()
    start_time = time.perf_counter()

    for idx, out in enumerate(data_loader):
        bat_idxs, vid_seq, sk_seq = out

        batch_size = vid_seq.size(0)

        tr_stats["time_data_loading"].update(time.perf_counter() - start_time)

        # Visualize images for tensorboard on two iterations.
        if (iteration == 0) or (iteration == args.print_freq):
            write_out_images(vid_seq, stat_writer, iteration)

        # Memory selection.
        start_time = time.perf_counter()

        mem_vid, mem_sk, mem_vid_cont, mem_sk_cont = queued_memories(memories, bat_idxs,
                                                                     args.memory_contrast * len(args.gpu), mem_queue)

        tr_stats["time_memory_selection"].update(time.perf_counter() - start_time)

        # Cuda Transfer
        start_time = time.perf_counter()

        vid_seq = vid_seq.to(cuda_device)
        sk_seq = sk_seq.to(cuda_device)

        mem_vid = mem_vid.to(cuda_device)
        mem_sk = mem_sk.to(cuda_device)
        mem_vid_cont = mem_vid_cont.to(cuda_device)
        mem_sk_cont = mem_sk_cont.to(cuda_device)

        tr_stats["time_cuda_transfer"].update(time.perf_counter() - start_time)

        # Forward pass: Calculation
        start_time = time.perf_counter()

        score, targets, rep_vid, rep_sk = model(vid_seq, sk_seq, mem_vid, mem_sk, mem_vid_cont, mem_sk_cont)

        # Forward pass: Unpack results
        score_sk_to_rgb = score["sk_to_rgb"]
        score_rgb_to_sk = score["rgb_to_sk"]

        targets_sk = targets["sk_to_rgb"]
        targets_rgb = targets["rgb_to_sk"]

        targets_sk = targets_sk.detach()
        targets_rgb = targets_rgb.detach()

        tr_stats["time_forward"].update(time.perf_counter() - start_time)

        # Memory update and statistics.
        start_time = time.perf_counter()

        mem_sk_old = memories["skeleton"][bat_idxs]
        mem_rgb_old = memories["video"][bat_idxs]

        calculate_statistics_mem_contrast(rep_vid, rep_sk, mem_vid, mem_sk, mem_vid_cont, mem_sk_cont, tr_stats)

        if delta is not None:
            sk_prox_reg_dist = torch.mean(torch.norm(rep_sk - mem_sk_old.to(rep_sk.device), dim=1), dim=0)
            rgb_prox_reg_dist = torch.mean(torch.norm(rep_vid - mem_rgb_old.to(rep_vid.device), dim=1), dim=0)

            prox_reg_loss = delta * (torch.mean(torch.stack([sk_prox_reg_dist, rgb_prox_reg_dist])))
            tr_stats["prox_reg_loss"].update(prox_reg_loss.item(), batch_size)
        else:
            tr_stats["prox_reg_loss"].update(None, batch_size)
            prox_reg_loss = None

        mem_sk_new = alph * rep_sk.clone().detach().cpu() + (1. - alph) * mem_sk_old
        mem_sk_new.requires_grad = False

        mem_rgb_new = alph * rep_vid.clone().detach().cpu() + (1. - alph) * mem_rgb_old
        mem_rgb_new.requires_grad = False

        memories["skeleton"][bat_idxs] = mem_sk_new
        memories["video"][bat_idxs] = mem_rgb_new

        tr_stats["time_memory_update"].update(time.perf_counter() - start_time)

        # Calculate Accuracies
        top1_rgb, top3_rgb, top5_rgb = calc_topk_accuracy(score_rgb_to_sk, targets_rgb, (1, 3, 5))
        top1_sk, top3_sk, top5_sk = calc_topk_accuracy(score_sk_to_rgb, targets_sk, (1, 3, 5))

        tr_stats["accuracy_sk"]["top1"].update(top1_sk.item(), batch_size)
        tr_stats["accuracy_sk"]["top3"].update(top3_sk.item(), batch_size)
        tr_stats["accuracy_sk"]["top5"].update(top5_sk.item(), batch_size)

        tr_stats["accuracy_rgb"]["top1"].update(top1_rgb.item(), batch_size)
        tr_stats["accuracy_rgb"]["top3"].update(top3_rgb.item(), batch_size)
        tr_stats["accuracy_rgb"]["top5"].update(top5_rgb.item(), batch_size)

        # Loss Calculation and backward pass.
        start_time = time.perf_counter()

        loss_rgb = criterion(score_rgb_to_sk, targets_rgb)
        loss_sk = criterion(score_sk_to_rgb, targets_sk)

        total_loss = torch.mean(torch.stack([loss_sk, loss_rgb])) if prox_reg_loss is None else torch.mean(
            torch.stack([loss_sk, loss_rgb, prox_reg_loss]))

        tr_stats["vid_loss"].update(loss_rgb.item(), batch_size)
        tr_stats["sk_loss"].update(loss_sk.item(), batch_size)
        tr_stats["total_loss"].update(total_loss.item(), batch_size)

        optimizer.zero_grad()
        total_loss.backward()

        optimizer.step()

        tr_stats["time_backward"].update(time.perf_counter() - start_time)

        if idx % args.print_freq == 0:
            print_tr_stats_iteration_mem_contrast(tr_stats, epoch, idx, len(data_loader), time.perf_counter() - tic)

        iteration += 1
        write_stats_mem_contrast_iteration(tr_stats, stat_writer, iteration, args)

        start_time = time.perf_counter()
        tic = time.perf_counter()
        # Next iteration

    # write_stats_mem_contrast_epoch(tr_stats, writer_train, epoch)
    print_stats_timings_mem_contrast(tr_stats)

    avg_acc = (tr_stats["accuracy_sk"]["top1"].val + tr_stats["accuracy_rgb"]["top1"].val) / 2.

    return iteration, avg_acc


def write_stats_mem_contrast_epoch(tr_stats, writer_train, epoch, args):
    # save curves
    avg_acc = (tr_stats["accuracy_sk"]["top1"].val + tr_stats["accuracy_rgb"]["top1"].val) / 2.

    losses_dict = {'loss':           tr_stats["total_loss"].val,
                   'loss_vid_to_sk': tr_stats["vid_loss"].val,
                   'loss_sk_to_vid': tr_stats["sk_loss"].val}

    if args.prox_reg_multiplier:
        losses_dict['loss_prox_reg'] = tr_stats["prox_reg_loss"].val

    writer_train.add_scalars('training_losses', losses_dict, epoch)

    writer_train.add_scalars('vid_to_sk_accuracy', {'top1': tr_stats["accuracy_rgb"]["top1"].val,
                                                    'top3': tr_stats["accuracy_rgb"]["top3"].val,
                                                    'top5': tr_stats["accuracy_rgb"]["top5"].val}, epoch)

    writer_train.add_scalars('sk_to_vid_accuracy', {'top1': tr_stats["accuracy_sk"]["top1"].val,
                                                    'top3': tr_stats["accuracy_sk"]["top3"].val,
                                                    'top5': tr_stats["accuracy_sk"]["top5"].val}, epoch)

    writer_train.add_scalars('rep_similarities', {'cross_view_tp':       tr_stats["cross_view_sim"].val,
                                                  'cross_view_noise':    tr_stats["cross_view_noise_sim"].val,
                                                  'vid_to_sk_mem_noise': tr_stats["cross_view_contrast_sim_rgb"].val,
                                                  'sk_to_vid_mem_noise': tr_stats["cross_view_contrast_sim_sk"].val,
                                                  'vid_to_mem':          tr_stats["vid_prox_reg_sim"].val,
                                                  'sk_to_mem':           tr_stats["sk_prox_reg_sim"].val}, epoch)


def write_stats_mem_contrast_iteration(tr_stats, writer_train, iteration, args):
    # save curves
    avg_acc = (tr_stats["accuracy_sk"]["top1"].local_avg + tr_stats["accuracy_rgb"]["top1"].local_avg) / 2.

    losses_dict = {'loss':           tr_stats["total_loss"].local_avg,
                   'loss_vid_to_sk': tr_stats["vid_loss"].local_avg,
                   'loss_sk_to_vid': tr_stats["sk_loss"].local_avg}

    if args.prox_reg_multiplier:
        losses_dict['loss_prox_reg'] = tr_stats["prox_reg_loss"].local_avg

    writer_train.add_scalars('it/training_losses', losses_dict, iteration)

    writer_train.add_scalars('it/vid_to_sk_accuracy', {'top1': tr_stats["accuracy_rgb"]["top1"].local_avg,
                                                       'top3': tr_stats["accuracy_rgb"]["top3"].local_avg,
                                                       'top5': tr_stats["accuracy_rgb"]["top5"].local_avg},
                             iteration)

    writer_train.add_scalars('it/sk_to_vid_accuracy', {'top1': tr_stats["accuracy_sk"]["top1"].local_avg,
                                                       'top3': tr_stats["accuracy_sk"]["top3"].local_avg,
                                                       'top5': tr_stats["accuracy_sk"]["top5"].local_avg},
                             iteration)

    rep_sim_dict = {'cv_tp_sim_mean':   tr_stats["cv_rep_sim_m"].local_avg,
                    'cv_rand_sim_mean': tr_stats["cv_rand_sim_m"].local_avg,
                    'cv_tp_sim_std':    tr_stats["cv_rep_sim_s"].local_avg,
                    'cv_rnd_sim_std':   tr_stats["cv_rand_sim_s"].local_avg,
                    }

    writer_train.add_scalars('it/cv_new_rep_sim', rep_sim_dict, iteration)

    cont_sim_dict = {'cv_vid_to_sk_cont_mean': tr_stats["cv_cont_sim_vid_m"].local_avg,
                     'cv_vid_to_sk_cont_std':  tr_stats["cv_cont_sim_vid_s"].local_avg,
                     'cv_sk_to_vid_cont_mean': tr_stats["cv_cont_sim_vid_m"].local_avg,
                     'cv_sk_to_vid_cont_std':  tr_stats["cv_cont_sim_vid_s"].local_avg,
                     }

    writer_train.add_scalars('it/cv_cont_rep_sim', cont_sim_dict, iteration)

    mem_sim_dict = {
        'vid_to_mem': tr_stats["vid_prox_reg_sim"].local_avg,
        'sk_to_mem':  tr_stats["sk_prox_reg_sim"].local_avg
        }

    writer_train.add_scalars('it/mem_rep_sim', mem_sim_dict, iteration)


def print_stats_timings_mem_contrast(tr_stats):
    print(
        f'Data Loading: {tr_stats["time_data_loading"].val:.4f}s | '
        f'Cuda Transfer: {tr_stats["time_cuda_transfer"].val:.4f}s | '
        f'Memory Selection: {tr_stats["time_memory_selection"].val:.4f}s | '
        f'Forward: {tr_stats["time_forward"].val:.4f}s | '
        f'Backward: {tr_stats["time_backward"].val:.4f}s | '
        f'Memory Update: {tr_stats["time_memory_update"].val:.4f}s')


def print_tr_stats_iteration_mem_contrast(stats: dict, epoch, idx, batch_count, duration):
    prox_reg_str = f'Prox Reg {stats["prox_reg_loss"].local_avg:.2f} ' if stats["prox_reg_loss"].val is not None else ''

    cv_sim_diff = stats["cv_rep_sim_m"].local_avg - stats["cv_rand_sim_m"].local_avg

    print(f'Epoch: [{epoch}][{idx}/{batch_count}]\t '
          f'Losses Total {stats["total_loss"].local_avg:.6f} '
          f'SK {stats["sk_loss"].local_avg:.2f} '
          f'Vid {stats["vid_loss"].local_avg:.2f} ' + prox_reg_str +
          '\tSk-RGB RGB-Sk Acc: '
          f'top1 {stats["accuracy_sk"]["top1"].local_avg:.4f} {stats["accuracy_rgb"]["top1"].local_avg:.4f} | '
          f'top3 {stats["accuracy_sk"]["top3"].local_avg:.4f} {stats["accuracy_rgb"]["top3"].local_avg:.4f} | '
          f'top5 {stats["accuracy_sk"]["top5"].local_avg:.4f} {stats["accuracy_rgb"]["top5"].local_avg:.4f} | '
          f'T:{duration:.2f}\n'
          '                     '  # For alignment 
          f'CV Sim M {stats["cv_rep_sim_m"].local_avg:.3f} ({cv_sim_diff:+1.0e}) '
          f'S {stats["cv_rep_sim_s"].local_avg:1.0e} \n'
          '                     '
          f'CV Rnd M {stats["cv_rand_sim_m"].local_avg:.3f}          S {stats["cv_rand_sim_s"].local_avg:1.0e} \n'
          '                     '  # For alignment 
          f'Vid-SK Cont M {stats["cv_cont_sim_vid_m"].local_avg:.3f} S {stats["cv_cont_sim_vid_s"].local_avg:1.0e} \n'
          '                     '  # For alignment 
          f'SK-Vid Cont M {stats["cv_cont_sim_sk_m"].local_avg:.3f} S {stats["cv_cont_sim_sk_s"].local_avg:1.0e} \n'
          '                     '
          f'SK Prox Sim {stats["sk_prox_reg_sim"].local_avg:.3f} | '
          f'RGB Prox Sim {stats["vid_prox_reg_sim"].local_avg:.3f}\n')


def calculate_statistics_mem_contrast(rep_vid: torch.Tensor, rep_sk: torch.Tensor, mem_vid: torch.Tensor,
                                      mem_sk: torch.Tensor,
                                      mem_vid_cont: torch.Tensor, mem_sk_cont: torch.Tensor, tr_stats):
    batch_size = rep_vid.shape[0]

    # Statistics on the calculated reps.
    cross_view_sim = torch.sum(rep_vid * rep_sk, dim=1)

    cross_view_sim_m = torch.mean(cross_view_sim, dim=0)
    cross_view_sim_s = torch.std(cross_view_sim, dim=0)

    tr_stats["cv_rep_sim_m"].update(cross_view_sim_m.item(), batch_size)
    tr_stats["cv_rep_sim_s"].update(cross_view_sim_s.item(), batch_size)

    # Assuming that the batches are random, thid dealigns matching views, making the new pairing random.
    # This provides a baseline for the similarity of random representations.
    rand_rep_sk = rep_sk.roll(shifts=1, dims=0)  # TODO: Check if copy or changed tensor
    cross_view_rand_sim = torch.sum(rep_vid * rand_rep_sk, dim=1)
    cross_view_rand_sim_m = torch.mean(cross_view_rand_sim, dim=0)
    cross_view_rand_sim_s = torch.std(cross_view_rand_sim, dim=0)

    tr_stats["cv_rand_sim_m"].update(cross_view_rand_sim_m.item(), batch_size)
    tr_stats["cv_rand_sim_s"].update(cross_view_rand_sim_s.item(), batch_size)

    intsct_count = min(mem_sk_cont.shape[0], rep_vid.shape[0])

    if intsct_count > 0:
        cv_contrast_sim_vid = torch.sum(rep_vid[:intsct_count] * mem_sk_cont[:intsct_count], dim=1)
        cv_contrast_sim_vid_m = torch.mean(cv_contrast_sim_vid, dim=0)
        cv_contrast_sim_vid_s = torch.std(cv_contrast_sim_vid, dim=0)

        cv_contrast_sim_sk = torch.sum(rep_sk[:intsct_count] * mem_vid_cont[:intsct_count], dim=1)
        cv_contrast_sim_sk_m = torch.mean(cv_contrast_sim_sk, dim=0)
        cv_contrast_sim_sk_s = torch.std(cv_contrast_sim_sk, dim=0)

        tr_stats["cv_cont_sim_vid_m"].update(cv_contrast_sim_vid_m.item(), batch_size)
        tr_stats["cv_cont_sim_vid_s"].update(cv_contrast_sim_vid_s.item(), batch_size)

        tr_stats["cv_cont_sim_sk_m"].update(cv_contrast_sim_sk_m.item(), batch_size)
        tr_stats["cv_cont_sim_sk_s"].update(cv_contrast_sim_sk_s.item(), batch_size)

    sk_prox_reg_sim = torch.mean(torch.sum(rep_sk * mem_sk.to(rep_sk.device), dim=1), dim=0)
    rgb_prox_reg_sim = torch.mean(torch.sum(rep_vid * mem_vid.to(rep_vid.device), dim=1), dim=0)

    tr_stats["sk_prox_reg_sim"].update(sk_prox_reg_sim.item(), batch_size)
    tr_stats["vid_prox_reg_sim"].update(rgb_prox_reg_sim.item(), batch_size)


def queued_memories(memories: dict, bat_idxs: torch.Tensor, count: int, mem_queue: deque):
    assert len(bat_idxs) + len(mem_queue) <= count

    cont_idxs = list(mem_queue)
    mem_queue.extend(bat_idxs.tolist())

    return memories["video"][bat_idxs], memories["skeleton"][bat_idxs], \
           memories["video"][cont_idxs], memories["skeleton"][cont_idxs]


def random_memories(memories: dict, bat_idxs: torch.Tensor, count: int):
    memory_count = memories["video"].shape[0]

    perm = list(range(memory_count))
    bat_idxs_lst = sorted(bat_idxs.tolist(), reverse=True)
    for bat_idx in bat_idxs_lst:
        perm.pop(bat_idx)

    random.shuffle(perm)

    rand_idxs = torch.tensor(perm[:count - len(bat_idxs)])

    return memories["video"][bat_idxs], memories["skeleton"][bat_idxs], \
           memories["video"][rand_idxs], memories["skeleton"][rand_idxs]
