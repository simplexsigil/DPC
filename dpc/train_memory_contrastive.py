import random
import time
from collections import deque

import torch

import train_batch_contrastive as tbc
from loss_functions import memory_contrast_scores
from utils import AverageMeter, write_out_images, calc_topk_accuracy, write_out_checkpoint


def training_loop(model, optimizer, criterion, train_loader, val_loader, writer_train, writer_val, args, cuda_device):
    iteration = args.start_iteration
    best_val_acc = args.best_val_acc
    best_val_loss = args.best_val_loss
    best_train_acc = args.best_train_acc
    best_train_loss = args.best_train_loss

    memories, mem_queue = initialize_memories(len(train_loader.dataset), args.representation_size, args.memory_contrast,
                                              len(args.gpu), args.batch_size)

    # Main loop
    for epoch in range(args.start_epoch, args.epochs):
        iteration, train_loss, train_acc = train_skvid_mem_contrast(train_loader, memories, mem_queue, model, optimizer,
                                                                    criterion,
                                                                    epoch, iteration, args, writer_train, cuda_device)

        val_loss, val_acc = tbc.validate(val_loader, model, criterion, cuda_device,
                                         epoch, args, writer_val)

        if args.use_dali:
            train_loader.reset()
            val_loader.reset()

        best_val_acc = val_acc if best_val_acc is None or val_acc > best_val_acc else best_val_acc
        best_val_loss = val_loss if best_val_loss is None or val_loss < best_val_loss else best_val_loss

        best_train_acc = train_acc if best_train_acc is None or train_acc > best_train_acc else best_train_acc
        best_train_loss = train_loss if best_train_loss is None or train_loss < best_train_loss else best_train_loss

        write_out_checkpoint(epoch, iteration, model, optimizer, args,
                             train_loss, train_acc, val_loss, val_acc,
                             best_train_loss, best_train_acc, best_val_loss, best_val_acc)

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
    tr_stats = {"time_data_loading":       AverageMeter(locality=args.print_freq),
                "time_cuda_transfer":      AverageMeter(locality=args.print_freq),
                "time_memory_selection":   AverageMeter(locality=args.print_freq),
                "time_memory_update":      AverageMeter(locality=args.print_freq),
                "time_forward":            AverageMeter(locality=args.print_freq),
                "time_scoring":            AverageMeter(locality=args.print_freq),
                "time_backward":           AverageMeter(locality=args.print_freq),
                "time_all":                AverageMeter(locality=args.print_freq),

                "total_loss":              AverageMeter(locality=args.print_freq),
                "sk_loss":                 AverageMeter(locality=args.print_freq),
                "vid_loss":                AverageMeter(locality=args.print_freq),

                "cv_rep_sim_m":            AverageMeter(locality=args.print_freq),
                "cv_rep_sim_s":            AverageMeter(locality=args.print_freq),
                "cv_rand_sim_m":           AverageMeter(locality=args.print_freq),
                "cv_rand_sim_s":           AverageMeter(locality=args.print_freq),

                "cv_rep_sim_vid_sk_mem_m": AverageMeter(locality=args.print_freq),
                "cv_rep_sim_vid_sk_mem_s": AverageMeter(locality=args.print_freq),
                "cv_rep_sim_sk_vid_mem_m": AverageMeter(locality=args.print_freq),
                "cv_rep_sim_sk_vid_mem_s": AverageMeter(locality=args.print_freq),

                "cv_cont_sim_vid_m":       AverageMeter(locality=args.print_freq),
                "cv_cont_sim_vid_s":       AverageMeter(locality=args.print_freq),
                "cv_cont_sim_sk_m":        AverageMeter(locality=args.print_freq),
                "cv_cont_sim_sk_s":        AverageMeter(locality=args.print_freq),

                "sk_prox_reg_sim":         AverageMeter(locality=args.print_freq),
                "vid_prox_reg_sim":        AverageMeter(locality=args.print_freq),

                "prox_reg_loss":           AverageMeter(locality=args.print_freq),
                "accuracy_sk":             {"top1": AverageMeter(locality=args.print_freq),
                                            "top3": AverageMeter(locality=args.print_freq),
                                            "top5": AverageMeter(locality=args.print_freq)},
                "accuracy_vid":            {"top1": AverageMeter(locality=args.print_freq),
                                            "top3": AverageMeter(locality=args.print_freq),
                                            "top5": AverageMeter(locality=args.print_freq)},
                }

    alph = args.memory_update_rate
    delta = args.prox_reg_multiplier

    model.train()

    time_all = time.perf_counter()
    start_time = time.perf_counter()

    for idx, out in enumerate(data_loader):
        bat_idxs, vid_seq, sk_seq = out

        batch_size = vid_seq.size(0)

        tr_stats["time_data_loading"].update(time.perf_counter() - start_time)

        # Visualize images for tensorboard on two iterations.
        if (iteration == 0) or (iteration == args.print_freq):
            write_out_images(vid_seq, stat_writer, iteration, args.img_dim)

        # Memory selection.
        start_time = time.perf_counter()

        if args.use_new_tp:
            mem_vid, mem_sk, mem_vid_cont, mem_sk_cont = queued_memories(memories, bat_idxs,
                                                                         args.memory_contrast,
                                                                         mem_queue)

            mem_vid_cont = mem_vid_cont.repeat((len(args.gpu), 1))
            mem_sk_cont = mem_sk_cont.repeat((len(args.gpu), 1))
        else:
            mem_vid, mem_sk, mem_vid_cont, mem_sk_cont = random_memories(memories, bat_idxs,
                                                                         args.memory_contrast)

            mem_vid_cont = mem_vid_cont.repeat((len(args.gpu), 1))
            mem_sk_cont = mem_sk_cont.repeat((len(args.gpu), 1))

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

        rep_vid, rep_sk = model(vid_seq, sk_seq, mem_vid, mem_sk, mem_vid_cont, mem_sk_cont,
                                no_scoring=True)

        tr_stats["time_forward"].update(time.perf_counter() - start_time)

        start_time = time.perf_counter()

        # Memory Contrast
        (score_rgb_to_sk, score_sk_to_rgb,
         targets_sk, targets_rgb) = memory_contrast_scores(x=rep_vid, y=rep_sk,
                                                           x_mem=mem_vid,
                                                           y_mem=mem_sk,
                                                           x_cont=mem_vid_cont,
                                                           y_cont=mem_sk_cont,
                                                           matching_fn=args.score_function,
                                                           contrast_type=args.memory_contrast_type)

        tr_stats["time_scoring"].update(time.perf_counter() - start_time)

        # Memory update and statistics.
        start_time = time.perf_counter()

        calculate_statistics_mem_contrast(rep_vid, rep_sk, mem_vid, mem_sk, mem_vid_cont, mem_sk_cont, tr_stats)

        mem_sk_old = memories["skeleton"][bat_idxs]
        mem_rgb_old = memories["video"][bat_idxs]

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
        top1_rgb, top3_rgb, top5_rgb = calc_topk_accuracy(score_rgb_to_sk, targets_rgb,
                                                          (1, min(3, batch_size), min(5, batch_size)))
        top1_sk, top3_sk, top5_sk = calc_topk_accuracy(score_sk_to_rgb, targets_sk,
                                                       (1, min(3, batch_size), min(5, batch_size)))

        tr_stats["accuracy_sk"]["top1"].update(top1_sk.item(), batch_size)
        tr_stats["accuracy_sk"]["top3"].update(top3_sk.item(), batch_size)
        tr_stats["accuracy_sk"]["top5"].update(top5_sk.item(), batch_size)

        tr_stats["accuracy_vid"]["top1"].update(top1_rgb.item(), batch_size)
        tr_stats["accuracy_vid"]["top3"].update(top3_rgb.item(), batch_size)
        tr_stats["accuracy_vid"]["top5"].update(top5_rgb.item(), batch_size)

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
        tr_stats["time_all"].update(time.perf_counter() - time_all)

        if idx % args.print_freq == 0:
            print_tr_stats_iteration_mem_contrast(tr_stats, epoch, idx, len(data_loader),
                                                  time.perf_counter() - time_all)
            write_stats_mem_contrast_iteration(tr_stats, stat_writer, iteration, args)

        iteration += 1

        start_time = time.perf_counter()
        time_all = time.perf_counter()
        # Next iteration

    write_stats_mem_contrast_epoch(tr_stats, stat_writer, epoch, args)
    print_stats_timings_mem_contrast(tr_stats)

    avg_acc = (tr_stats["accuracy_sk"]["top1"].avg + tr_stats["accuracy_vid"]["top1"].avg) / 2.

    return iteration, tr_stats["total_loss"].avg, avg_acc


def write_stats_mem_contrast_epoch(tr_stats, writer_train, epoch, args):
    # save curves
    avg_acc = (tr_stats["accuracy_sk"]["top1"].avg + tr_stats["accuracy_vid"]["top1"].avg) / 2.

    writer_train.add_scalars('ep/Accuracies', {'Train Acc': avg_acc}, epoch)

    losses_dict = {'Train Loss':           tr_stats["total_loss"].avg,
                   'Train Loss Vid to Sk': tr_stats["vid_loss"].avg,
                   'Train Loss Sk to Vid': tr_stats["sk_loss"].avg}

    if args.prox_reg_multiplier:
        losses_dict['Proximal Regularization Loss'] = tr_stats["prox_reg_loss"].avg

    writer_train.add_scalars('ep/Losses', losses_dict, epoch)

    writer_train.add_scalars('ep/Accuracies_Vid_to_Sk', {'top1': tr_stats["accuracy_vid"]["top1"].avg,
                                                         'top3': tr_stats["accuracy_vid"]["top3"].avg,
                                                         'top5': tr_stats["accuracy_vid"]["top5"].avg}, epoch)

    writer_train.add_scalars('ep/Accuracies_Sk_to_Vid', {'top1': tr_stats["accuracy_sk"]["top1"].avg,
                                                         'top3': tr_stats["accuracy_sk"]["top3"].avg,
                                                         'top5': tr_stats["accuracy_sk"]["top5"].avg}, epoch)


def write_stats_mem_contrast_iteration(tr_stats, writer_train, iteration, args):
    # save curves
    avg_acc = (tr_stats["accuracy_sk"]["top1"].local_avg + tr_stats["accuracy_vid"]["top1"].local_avg) / 2.

    losses_dict = {'Loss':           tr_stats["total_loss"].local_avg,
                   'Loss Vid to Sk': tr_stats["vid_loss"].local_avg,
                   'Loss Sk to Vid': tr_stats["sk_loss"].local_avg}

    if args.prox_reg_multiplier:
        losses_dict['Proximal Regularization Loss'] = tr_stats["prox_reg_loss"].local_avg

    writer_train.add_scalars('it/Losses', losses_dict, iteration)

    writer_train.add_scalars('it/Vid to Sk Contrast Accuracy', {'top1': tr_stats["accuracy_vid"]["top1"].local_avg,
                                                                'top3': tr_stats["accuracy_vid"]["top3"].local_avg,
                                                                'top5': tr_stats["accuracy_vid"]["top5"].local_avg},
                             iteration)

    writer_train.add_scalars('it/Sk to Vid Contrast Accuracy', {'top1': tr_stats["accuracy_sk"]["top1"].local_avg,
                                                                'top3': tr_stats["accuracy_sk"]["top3"].local_avg,
                                                                'top5': tr_stats["accuracy_sk"]["top5"].local_avg},
                             iteration)

    rep_sim_dict = {'Current Rep Sim (Mean)':      tr_stats["cv_rep_sim_m"].local_avg,
                    'Rand Current Rep Sim (Mean)': tr_stats["cv_rand_sim_m"].local_avg,
                    'Current Rep Sim (Std)':       tr_stats["cv_rep_sim_s"].local_avg,
                    'Rand Current Rep Sim (Std)':  tr_stats["cv_rand_sim_s"].local_avg,
                    }

    writer_train.add_scalars('it/Current Cross View Rep Similarities', rep_sim_dict, iteration)

    cv_vid_sk_mem_sim_dict = {'Vid to Sk Mem Sim (Mean)':      tr_stats["cv_rep_sim_vid_sk_mem_m"].local_avg,
                              'Vid to Sk Mem Sim (Std)':       tr_stats["cv_rep_sim_vid_sk_mem_s"].local_avg,
                              'Vid to Sk Mem Rand Sim (Mean)': tr_stats["cv_cont_sim_vid_m"].local_avg,
                              'Vid to Sk Mem Rand Sim (Std)':  tr_stats["cv_cont_sim_vid_s"].local_avg
                              }

    writer_train.add_scalars('it/Current Vid Rep to Sk Mem', cv_vid_sk_mem_sim_dict, iteration)

    cv_sk_vid_mem_sim_dict = {
        'Sk to Vid Mem Sim (Mean)':      tr_stats["cv_rep_sim_sk_vid_mem_m"].local_avg,
        'Sk to Vid Mem Sim (Std)':       tr_stats["cv_rep_sim_sk_vid_mem_s"].local_avg,
        'Sk to Vid Mem Rand Sim (Mean)': tr_stats["cv_cont_sim_sk_m"].local_avg,
        'Sk to Vid Mem Rand Sim (Std)':  tr_stats["cv_cont_sim_sk_s"].local_avg,
        }

    writer_train.add_scalars('it/Current Sk Rep to Vid Mem', cv_sk_vid_mem_sim_dict, iteration)

    mem_sim_dict = {
        'Vid to Mem': tr_stats["vid_prox_reg_sim"].local_avg,
        'Sk to Mem':  tr_stats["sk_prox_reg_sim"].local_avg
        }

    writer_train.add_scalars('it/Rep to own Mem Sim', mem_sim_dict, iteration)

    all_calced_timings = sum([tr_stats[tms].local_avg for tms in ["time_data_loading",
                                                                  "time_memory_selection",
                                                                  "time_cuda_transfer",
                                                                  "time_forward",
                                                                  "time_scoring",
                                                                  "time_memory_update",
                                                                  "time_backward",
                                                                  ]])
    timing_dict = {'Loading Data':                       tr_stats["time_data_loading"].local_avg,
                   'Memory Selection':                   tr_stats["time_memory_selection"].local_avg,
                   'Cuda Transfer':                      tr_stats["time_cuda_transfer"].local_avg,
                   'Forward Pass':                       tr_stats["time_forward"].local_avg,
                   'Scoring':                            tr_stats["time_scoring"].local_avg,
                   'Memory Update':                      tr_stats["time_memory_update"].local_avg,
                   'Backward Pass':                      tr_stats["time_backward"].local_avg,
                   'Loading + Memory Selection + '
                   'Transfer + Forward + '
                   'Scoring + Memory Update + Backward': all_calced_timings,
                   'All':                                tr_stats["time_all"].local_avg
                   }

    writer_train.add_scalars('it/Batch-Wise_Timings', timing_dict, iteration)


def print_stats_timings_mem_contrast(tr_stats):
    print(
        f'Data Loading: {tr_stats["time_data_loading"].avg:.4f}s | '
        f'Cuda Transfer: {tr_stats["time_cuda_transfer"].avg:.4f}s | '
        f'Memory Selection: {tr_stats["time_memory_selection"].avg:.4f}s | '
        f'Forward: {tr_stats["time_forward"].avg:.4f}s | '
        f'Backward: {tr_stats["time_backward"].avg:.4f}s | '
        f'Memory Update: {tr_stats["time_memory_update"].avg:.4f}s | '
        f'All: {tr_stats["time_all"].avg:.4f}s')


def print_tr_stats_iteration_mem_contrast(stats: dict, epoch, idx, batch_count, duration):
    prox_reg_str = f'Prox Reg {stats["prox_reg_loss"].local_avg:.2f} ' if stats["prox_reg_loss"].val is not None else ''

    cv_sim_diff = stats["cv_rep_sim_m"].local_avg - stats["cv_rand_sim_m"].local_avg

    print(f'Epoch: [{epoch}][{idx}/{batch_count}]\t '
          f'Losses Total {stats["total_loss"].local_avg:.6f} '
          f'SK {stats["sk_loss"].local_avg:.2f} '
          f'Vid {stats["vid_loss"].local_avg:.2f} ' + prox_reg_str +
          '\tSk-RGB RGB-Sk Acc: '
          f'top1 {stats["accuracy_sk"]["top1"].local_avg:.4f} {stats["accuracy_vid"]["top1"].local_avg:.4f} | '
          f'top3 {stats["accuracy_sk"]["top3"].local_avg:.4f} {stats["accuracy_vid"]["top3"].local_avg:.4f} | '
          f'top5 {stats["accuracy_sk"]["top5"].local_avg:.4f} {stats["accuracy_vid"]["top5"].local_avg:.4f} | '
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


def get_cv_sim_stats(x: torch.Tensor, y: torch.Tensor):
    sim = torch.sum(x * y, dim=1)
    sim_m = torch.mean(sim, dim=0)
    sim_s = torch.std(sim, dim=0)

    return sim_m, sim_s


def calculate_statistics_mem_contrast(rep_vid: torch.Tensor, rep_sk: torch.Tensor, mem_vid: torch.Tensor,
                                      mem_sk: torch.Tensor,
                                      mem_vid_cont: torch.Tensor, mem_sk_cont: torch.Tensor, tr_stats):
    batch_size = rep_vid.shape[0]

    # Statistics on the calculated reps.

    # Sim between new representation true positives.
    cv_sim_m, cv_sim_s = get_cv_sim_stats(rep_vid, rep_sk)

    tr_stats["cv_rep_sim_m"].update(cv_sim_m.item(), batch_size)
    tr_stats["cv_rep_sim_s"].update(cv_sim_s.item(), batch_size)

    # Assuming that the batches are random, this dealigns matching views, making the new pairing random.
    # This provides a baseline for the similarity of randomly chosen representations.
    rep_sk_rnd = rep_sk.roll(shifts=1, dims=0)
    cv_rand_sim_m, cv_rand_sim_s = get_cv_sim_stats(rep_vid, rep_sk_rnd)

    tr_stats["cv_rand_sim_m"].update(cv_rand_sim_m.item(), batch_size)
    tr_stats["cv_rand_sim_s"].update(cv_rand_sim_s.item(), batch_size)

    cvm_vid_m, cvm_vid_s = get_cv_sim_stats(rep_vid, mem_sk.to(rep_vid.device))

    # Sim between representation and memory of true positive.
    tr_stats["cv_rep_sim_vid_sk_mem_m"].update(cvm_vid_m.item(), batch_size)
    tr_stats["cv_rep_sim_vid_sk_mem_s"].update(cvm_vid_s.item(), batch_size)

    cvm_sk_m, cvm_sk_s = get_cv_sim_stats(rep_sk, mem_vid.to(rep_sk.device))

    tr_stats["cv_rep_sim_sk_vid_mem_m"].update(cvm_sk_m.item(), batch_size)
    tr_stats["cv_rep_sim_sk_vid_mem_s"].update(cvm_sk_s.item(), batch_size)

    # Sim between representation and random memory.
    intsct_count = min(rep_vid.shape[0], mem_vid_cont.shape[0])

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

    bat_idxs_lst = sorted(bat_idxs.tolist(), reverse=True)
    perm = list(range(memory_count))

    for bat_idx in bat_idxs_lst:
        perm.pop(bat_idx)

    random.shuffle(perm)

    rand_idxs = torch.tensor(perm[:count - 1])

    return memories["video"][bat_idxs], memories["skeleton"][bat_idxs], \
           memories["video"][rand_idxs], memories["skeleton"][rand_idxs]
