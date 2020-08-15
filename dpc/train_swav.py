import os
import time

import torch

import loss_functions
from utils import AverageMeter, calc_topk_accuracy, write_out_checkpoint, write_out_images


def training_loop(model, optimizer, lr_schedule, criterion, train_loader, val_loader, writer_train, writer_val, args,
                  cuda_device):
    iteration = args.start_iteration
    best_val_acc = args.best_val_acc
    best_val_loss = args.best_val_loss

    best_projection_sim = args.best_projection_sim
    best_train_loss = args.best_train_loss

    # build the queue
    queue = None
    queue_path = os.path.join(args.model_path, "queue.pth")
    if os.path.isfile(queue_path):
        queue = torch.load(queue_path)["queue"]
    # the queue needs to be divisible by the batch size
    args.queue_length -= args.queue_length % args.batch_size

    # Main loop
    for epoch in range(args.start_epoch, args.epochs):
        iteration, train_loss, projection_sim = train_skvid_swav(train_loader, model, optimizer, lr_schedule, epoch,
                                                                 iteration, args, writer_train, cuda_device, queue)

        val_loss, val_acc = validate(val_loader, model, criterion, cuda_device,
                                     epoch, args, writer_val)

        best_val_acc = val_acc if best_val_acc is None or val_acc > best_val_acc else best_val_acc
        best_val_loss = val_loss if best_val_loss is None or val_loss < best_val_loss else best_val_loss

        best_projection_sim = projection_sim if best_projection_sim is None or projection_sim > best_projection_sim else projection_sim
        best_train_loss = train_loss if best_train_loss is None or train_loss < best_train_loss else best_train_loss

        write_out_checkpoint(epoch, iteration, model, optimizer, args,
                             train_loss, projection_sim, val_loss, val_acc,
                             best_train_loss, best_projection_sim, best_val_loss, best_val_acc)

        if args.use_dali:
            train_loader.reset()
            val_loader.reset()

    print('Training from ep %d to ep %d finished' % (args.start_epoch, args.epochs))


def train_skvid_swav(data_loader, model, optimizer, lr_schedule, epoch, iteration, args, writer_train,
                     cuda_device, queue=None):
    tr_stats = {"time_data_loading":  AverageMeter(locality=args.print_freq),
                "time_cuda_transfer": AverageMeter(locality=args.print_freq),
                "time_forward":       AverageMeter(locality=args.print_freq),
                "time_scoring":       AverageMeter(locality=args.print_freq),
                "time_backward":      AverageMeter(locality=args.print_freq),
                "time_all":           AverageMeter(locality=args.print_freq),

                "cv_rep_sim_m":       AverageMeter(locality=args.print_freq),
                "cv_rep_sim_s":       AverageMeter(locality=args.print_freq),
                "cv_rand_sim_m":      AverageMeter(locality=args.print_freq),
                "cv_rand_sim_s":      AverageMeter(locality=args.print_freq),

                "cv_proj_sim_m":      AverageMeter(locality=args.print_freq),
                "cv_proj_sim_s":      AverageMeter(locality=args.print_freq),
                "cv_proj_rnd_sim_m":  AverageMeter(locality=args.print_freq),
                "cv_proj_rnd_sim_s":  AverageMeter(locality=args.print_freq),

                "total_loss":         AverageMeter(locality=args.print_freq),

                "learning_rate":      AverageMeter(locality=args.print_freq),
                }

    model.train()

    dl_time = time.perf_counter()
    all_time = time.perf_counter()

    for idx, out in enumerate(data_loader):
        bat_idxs, vid_seq, sk_seq = out

        batch_size = vid_seq.size(0)

        tr_stats["time_data_loading"].update(time.perf_counter() - dl_time)

        if lr_schedule is not None:
            # update learning rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_schedule[iteration]

            tr_stats["learning_rate"].update(lr_schedule[iteration])
        else:
            tr_stats["learning_rate"].update(args.lr)

        # Visualize images for tensorboard.
        if iteration == 0:
            write_out_images(vid_seq, writer_train, iteration, img_dim=args.img_dim)

        # Cuda Transfer
        s_cud_time = time.perf_counter()

        vid_seq = vid_seq.to(cuda_device)
        sk_seq = sk_seq.to(cuda_device)

        e_cud_time = time.perf_counter()
        tr_stats["time_cuda_transfer"].update(e_cud_time - s_cud_time)

        # Forward pass: Calculation
        s_forw_time = time.perf_counter()

        # normalize the prototypes
        with torch.no_grad():
            # TODO: why clone? Because of grads?
            w = model.module.prototypes.weight.data.clone()
            w = torch.nn.functional.normalize(w, dim=1, p=2)
            model.module.prototypes.weight.copy_(w)

        repr_vid, repr_sk, repr_vid_proj, repr_sk_proj = model(vid_seq, sk_seq)
        repr_vid, repr_sk = repr_vid.detach(), repr_sk.detach()

        e_forw_time = time.perf_counter()
        tr_stats["time_forward"].update(e_forw_time - s_forw_time)

        # Scoring
        s_score_time = time.perf_counter()

        # ============ swav loss ... ============
        # Adapted from https://github.com/facebookresearch/swav/blob/master/main_swav.py

        # optionally starts a queue
        if args.queue_length > 0 and iteration >= args.iters_queue_starts and queue is None:
            queue = {}

            queue["vid"] = torch.zeros(args.queue_length, args.representation_size, requires_grad=True).cuda()
            queue["sk"] = torch.zeros(args.queue_length, args.representation_size, requires_grad=True).cuda()

        loss = 0

        all_vid_proj = repr_vid_proj
        all_sk_proj = repr_sk_proj

        with torch.no_grad():
            # time to use the queue
            if queue is not None:
                queue_vid = queue["vid"]
                queue_sk = queue["sk"]

                non_zero_row = torch.any(queue_vid != 0, dim=1)

                if torch.any(non_zero_row):
                    queue_vid_proj = torch.mm(queue_vid[non_zero_row], model.module.prototypes.weight.t())
                    queue_sk_proj = torch.mm(queue_sk[non_zero_row], model.module.prototypes.weight.t())

                    all_vid_proj = torch.cat((queue_vid_proj, repr_vid_proj))
                    all_sk_proj = torch.cat((queue_sk_proj, repr_sk_proj))

                # fill the queue
                queue_vid[batch_size:] = queue_vid[:-batch_size].clone()
                queue_vid[:batch_size] = repr_vid

                queue_sk[batch_size:] = queue_sk[:-batch_size].clone()
                queue_sk[:batch_size] = repr_sk

            # get assignments
            q_vid = torch.exp(all_vid_proj / args.sinkhorn_knopp_epsilon).t()
            q_vid = sinkhorn(q_vid, args.sinkhorn_iterations)
            q_vid = q_vid.t()[-batch_size:]

            q_sk = torch.exp(all_sk_proj / args.sinkhorn_knopp_epsilon).t()
            q_sk = sinkhorn(q_sk, args.sinkhorn_iterations)
            q_sk = q_sk.t()[-batch_size:]

        p_vid = torch.nn.functional.softmax(repr_vid_proj / args.swav_temperature, dim=1)
        p_sk = torch.nn.functional.softmax(repr_sk_proj / args.swav_temperature, dim=1)

        loss += torch.mean(torch.sum(-q_vid * torch.log(p_sk) - (1 - q_vid) * torch.log(1 - p_sk),
                                     dim=1))
        loss += torch.mean(torch.sum(-q_sk * torch.log(p_vid) - (1 - q_sk) * torch.log(1 - p_vid),
                                     dim=1))

        loss /= 2

        e_score_time = time.perf_counter()
        tr_stats["time_scoring"].update(e_score_time - s_score_time)

        with torch.no_grad():
            calculate_tr_stats(repr_vid, repr_sk, repr_vid_proj, repr_sk_proj, tr_stats)

        # Loss Calculation and backward pass.
        s_back_time = time.perf_counter()

        optimizer.zero_grad()

        tr_stats["total_loss"].update(loss.item(), batch_size)

        loss.backward()

        # cancel some gradients
        if iteration < args.iters_freeze_prototypes:
            for name, p in model.named_parameters():
                if "prototypes" in name:
                    p.grad = None

        optimizer.step()

        e_back_time = time.perf_counter()
        tr_stats["time_backward"].update(e_back_time - s_back_time)

        tr_stats["time_all"].update(time.perf_counter() - all_time)

        if idx % args.print_freq == 0:
            write_stats_swav_iteration(tr_stats, writer_train, iteration)
            print_tr_stats_loc_avg(tr_stats, epoch, idx, len(data_loader), tr_stats["time_all"].local_avg)

        iteration += 1

        dl_time = time.perf_counter()
        all_time = time.perf_counter()
        # Next iteration

    write_stats_swav_epoch(tr_stats, writer_train, epoch)
    print_tr_stats_timings_avg(tr_stats)

    return iteration, tr_stats["total_loss"].avg, tr_stats["cv_proj_sim_m"].avg


def sinkhorn(Q, nmb_iters):
    with torch.no_grad():
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        u = torch.zeros(Q.shape[0]).cuda(non_blocking=True)
        r = torch.ones(Q.shape[0]).cuda(non_blocking=True) / Q.shape[0]  # 1/k
        c = torch.ones(Q.shape[1]).cuda(non_blocking=True) / Q.shape[1]  # 1/b

        curr_sum = torch.sum(Q, dim=1)

        for it in range(nmb_iters):
            u = curr_sum
            Q *= (r / u).unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
            curr_sum = torch.sum(Q, dim=1)

        Q = (Q / torch.sum(Q, dim=0, keepdim=True)).float()
        return Q


def get_cv_sim_stats(x: torch.Tensor, y: torch.Tensor):
    sim = torch.sum(x * y, dim=1)
    sim_m = torch.mean(sim, dim=0)
    sim_s = torch.std(sim, dim=0)

    return sim_m, sim_s


def calculate_tr_stats(rep_vid: torch.Tensor, rep_sk: torch.Tensor, rep_vid_proj, rep_sk_proj, tr_stats):
    batch_size = rep_vid.shape[0]

    # Statistics on the calculated reps.
    cross_view_sim_m, cross_view_sim_s = get_cv_sim_stats(rep_vid, rep_sk)

    tr_stats["cv_rep_sim_m"].update(cross_view_sim_m.item(), batch_size)
    tr_stats["cv_rep_sim_s"].update(cross_view_sim_s.item(), batch_size)

    # Assuming that the batches are random, this dealigns matching views, making the new pairing random.
    # This provides a baseline for the similarity of random representations.
    rand_rep_sk = rep_sk.roll(shifts=1, dims=0)  # TODO: Check if copy or changed tensor
    cross_view_rand_sim_m, cross_view_rand_sim_s = get_cv_sim_stats(rep_vid, rand_rep_sk)

    tr_stats["cv_rand_sim_m"].update(cross_view_rand_sim_m.item(), batch_size)
    tr_stats["cv_rand_sim_s"].update(cross_view_rand_sim_s.item(), batch_size)

    rep_vid_proj_n = torch.nn.functional.normalize(rep_vid_proj, dim=1)
    rep_sk_proj_n = torch.nn.functional.normalize(rep_sk_proj, dim=1)

    cv_proj_sim_m, cv_proj_sim_s = get_cv_sim_stats(rep_vid_proj_n, rep_sk_proj_n)

    tr_stats["cv_proj_sim_m"].update(cv_proj_sim_m.item(), batch_size)
    tr_stats["cv_proj_sim_s"].update(cv_proj_sim_s.item(), batch_size)

    rand_rep_sk_proj = rep_sk_proj.roll(shifts=1, dims=0)
    rand_rep_sk_proj_n = torch.nn.functional.normalize(rand_rep_sk_proj, dim=1)

    cv_proj_rnd_sim_m, cv_proj_rnd_sim_s = get_cv_sim_stats(rep_vid_proj_n, rand_rep_sk_proj_n)

    tr_stats["cv_proj_rnd_sim_m"].update(cv_proj_rnd_sim_m.item(), batch_size)
    tr_stats["cv_proj_rnd_sim_s"].update(cv_proj_rnd_sim_s.item(), batch_size)


def write_stats_swav_epoch(tr_stats, writer_train, epoch):
    writer_train.add_scalars('ep/Losses', {'Training Loss': tr_stats["total_loss"].avg}, epoch)

    writer_train.add_scalars('ep/Projection Similarities',
                             {'Projection Similarity (Mean)':      tr_stats["cv_proj_sim_m"].avg,
                              'Projection Similarity (Std)':       tr_stats["cv_proj_sim_s"].avg,
                              'Rand Projection Similarity (Mean)': tr_stats["cv_proj_rnd_sim_m"].avg,
                              'Rand Projection Similarity (Std)':  tr_stats["cv_proj_rnd_sim_s"].avg,
                              }, epoch)

    rep_sim_dict = {'Representation Similarity (Mean)':      tr_stats["cv_rep_sim_m"].avg,
                    'Rand Representation Similarity (Mean)': tr_stats["cv_rand_sim_m"].avg,
                    'Representation Similarity (Std)':       tr_stats["cv_rep_sim_s"].avg,
                    'Rand Representation Similarity (Std)':  tr_stats["cv_rand_sim_s"].avg,
                    }

    writer_train.add_scalars('it/New_Representation_Similarities', rep_sim_dict, epoch)


def write_stats_swav_iteration(tr_stats, writer_train, iteration):
    writer_train.add_scalars('it/Train_Loss', {'loss': tr_stats["total_loss"].local_avg}, iteration)

    rep_sim_dict = {'Representation Similarity (Mean)':      tr_stats["cv_rep_sim_m"].local_avg,
                    'Rand Representation Similarity (Mean)': tr_stats["cv_rand_sim_m"].local_avg,
                    'Representation Similarity (Std)':       tr_stats["cv_rep_sim_s"].local_avg,
                    'Rand Representation Similarity (Std)':  tr_stats["cv_rand_sim_s"].local_avg,
                    }

    writer_train.add_scalars('it/New_Representation_Similarities', rep_sim_dict, iteration)

    writer_train.add_scalars('it/Projection Similarities',
                             {'Projection Similarity (Mean)':      tr_stats["cv_proj_sim_m"].local_avg,
                              'Projection Similarity (Std)':       tr_stats["cv_proj_sim_s"].local_avg,
                              'Rand Projection Similarity (Mean)': tr_stats["cv_proj_rnd_sim_m"].local_avg,
                              'Rand Projection Similarity (Std)':  tr_stats["cv_proj_rnd_sim_s"].local_avg,
                              }, iteration)

    all_calced_timings = sum([tr_stats[tms].local_avg for tms in ["time_data_loading",
                                                                  "time_cuda_transfer",
                                                                  "time_forward",
                                                                  "time_scoring",
                                                                  "time_backward",
                                                                  ]])
    timing_dict = {'Loading Data':                 tr_stats["time_data_loading"].local_avg,
                   'Cuda Transfer':                tr_stats["time_cuda_transfer"].local_avg,
                   'Forward Pass':                 tr_stats["time_forward"].local_avg,
                   'Scoring':                      tr_stats["time_scoring"].local_avg,
                   'Backward Pass':                tr_stats["time_backward"].local_avg,
                   'Loading + Transfer + '
                   'Forward + Scoring + Backward': all_calced_timings,
                   'All':                          tr_stats["time_all"].local_avg
                   }

    writer_train.add_scalars('it/Batch-Wise_Timings', timing_dict, iteration)

    writer_train.add_scalars('it/Learning Rate', {"Learning Rate": tr_stats["learning_rate"].local_avg}, iteration)


def print_tr_stats_loc_avg(stats: dict, epoch, idx, batch_count, duration):
    proj_sim_diff = stats["cv_proj_sim_m"].local_avg - stats["cv_proj_rnd_sim_m"].local_avg

    print(f'Epoch: [{epoch}][{idx}/{batch_count}]\t '
          f'Loss {stats["total_loss"].local_avg:.6f} '
          f'\tProjection Sim:  M {stats["cv_proj_sim_m"].local_avg:.4f} ({proj_sim_diff:+1.0e}) S {stats["cv_proj_sim_s"].local_avg:.4f} | '
          f'Rand Sim:  M {stats["cv_proj_rnd_sim_m"].local_avg:.4f} S {stats["cv_proj_rnd_sim_s"].local_avg:.4f} | '
          f'T:{duration:.2f}')


def print_tr_stats_timings_avg(tr_stats):
    print('Batch-wise Timings:\n'
          f'Data Loading: {tr_stats["time_data_loading"].avg:.4f}s | '
          f'Cuda Transfer: {tr_stats["time_cuda_transfer"].avg:.4f}s | '
          f'Forward: {tr_stats["time_forward"].avg:.4f}s | '
          f'Backward: {tr_stats["time_backward"].avg:.4f}s | '
          f'All: {tr_stats["time_all"].avg:.4f}s\n')


def validate(data_loader, model, criterion, cuda_device, epoch, args, writer_val):
    val_stats = {"time_data_loading":  AverageMeter(locality=args.print_freq),
                 "time_cuda_transfer": AverageMeter(locality=args.print_freq),
                 "time_forward":       AverageMeter(locality=args.print_freq),
                 "time_backward":      AverageMeter(locality=args.print_freq),
                 "time_all":           AverageMeter(locality=args.print_freq),

                 "total_loss":         AverageMeter(locality=args.print_freq),

                 "accuracy":           {"top1": AverageMeter(locality=args.print_freq),
                                        "top3": AverageMeter(locality=args.print_freq),
                                        "top5": AverageMeter(locality=args.print_freq)}
                 }

    model.eval()

    start_time = time.perf_counter()
    all_time = time.perf_counter()

    with torch.no_grad():
        for idx, out in enumerate(data_loader):
            bat_idx, vid_seq, sk_seq = out

            val_stats["time_data_loading"].update(time.perf_counter() - start_time)
            start_time = time.perf_counter()

            vid_seq = vid_seq.to(cuda_device)

            val_stats["time_cuda_transfer"].update(time.perf_counter() - start_time)
            start_time = time.perf_counter()

            batch_size = vid_seq.size(0)

            repr_vid, repr_sk, _, _ = model(vid_seq, sk_seq)

            val_stats["time_forward"].update(time.perf_counter() - start_time)

            score = loss_functions.pairwise_scores(x=repr_sk, y=repr_vid, matching_fn=args.score_function,
                                                   temp_tao=args.temperature)

            del vid_seq, sk_seq

            targets = list(range(len(score)))
            targets = torch.tensor(targets, requires_grad=False, dtype=torch.long).to(repr_vid.device)

            loss = criterion(score, targets)

            top1, top3, top5 = calc_topk_accuracy(score, targets, (1, 3, 4))

            val_stats["total_loss"].update(loss.item(), batch_size)
            val_stats["accuracy"]["top1"].update(top1.item(), batch_size)
            val_stats["accuracy"]["top3"].update(top3.item(), batch_size)
            val_stats["accuracy"]["top5"].update(top5.item(), batch_size)

            del score, targets, loss

            val_stats["time_all"].update(time.perf_counter() - all_time)

            start_time = time.perf_counter()
            all_time = time.perf_counter()

    print_val_avg(val_stats, epoch, args)

    write_val_stats_avg(val_stats, writer_val, epoch)

    return val_stats["total_loss"].avg, val_stats["accuracy"]["top1"].avg


def print_val_avg(val_stats, epoch, args):
    print(f'[{epoch}/{args.epochs}] Loss {val_stats["total_loss"].avg:.4f}\t'
          f'Acc: '
          f'top1 {val_stats["accuracy"]["top1"].avg:.4f}; '
          f'top3 {val_stats["accuracy"]["top3"].avg:.4f}; '
          f'top5 {val_stats["accuracy"]["top5"].avg:.4f}\n')


def write_val_stats_avg(val_stats, writer_val, epoch):
    writer_val.add_scalars('ep/Losses', {'Validation Loss': val_stats["total_loss"].avg}, epoch)

    writer_val.add_scalars('ep/Accuracies', {'Val Acc': val_stats["accuracy"]["top1"].avg}, epoch)

    writer_val.add_scalars('ep/Val_Accuracy', {"top1": val_stats["accuracy"]["top1"].avg,
                                               "top3": val_stats["accuracy"]["top3"].avg,
                                               "top5": val_stats["accuracy"]["top5"].avg}, epoch)

    writer_val.add_scalars('ep/Validation_Timings', {"Data Loading":  val_stats["time_data_loading"].avg,
                                                     "Cuda Transfer": val_stats["time_cuda_transfer"].avg,
                                                     "Forward Pass":  val_stats["time_forward"].avg,
                                                     "Total":         val_stats["time_all"].avg}, epoch)

    # writer_val.add_embedding(val_stats["reps"].numpy(), tag="ep/Validation Representations", global_step=epoch)
