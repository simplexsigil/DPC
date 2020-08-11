import time

import torch

import loss_functions
from utils import AverageMeter, calc_topk_accuracy, write_out_checkpoint, write_out_images


def training_loop(model, optimizer, criterion, train_loader, val_loader, writer_train, writer_val, args, cuda_device):
    iteration = args.start_iteration
    best_val_acc = args.best_val_acc
    best_val_loss = args.best_val_loss

    best_projection_sim = args.best_projection_sim
    best_train_loss = args.best_train_loss

    # Main loop
    for epoch in range(args.start_epoch, args.epochs):
        iteration, train_loss, projection_sim = train_skvid_swav(train_loader, model, optimizer, criterion,
                                                                 epoch, iteration, args, writer_train, cuda_device)

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


def train_skvid_swav(data_loader, model, optimizer, criterion, epoch, iteration, args, writer_train,
                     cuda_device):
    tr_stats = {"time_data_loading": AverageMeter(locality=args.print_freq),
                "time_cuda_transfer": AverageMeter(locality=args.print_freq),
                "time_forward": AverageMeter(locality=args.print_freq),
                "time_scoring": AverageMeter(locality=args.print_freq),
                "time_backward": AverageMeter(locality=args.print_freq),
                "time_all": AverageMeter(locality=args.print_freq),

                "cv_rep_sim_m": AverageMeter(locality=args.print_freq),
                "cv_rep_sim_s": AverageMeter(locality=args.print_freq),
                "cv_rand_sim_m": AverageMeter(locality=args.print_freq),
                "cv_rand_sim_s": AverageMeter(locality=args.print_freq),

                "cv_proj_sim_m": AverageMeter(locality=args.print_freq),
                "cv_proj_sim_s": AverageMeter(locality=args.print_freq),
                "cv_proj_rnd_sim_m": AverageMeter(locality=args.print_freq),
                "cv_proj_rnd_sim_s": AverageMeter(locality=args.print_freq),

                "total_loss": AverageMeter(locality=args.print_freq),
                }

    model.train()

    tic = time.perf_counter()
    dl_time = time.perf_counter()
    all_time = time.perf_counter()

    for idx, out in enumerate(data_loader):
        bat_idxs, vid_seq, sk_seq = out

        batch_size = vid_seq.size(0)

        tr_stats["time_data_loading"].update(time.perf_counter() - dl_time)

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

        e_forw_time = time.perf_counter()
        tr_stats["time_forward"].update(e_forw_time - s_forw_time)

        # Scoring
        s_score_time = time.perf_counter()

        # ============ swav loss ... ============
        loss = 0
        with torch.no_grad():
            # get assignments
            q_vid = torch.exp(repr_vid_proj / args.epsilon).t()
            q_vid = sinkhorn(q_vid, args.sinkhorn_iterations)

            q_sk = torch.exp(repr_sk_proj / args.epsilon).t()
            q_sk = sinkhorn(q_sk, args.sinkhorn_iterations)

        p_vid = torch.nn.functional.softmax(repr_vid_proj / args.temperature)
        p_sk = torch.nn.functional.softmax(repr_sk_proj / args.temperature)

        loss = torch.zeros(1, device=repr_vid_proj.device)

        loss -= torch.mean(torch.sum(q_vid * torch.log(p_sk), dim=1))
        loss -= torch.mean(torch.sum(q_sk * torch.log(p_vid), dim=1))

        loss /= 2

        optimizer.zero_grad()

        e_score_time = time.perf_counter()
        tr_stats["time_scoring"].update(e_score_time - s_score_time)

        calculate_tr_stats(repr_vid, repr_sk, repr_vid_proj, repr_sk_proj, tr_stats)

        # Loss Calculation and backward pass.
        s_back_time = time.perf_counter()

        tr_stats["total_loss"].update(loss.item(), batch_size)

        loss.backward()

        # cancel some gradients
        if iteration < args.freeze_prototypes_niters:
            for name, p in model.named_parameters():
                if "prototypes" in name:
                    p.grad = None

        optimizer.step()

        e_back_time = time.perf_counter()
        tr_stats["time_backward"].update(e_back_time - s_back_time)

        tr_stats["time_all"].update(time.perf_counter() - all_time)

        if idx % args.print_freq == 0:
            e_tic = time.perf_counter()
            write_stats_batch_contrast_iteration(tr_stats, writer_train, iteration)
            print_tr_stats_loc_avg(tr_stats, epoch, idx, len(data_loader), e_tic - tic)

        iteration += 1

        tic = time.perf_counter()
        dl_time = time.perf_counter()
        all_time = time.perf_counter()
        # Next iteration

    write_stats_batch_contrast_epoch(tr_stats, writer_train, epoch)
    print_tr_stats_timings_avg(tr_stats)

    return iteration, tr_stats["total_loss"].avg, tr_stats["cv_proj_sim_m"].avg


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

    cv_proj_sim_m, cv_proj_sim_s = get_cv_sim_stats(rep_vid_proj, rep_sk_proj)

    tr_stats["cv_proj_sim_m"].update(cv_proj_sim_m.item(), batch_size)
    tr_stats["cv_proj_sim_s"].update(cv_proj_sim_s.item(), batch_size)

    rand_rep_sk_proj = rep_sk_proj.roll(shifts=1, dims=0)

    cv_proj_rnd_sim_m, cv_proj_rnd_sim_s = get_cv_sim_stats(rep_vid_proj, rand_rep_sk_proj)

    tr_stats["cv_proj_rnd_sim_m"].update(cv_proj_rnd_sim_m.item(), batch_size)
    tr_stats["cv_proj_rnd_sim_s"].update(cv_proj_rnd_sim_s.item(), batch_size)


def write_stats_batch_contrast_epoch(tr_stats, writer_train, epoch):
    writer_train.add_scalars('ep/Losses', {'Training Loss': tr_stats["total_loss"].avg}, epoch)

    writer_train.add_scalars('ep/Projection Similarities',
                             {'Projection Similarity (Mean)': tr_stats["cv_proj_sim_m"].avg,
                              'Projection Similarity (Std)': tr_stats["cv_proj_sim_s"].avg,
                              'Rand Projection Similarity (Mean)': tr_stats["cv_proj_rnd_sim_m"].avg,
                              'Rand Projection Similarity (Std)': tr_stats["cv_proj_rnd_sim_s"].avg,
                              }, epoch)


def write_stats_batch_contrast_iteration(tr_stats, writer_train, iteration):
    writer_train.add_scalars('it/Train_Loss', {'loss': tr_stats["total_loss"].local_avg}, iteration)

    rep_sim_dict = {'cv_tp_sim_mean': tr_stats["cv_rep_sim_m"].local_avg,
                    'cv_rand_sim_mean': tr_stats["cv_rand_sim_m"].local_avg,
                    'cv_tp_sim_std': tr_stats["cv_rep_sim_s"].local_avg,
                    'cv_rnd_sim_std': tr_stats["cv_rand_sim_s"].local_avg,
                    }

    writer_train.add_scalars('it/New_Representation_Similarities', rep_sim_dict, iteration)

    writer_train.add_scalars('it/Projection Similarities',
                             {'Projection Similarity (Mean)': tr_stats["cv_proj_sim_m"].local_avg,
                              'Projection Similarity (Std)': tr_stats["cv_proj_sim_s"].local_avg,
                              'Rand Projection Similarity (Mean)': tr_stats["cv_proj_rnd_sim_m"].local_avg,
                              'Rand Projection Similarity (Std)': tr_stats["cv_proj_rnd_sim_s"].local_avg,
                              }, iteration)

    all_calced_timings = sum([tr_stats[tms].local_avg for tms in ["time_data_loading",
                                                                  "time_cuda_transfer",
                                                                  "time_forward",
                                                                  "time_scoring",
                                                                  "time_backward",
                                                                  ]])
    timing_dict = {'Loading Data': tr_stats["time_data_loading"].local_avg,
                   'Cuda Transfer': tr_stats["time_cuda_transfer"].local_avg,
                   'Forward Pass': tr_stats["time_forward"].local_avg,
                   'Scoring': tr_stats["time_scoring"].local_avg,
                   'Backward Pass': tr_stats["time_backward"].local_avg,
                   'Loading + Transfer + '
                   'Forward + Scoring + Backward': all_calced_timings,
                   'All': tr_stats["time_all"].local_avg
                   }

    writer_train.add_scalars('it/Batch-Wise_Timings', timing_dict, iteration)


def print_tr_stats_loc_avg(stats: dict, epoch, idx, batch_count, duration):
    print(f'Epoch: [{epoch}][{idx}/{batch_count}]\t '
          f'Loss {stats["total_loss"].local_avg:.6f} '
          f'\tProjection Sim:  M {stats["cv_proj_sim_m"].local_avg:.4f} S {stats["cv_proj_sim_s"].local_avg:.4f} | '
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
    val_stats = {"time_data_loading": AverageMeter(locality=args.print_freq),
                 "time_cuda_transfer": AverageMeter(locality=args.print_freq),
                 "time_forward": AverageMeter(locality=args.print_freq),
                 "time_backward": AverageMeter(locality=args.print_freq),
                 "time_all": AverageMeter(locality=args.print_freq),

                 "total_loss": AverageMeter(locality=args.print_freq),

                 "accuracy": {"top1": AverageMeter(locality=args.print_freq),
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

            repr_vid, repr_sk = model(vid_seq, sk_seq)

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

    writer_val.add_scalars('ep/Validation_Timings', {"Data Loading": val_stats["time_data_loading"].avg,
                                                     "Cuda Transfer": val_stats["time_cuda_transfer"].avg,
                                                     "Forward Pass": val_stats["time_forward"].avg,
                                                     "Total": val_stats["time_all"].avg}, epoch)


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
        return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()
