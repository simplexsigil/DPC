import torch

from utils import AverageMeter, calc_topk_accuracy


def train_skvid_batch_contrast(data_loader, model, optimizer, epoch, args=None):
    global iteration

    tr_stats = {"time_data_loading":  AverageMeter(locality=args.print_freq),
                "time_cuda_transfer": AverageMeter(locality=args.print_freq),
                "time_forward":       AverageMeter(locality=args.print_freq),
                "time_backward":      AverageMeter(locality=args.print_freq),
                "loss":               AverageMeter(locality=args.print_freq),
                "cross_view_sim":     AverageMeter(locality=args.print_freq),
                "accuracy":           {"top1": AverageMeter(locality=args.print_freq),
                                       "top3": AverageMeter(locality=args.print_freq),
                                       "top5": AverageMeter(locality=args.print_freq)},

                }

    model.train()

    tic = time.perf_counter()
    start_time = time.perf_counter()

    for idx, out in enumerate(data_loader):
        bat_idxs, vid_seq, sk_seq = out

        batch_size = vid_seq.size(0)

        tr_stats["time_data_loading"].update(time.perf_counter() - start_time)

        # Visualize images for tensorboard on two iterations.
        if (iteration == 0) or (iteration == args.print_freq):
            write_out_images(vid_seq, writer_train)

        # Cuda Transfer
        start_time = time.perf_counter()

        vid_seq = vid_seq.to(cuda_device)
        sk_seq = sk_seq.to(cuda_device)

        tr_stats["time_cuda_transfer"].update(time.perf_counter() - start_time)

        # Forward pass: Calculation
        start_time = time.perf_counter()

        score, targets, repr_vid, repr_sk = model(vid_seq, sk_seq, None, None, None, None)

        targets = targets.detach()

        tr_stats["time_forward"].update(time.perf_counter() - start_time)

        # Average distance of representations
        cross_view_sim = torch.mean(torch.sum(repr_vid * repr_sk, dim=1), dim=0)

        tr_stats["cross_view_sim"].update(cross_view_sim.item(), batch_size)

        # Calculate Accuracies
        top1, top3, top5 = calc_topk_accuracy(score, targets, (1, 3, 5))

        tr_stats["accuracy"]["top1"].update(top1.item(), batch_size)
        tr_stats["accuracy"]["top3"].update(top3.item(), batch_size)
        tr_stats["accuracy"]["top5"].update(top5.item(), batch_size)

        # Loss Calculation and backward pass.
        start_time = time.perf_counter()

        loss = criterion(score, targets)

        tr_stats["loss"].update(loss.item(), batch_size)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        tr_stats["time_backward"].update(time.perf_counter() - start_time)

        if idx % args.print_freq == 0:
            print_tr_stats_iteration_batch_contrast(tr_stats, epoch, idx, len(data_loader), time.perf_counter() - tic)

        iteration += 1
        write_stats_batch_contrast_iteration(tr_stats, writer_train, iteration)

        start_time = time.perf_counter()
        tic = time.perf_counter()
        # Next iteration

    write_stats_batch_contrast_epoch(tr_stats, writer_train, epoch)
    print_stats_timings_batch_contrast(tr_stats)

    accuracies = [tr_stats["accuracy"]["top1"].val, tr_stats["accuracy"]["top3"].val, tr_stats["accuracy"]["top5"].val]

    return tr_stats["loss"].val, accuracies[0], accuracies


def write_stats_batch_contrast_epoch(tr_stats, writer_train, epoch):
    # save curve
    writer_train.add_scalar('training_loss', tr_stats["loss"].val, epoch)

    writer_train.add_scalars('training_accuracies', {'top1': tr_stats["accuracy"]["top1"].val,
                                                     'top3': tr_stats["accuracy"]["top3"].val,
                                                     'top5': tr_stats["accuracy"]["top5"].val}, epoch)

    writer_train.add_scalar('representation_similarity', tr_stats["cross_view_sim"].val, epoch)


def write_stats_batch_contrast_iteration(tr_stats, writer_train, iteration):
    pass
    # writer_train.add_scalar('iterations/loss_rgb', tr_stats["total_loss"].local_avg, iteration)
    # writer_train.add_scalar('iterations/loss_sk', tr_stats["total_loss"].local_avg, iteration)
    # writer_train.add_scalar('iterations/total_loss', tr_stats["total_loss"].local_avg, iteration)
    # writer_train.add_scalar('iterations/accuracy_sk', tr_stats["accuracy_sk"]["top1"].local_avg, iteration)
    # writer_train.add_scalar('iterations/accuracy_rgb', tr_stats["accuracy_rgb"]["top1"].local_avg, iteration)
    # writer_train.add_scalar('iterations/sk_prox_reg_sim', tr_stats["sk_prox_reg_sim"].local_avg, iteration)
    # writer_train.add_scalar('iterations/vid_prox_reg_sim', tr_stats["vid_prox_reg_sim"].local_avg, iteration)
    # writer_train.add_scalar('iterations/cross_view_dist', tr_stats["cross_view_sim"].local_avg, iteration)


def print_stats_timings_batch_contrast(tr_stats):
    print(
        f'Data Loading: {tr_stats["time_data_loading"].val:.4f}s | '
        f'Cuda Transfer: {tr_stats["time_cuda_transfer"].val:.4f}s | '
        f'Forward: {tr_stats["time_forward"].val:.4f}s | '
        f'Backward: {tr_stats["time_backward"].val:.4f}s')


def print_tr_stats_iteration_batch_contrast(stats: dict, epoch, idx, batch_count, duration):
    print(f'Epoch: [{epoch}][{idx}/{batch_count}]\t '
          f'Loss {stats["loss"].local_avg:.6f} '
          '\tAcc: '
          f'top1 {stats["accuracy"]["top1"].local_avg:.4f} | '
          f'top3 {stats["accuracy"]["top3"].local_avg:.4f} | '
          f'top5 {stats["accuracy"]["top5"].local_avg:.4f} '
          f'Sim: {stats["cross_view_sim"].local_avg:.4f} '
          f'T:{duration:.2f}\n'
          )


def validate_batch_contrast(data_loader, model, criterion, cuda_device, epoch, args):
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
