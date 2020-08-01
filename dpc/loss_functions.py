import torch


def memory_contrast_scores(x: torch.Tensor,
                           y: torch.Tensor,
                           x_mem: torch.Tensor,
                           y_mem: torch.Tensor,
                           x_cont: torch.Tensor,
                           y_cont: torch.Tensor,
                           matching_fn: str,
                           use_current_contrast=True,
                           temp_tao=0.1,
                           contrast_type="cross") -> (torch.Tensor, torch.Tensor):
    if matching_fn == "cos-nt-xent":
        # Implement memory contrast
        # We always set the first vector to be the ground truth.

        results_x = []
        results_y = []

        batch_size = x.shape[0]

        if use_current_contrast:
            if x_cont.shape[0] > 0:
                # To avoid batch contrast
                x_calc = x_cont.unsqueeze(dim=0).repeat(batch_size, 1, 1)
                y_calc = y_cont.unsqueeze(dim=0).repeat(batch_size, 1, 1)

                x_tp = x_mem.view((batch_size, 1, -1))
                y_tp = y_mem.view((batch_size, 1, -1))

                x_calc = torch.cat((x_tp, x_calc), dim=1)
                y_calc = torch.cat((y_tp, y_calc), dim=1)
            else:
                x_calc = x.unsqueeze(dim=0).repeat(batch_size, 1, 1)
                y_calc = y.unsqueeze(dim=0).repeat(batch_size, 1, 1)
        else:
            x_calc = torch.cat((x_mem, x_cont))
            y_calc = torch.cat((y_mem, y_cont))

        for i in range(batch_size):
            # The scores are calculated between the output of one modality and the output
            # of the other modality. The first vectors are the ground truth (other modality).
            if contrast_type == "cross":
                scores_x_i = pairwise_scores(x[i].view((1, -1)), y_calc[i], matching_fn=matching_fn)
                scores_y_i = pairwise_scores(y[i].view((1, -1)), x_calc[i], matching_fn=matching_fn)
            elif contrast_type == "self":
                scores_x_i = pairwise_scores(x[i].reshape((1, -1)), x_mem, matching_fn=matching_fn)
                scores_y_i = pairwise_scores(y[i].reshape((1, -1)), y_mem, matching_fn=matching_fn)
            else:
                raise ValueError

            results_x.append(scores_x_i)
            results_y.append(scores_y_i)

        results_x = torch.cat(results_x, dim=0)
        results_y = torch.cat(results_y, dim=0)

        return results_x, results_y
    else:
        raise ValueError


def pairwise_scores(x: torch.Tensor,
                    y: torch.Tensor,
                    matching_fn: str,
                    temp_tao=1.) -> torch.Tensor:
    """Efficiently calculate pairwise distances (or other similarity scores) between
    two sets of samples.
    # Arguments
        x: Query samples. A tensor of shape (n_x, d) where d is the embedding dimension
        y: Class prototypes. A tensor of shape (n_y, d) where d is the embedding dimension
        matching_fn: Distance metric/similarity score to compute between samples
        temp_tao: A temperature parameter as used for example with NT-Xent loss (Normalized Temp. Scaled Cross Ent.)
    """
    n_x = x.shape[0]
    n_y = y.shape[0]

    eps = 1e-7

    if matching_fn == 'l2':
        distances = (
                x.unsqueeze(1).expand(n_x, n_y, -1) -
                y.unsqueeze(0).expand(n_x, n_y, -1)
        ).pow(2).sum(dim=2)
        return distances
    elif matching_fn == 'cosine':
        normalised_x = x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + eps)
        normalised_y = y / (y.pow(2).sum(dim=1, keepdim=True).sqrt() + eps)

        expanded_x = normalised_x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = normalised_y.unsqueeze(0).expand(n_x, n_y, -1)

        cosine_similarities = (expanded_x * expanded_y).sum(dim=2)
        return 1 - cosine_similarities
    elif matching_fn == 'dot':
        return torch.matmul(x, y.transpose(0, 1))

    elif matching_fn == "cos-nt-xent":
        x_norm = x / torch.norm(x, dim=1, keepdim=True)
        y_norm = y / torch.norm(y, dim=1, keepdim=True)

        xy_n = torch.matmul(x_norm, y_norm.transpose(0, 1))
        xy_nt = xy_n / temp_tao

        return xy_nt

    elif matching_fn == "orth-nt-xent":
        x_norm = x / torch.norm(x, dim=1, keepdim=True)
        y_norm = y / torch.norm(y, dim=1, keepdim=True)
        xy_n = torch.matmul(x_norm, y_norm.transpose(0, 1))
        xy_nt = torch.acos(xy_n) / temp_tao

        return xy_nt

    elif matching_fn == "euc-nt-xent":
        x_norm = x / torch.norm(x, dim=1, keepdim=True)
        y_norm = y / torch.norm(y, dim=1, keepdim=True)

        x_sq = torch.sum((x_norm * x_norm), dim=1, keepdim=True)
        y_sq = torch.sum((y_norm * y_norm), dim=1, keepdim=True)

        y_sq = y_sq.transpose(0, 1)

        score = torch.matmul(x_norm, y_norm.transpose(0, 1))
        dst = torch.nn.functional.relu(x_sq - 2 * score + y_sq)

        '''
        eps_t = torch.full(d.shape, 1e-12)
        is_zero = d.abs().lt(1e-12)

        dst = torch.where(is_zero, eps_t, d)
        '''

        dst = torch.sqrt(dst)

        return -dst / temp_tao

    else:
        raise (ValueError('Unsupported similarity function'))
