import torch


def memory_contrast_scores(x: torch.Tensor,
                           y: torch.Tensor,
                           x_mem: torch.Tensor,
                           y_mem: torch.Tensor,
                           x_cont: torch.Tensor,
                           y_cont: torch.Tensor,
                           matching_fn: str,
                           use_current_tp=False,
                           temp_tao=0.1,
                           contrast_type="cross") -> (torch.Tensor, torch.Tensor):
    assert x_cont.shape[0] > 0
    assert not (use_current_tp and contrast_type == "self"), "Self contrast can only be used when with memories."
    assert matching_fn in ["cos-nt-xent"], f"Score function {matching_fn} not yet supported with mem contrast."

    if matching_fn == "cos-nt-xent":
        # Implement memory contrast
        # We always set the first vector to be the ground truth.

        score_x = []
        score_y = []

        batch_size = x.shape[0]

        # To avoid batch contrast - We do not want to contrast with other reps.
        # Each rep has its own copy of memories. Other solutions would include masking or selection processes.
        x_calc = x_cont.unsqueeze(dim=0).repeat(batch_size, 1, 1)
        y_calc = y_cont.unsqueeze(dim=0).repeat(batch_size, 1, 1)
        # Shape (batch_size, contrast_size, rep_size)

        if use_current_tp:  # A positive is the current rep of the other view.
            x_tp = x.view((batch_size, 1, -1))
            y_tp = y.view((batch_size, 1, -1))
        else:  # A positive is the memory of the other view.
            x_tp = x_mem.view((batch_size, 1, -1))
            y_tp = y_mem.view((batch_size, 1, -1))

        x_calc = torch.cat((x_tp, x_calc), dim=1)
        y_calc = torch.cat((y_tp, y_calc), dim=1)

        for i in range(batch_size):
            # The scores are calculated between the output of one modality and the output
            # of the other modality. The first vectors are the ground truth (other modality).
            if contrast_type == "cross":
                scores_x_i = pairwise_scores(x[i].view((1, -1)), y_calc[i], matching_fn=matching_fn)
                scores_y_i = pairwise_scores(y[i].view((1, -1)), x_calc[i], matching_fn=matching_fn)
            elif contrast_type == "self":
                scores_x_i = pairwise_scores(x[i].view((1, -1)), x_calc[i], matching_fn=matching_fn)
                scores_y_i = pairwise_scores(y[i].view((1, -1)), y_calc[i], matching_fn=matching_fn)
            else:
                raise ValueError

            score_x.append(scores_x_i)
            score_y.append(scores_y_i)

        score_x = torch.cat(score_x, dim=0)
        score_y = torch.cat(score_y, dim=0)

        target_x = torch.tensor([0] * batch_size, dtype=torch.long, device=x.device)
        target_y = torch.tensor([0] * batch_size, dtype=torch.long, device=x.device)
    else:
        raise ValueError

    return score_x, score_y, target_x, target_y


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

        dst = torch.sqrt(dst)

        return -dst / temp_tao

    else:
        raise (ValueError('Unsupported similarity function'))
