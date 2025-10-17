import torch


def score_entropy_loss_fn(graph, sampling_eps, model, batch):
    batch_size = batch.shape[0]
    device = batch.device

    # Sample random timesteps
    t = (1 - sampling_eps) * torch.rand(batch_size, device=device) + sampling_eps
    sigma = t.unsqueeze(-1)  # Add dimension for broadcasting

    # Sample noisy version
    x_noisy = graph.sample_transition(batch, sigma)

    # Get model prediction
    log_scores = model(x_noisy, sigma)

    # Compute score entropy loss
    loss = graph.score_entropy(log_scores, sigma, x_noisy, batch)

    # Take mean over spatial dimensions and batch
    return loss.mean()


def t_dce_loss_fn(graph, sampling_eps, model, batch):
    """t-DCE loss for time-independent models"""
    batch_size = batch.shape[0]
    device = batch.device

    # Sample random timesteps
    t = (1 - sampling_eps) * torch.rand(batch_size, device=device) + sampling_eps
    sigma = t.unsqueeze(-1)

    # Sample noisy version using absorbing diffusion
    x_noisy = graph.sample_transition(batch, sigma)

    # Get model prediction (conditional probabilities)
    log_probs = model(x_noisy)  # No time conditioning

    # Convert values to indices for gathering
    x_indices = graph.value_to_index(batch)

    # Mask for positions that were corrupted (different from clean)
    corrupted_mask = (x_noisy != batch)

    # Extract log probabilities for clean values at corrupted positions
    batch_indices = torch.arange(batch_size, device=device).unsqueeze(-1).unsqueeze(-1)
    height_indices = torch.arange(batch.shape[1], device=device).unsqueeze(0).unsqueeze(-1)
    width_indices = torch.arange(batch.shape[2], device=device).unsqueeze(0).unsqueeze(0)

    # Gather log probabilities for true values
    neg_log_probs = -torch.gather(
        log_probs[batch_indices, height_indices, width_indices],
        -1,
        x_indices.unsqueeze(-1)
    ).squeeze(-1)

    # Apply mask and compute loss only for corrupted positions
    loss = torch.where(corrupted_mask, neg_log_probs, torch.zeros_like(neg_log_probs))

    # Weight by noise level and take mean
    sigma_flat = sigma.squeeze(-1)
    esigma_minus_1 = torch.expm1(sigma_flat)
    weighted_loss = loss * esigma_minus_1.unsqueeze(-1).unsqueeze(-1)

    return weighted_loss.sum() / corrupted_mask.sum().clamp(min=1)

def lambda_dce_loss_fn(graph, sampling_eps, model, batch):
    """λ-DCE loss for time-independent models"""
    batch_size = batch.shape[0]
    device = batch.device

    # Sample λ uniformly from [0, 1]
    lambda_vals = torch.rand(batch_size, device=device)

    # Create corrupted version by flipping each element with probability λ
    flip_probs = lambda_vals.unsqueeze(-1).unsqueeze(-1)
    should_flip = torch.rand_like(batch.float()) < flip_probs
    x_corrupted = torch.where(should_flip, -batch, batch)

    # Get model prediction
    log_probs = model(x_corrupted)

    # Convert values to indices
    x_indices = graph.value_to_index(batch)

    # Extract log probabilities for clean values
    batch_indices = torch.arange(batch_size, device=device).unsqueeze(-1).unsqueeze(-1)
    height_indices = torch.arange(batch.shape[1], device=device).unsqueeze(0).unsqueeze(-1)
    width_indices = torch.arange(batch.shape[2], device=device).unsqueeze(0).unsqueeze(0)

    log_probs_clean = torch.gather(
        log_probs[batch_indices, height_indices, width_indices],
        -1,
        x_indices.unsqueeze(-1)
    ).squeeze(-1)

    # Compute λ-DCE loss: -E[log p(x|x_corrupted)] / λ
    loss_per_sample = -log_probs_clean.sum(dim=(-2, -1)) / lambda_vals.clamp(min=sampling_eps)

    return loss_per_sample.mean()



