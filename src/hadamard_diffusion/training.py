from functools import partial
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import wandb

from hadamard_diffusion.dataloader import PreShuffledHadamardDataset, create_train_eval_datasets
from hadamard_diffusion.losses import score_entropy_loss_fn, t_dce_loss_fn, lambda_dce_loss_fn
from hadamard_diffusion.score_model import HadamardScoreModel, HadamardRADDModel
from hadamard_diffusion.boolean_cube import BinaryUniformGraph, BinaryAbsorbingGraph
from hadamard_diffusion.sampling import get_binary_sampler, validate_hadamard_properties


def reparameterized_schedule(t, t_critical=0.013, concentration=10.0):
    """
    Reparameterize time to concentrate samples around critical temperature

    Args:
        uniform_t: Uniform time points in [0, 1]
        t_critical: Critical temperature from phase transition analysis
        concentration: Higher values = more concentration around t_critical
    """
    # Map [0,1] to [-1,1] for logit
    

    # Apply logit-like transform and scale
    logit_t = np.log(((1 + t) / (1 - t)) - 1) / concentration

    # Shift to center around t_critical and map back to [0,1]
    reparameterized_t = logit_t + t_critical

    # Ensure bounds and normalize
    return np.clip(reparameterized_t, 0, 1)



def get_loss_fn(graph, sampling_eps=1e-3, loss_type='score_entropy'):
    """Loss function for discrete diffusion training

    Args:
        graph: BinaryUniformGraph instance
        sampling_eps: Small epsilon for numerical stability
        loss_type: 'score_entropy', 't_dce', or 'lambda_dce'
    """

    if loss_type == 'score_entropy':
        return partial(score_entropy_loss_fn, graph, sampling_eps)
    elif loss_type == 't_dce':
        return partial(t_dce_loss_fn, graph, sampling_eps)
    elif loss_type == 'lambda_dce':
        return partial(lambda_dce_loss_fn, graph, sampling_eps)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def hadamard_energy(matrices):
    """Compute Hadamard energy: ||M @ M.T - n*I||_F

    Args:
        matrices: (batch, n, n) tensor

    Returns:
        energies: (batch,) tensor of Hadamard energies
    """
    # Convert to float for computation
    matrices = matrices.float()
    n = matrices.shape[-1]
    MMT = matrices @ matrices.transpose(-2, -1)
    identity = torch.eye(n, device=matrices.device, dtype=matrices.dtype) * n
    energy = torch.norm(MMT - identity, dim=(-2, -1))
    return energy


def track_denoising_hadamard_energy(model, graph, matrix_size, device, steps=50, num_samples=4, eps=1e-5):
    """Track Hadamard energy during the denoising process

    Args:
        model: Trained model
        graph: Graph instance
        matrix_size: Size of matrices
        device: Device to run on
        steps: Number of denoising steps
        num_samples: Number of samples to track
        eps: Small epsilon for numerical stability

    Returns:
        dict with denoising trajectory analysis
    """
    model.eval()

    with torch.no_grad():
        # Start from the limiting distribution (pure noise)
        x = graph.sample_limit(num_samples, matrix_size, matrix_size).to(device)

        # Create timestep schedule (reverse: 1.0 -> eps)
        timesteps = torch.linspace(1.0, eps, steps + 1, device=device)
        dt = (1.0 - eps) / steps

        # Track Hadamard energy and expected clean data at each step
        trajectory = []

        for i in range(steps + 1):
            t = timesteps[i]

            # Compute Hadamard energy of current state
            current_energy = hadamard_energy(x).mean().item()

            # Get expected clean data based on model type
            try:
                # RADD model - can compute expected values directly
                log_probs = model(x)
                probs = torch.exp(log_probs)

                # Expected clean data (convert probabilities to expected values)
                expected_clean = torch.zeros_like(x)
                expected_clean += probs[..., 0] * (-1)  # P(-1) * (-1)
                expected_clean += probs[..., 1] * 1     # P(+1) * (+1)

                expected_clean_energy = hadamard_energy(expected_clean).mean().item()
            except (TypeError, RuntimeError):
                # Score model - would need additional implementation to get expected clean data
                expected_clean_energy = None

            trajectory.append({
                'step': i,
                'timestep': t.item(),
                'current_hadamard_energy': current_energy,
                'expected_clean_hadamard_energy': expected_clean_energy
            })

            # Denoising step (simplified)
            if i < steps:
                try:
                    # Time-independent model (RADD)
                    log_probs = model(x)
                    # Convert to binary values using argmax for simplicity
                    x = graph.index_to_value(torch.argmax(log_probs, dim=-1))
                except TypeError:
                    # Time-dependent model - would need proper denoising implementation
                    # For now, just keep current state
                    pass

        return {
            'trajectory': trajectory,
            'final_hadamard_energy': trajectory[-1]['current_hadamard_energy'],
            'initial_hadamard_energy': trajectory[0]['current_hadamard_energy']
        }


def compute_elbo(model, x_clean, graph, sampling_eps=1e-3, num_samples=50, return_timestep_info=False):
    """
    Compute ELBO (Evidence Lower BOund) for held-out data

    Args:
        model: Trained score model
        x_clean: Clean Hadamard matrices (batch, height, width)
        graph: BinaryUniformGraph instance
        sampling_eps: Small epsilon for numerical stability
        num_samples: Number of timesteps to sample for Monte Carlo estimate
        return_timestep_info: If True, return detailed timestep analysis

    Returns:
        If return_timestep_info=False: ELBO estimate (scalar)
        If return_timestep_info=True: dict with ELBO and timestep analysis
    """
    model.eval()

    with torch.no_grad():
        batch_size = x_clean.shape[0]
        device = x_clean.device

        # Sample timesteps for Monte Carlo integration
        t = (1 - sampling_eps) * torch.rand(num_samples, device=device) + sampling_eps

        elbo_estimates = []
        timestep_info = []

        for i in range(num_samples):
            # Expand timestep to batch
            t_batch = t[i].expand(batch_size).unsqueeze(-1)

            # Forward diffusion: x_0 -> x_t
            x_noisy = graph.sample_transition(x_clean, t_batch)

            # Get model scores
            try:
                # Try time-independent first (RADD model)
                log_scores = model(x_noisy)
            except TypeError:
                # Time-dependent model (score model) needs timestep
                log_scores = model(x_noisy, t_batch)

            # Compute score entropy (the integrand of L_DWDSE)
            if hasattr(graph, 'score_entropy'):
                score_entropy = graph.score_entropy(log_scores, t_batch, x_noisy, x_clean)
            else:
                # For RADD models, compute negative log likelihood
                x_indices = graph.value_to_index(x_clean)
                batch_indices = torch.arange(batch_size, device=device).unsqueeze(-1).unsqueeze(-1)
                height_indices = torch.arange(x_clean.shape[1], device=device).unsqueeze(0).unsqueeze(-1)
                width_indices = torch.arange(x_clean.shape[2], device=device).unsqueeze(0).unsqueeze(0)

                score_entropy = -torch.gather(
                    log_scores[batch_indices, height_indices, width_indices],
                    -1,
                    x_indices.unsqueeze(-1)
                ).squeeze(-1).sum(dim=(-2, -1))

            # Store timestep information if requested
            if return_timestep_info:
                timestep_info.append({
                    't': t[i].item(),
                    'score_entropy_mean': score_entropy.mean().item(),
                    'score_entropy_std': score_entropy.std().item(),
                    'hadamard_energy_clean': hadamard_energy(x_clean).mean().item(),
                    'hadamard_energy_noisy': hadamard_energy(x_noisy).mean().item()
                })

            # Add to estimates
            elbo_estimates.append(score_entropy.mean())

        # Monte Carlo estimate of integral over time
        elbo_estimate = torch.stack(elbo_estimates).mean() * (1 - sampling_eps)

        # KL divergence to base distribution (usually small for uniform)
        # For uniform base distribution, this is approximately 0
        kl_divergence = 0.0

        total_elbo = elbo_estimate.item() + kl_divergence

        if return_timestep_info:
            return {
                'elbo': total_elbo,
                'timestep_analysis': timestep_info
            }
        else:
            return total_elbo


def evaluate_model(model, eval_data, graph, device, matrix_size=32, num_eval_samples=4,
                   include_random_comparison=True, detailed_analysis=True):
    """
    Comprehensive evaluation: ELBO + sampling + Hadamard validation

    Args:
        model: Trained score model
        eval_data: Held-out evaluation data (small batch)
        graph: BinaryUniformGraph instance
        device: Device to run on
        matrix_size: Size of matrices
        num_eval_samples: Number of samples to generate

    Returns:
        dict with evaluation metrics
    """
    model.eval()

    # 1. Compute ELBO on held-out data (valid Hadamard matrices)
    if detailed_analysis:
        elbo_result = compute_elbo(model, eval_data, graph, return_timestep_info=True)
        elbo_hadamard = elbo_result['elbo']
        timestep_analysis = elbo_result['timestep_analysis']
    else:
        elbo_hadamard = compute_elbo(model, eval_data, graph)
        timestep_analysis = None

    # 2. Optional: Generate random matrices for comparison
    elbo_random = None
    if include_random_comparison:
        batch_size = eval_data.shape[0]
        random_matrices = graph.sample_data_like(batch_size, matrix_size, matrix_size).to(device)

        # Make them more "random" by flipping some elements to break any accidental structure
        flip_mask = torch.rand_like(random_matrices.float()) < 0.3  # Flip 30% of elements
        random_matrices = torch.where(flip_mask, -random_matrices, random_matrices)

        elbo_random = compute_elbo(model, random_matrices, graph)

    # 3. Generate samples and validate Hadamard properties
    sampler = get_binary_sampler(
        model=model,
        graph=graph,
        matrix_size=matrix_size,
        steps=50,  # Moderate number of steps for evaluation
        predictor="euler",
        device=device,
        show_progress=False
    )

    # Generate samples
    with torch.no_grad():
        generated_samples = sampler(num_eval_samples)

    # Validate Hadamard properties
    validation_results = validate_hadamard_properties(generated_samples, tolerance=1.0)

    # 4. Track denoising Hadamard energy evolution
    denoising_analysis = None
    if detailed_analysis:
        denoising_analysis = track_denoising_hadamard_energy(
            model, graph, matrix_size, device, steps=20, num_samples=2
        )

    # 5. Compute additional metrics

    # Check value distribution (should be close to {-1, +1})
    unique_vals = torch.unique(generated_samples)
    value_range = (generated_samples.min().item(), generated_samples.max().item())

    # Orthogonality errors and Hadamard energies for analysis
    orthogonal_errors = []
    hadamard_energies = []
    for i in range(num_eval_samples):
        H = generated_samples[i]
        HHT = H @ H.T
        expected = matrix_size * torch.eye(matrix_size, device=H.device)
        error = torch.norm(HHT - expected).item()
        orthogonal_errors.append(error)
        hadamard_energies.append(hadamard_energy(H.unsqueeze(0)).item())

    # Build result dictionary
    result = {
        'elbo_hadamard': elbo_hadamard,
        'binary_rate': validation_results['binary_rate'],  # Always 100% due to discrete sampling
        'orthogonal_rate': validation_results['orthogonal_rate'],
        'valid_hadamard_rate': validation_results['valid_rate'],
        'mean_orthogonal_error': np.mean(orthogonal_errors),
        'std_orthogonal_error': np.std(orthogonal_errors),
        'mean_hadamard_energy': np.mean(hadamard_energies),
        'std_hadamard_energy': np.std(hadamard_energies),
        'value_range': value_range,
        'unique_values': unique_vals.cpu().tolist(),
        'num_unique_values': len(unique_vals)
    }

    # Add optional metrics
    if include_random_comparison and elbo_random is not None:
        result.update({
            'elbo_random': elbo_random,
            'elbo_difference': elbo_hadamard - elbo_random
        })

    if detailed_analysis:
        result.update({
            'timestep_analysis': timestep_analysis,
            'denoising_analysis': denoising_analysis
        })

    return result


class EMA:
    """Exponential moving average for model parameters"""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def train_hadamard_diffusion(
    data_source,
    num_epochs=100,
    batch_size=32,
    lr=1e-4,
    max_matrices=10000,
    matrix_size=32,
    model_kwargs=None,
    device=None,
    save_dir="checkpoints",
    log_interval=100,
    eval_interval=5,
    eval_batch_size=8,
    model_type='score',  # 'score' or 'radd'
    loss_type=None,  # Auto-select based on model_type if None
    graph_type='uniform',  # 'uniform' or 'absorbing'
    use_wandb=True,
    wandb_project="hadamard-diffusion",
    wandb_run_name=None,
    s3_bucket=None  # S3 bucket name (if None, assumes local loading)
):
    """
    Train Hadamard diffusion model

    Args:
        data_source: Local directory or S3 prefix containing hadamard_matrices_batch_*.npy files
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        max_matrices: Maximum number of matrices to load (None for all)
        matrix_size: Size of Hadamard matrices
        model_kwargs: Additional arguments for HadamardScoreModel/HadamardRADDModel
        device: Device to train on
        save_dir: Directory to save checkpoints
        log_interval: Steps between logging
        eval_interval: Epochs between evaluation
        eval_batch_size: Batch size for evaluation
        model_type: 'score' (time-dependent) or 'radd' (time-independent)
        loss_type: 'score_entropy', 't_dce', or 'lambda_dce' (auto-selected if None)
        graph_type: 'uniform' (uniform diffusion) or 'absorbing' (absorbing diffusion)
        use_wandb: Whether to use Weights & Biases logging
        wandb_project: W&B project name
        wandb_run_name: W&B run name (auto-generated if None)
        s3_bucket: S3 bucket name (if None, assumes local loading)
    """

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Training on device: {device}")

    # Create directories
    Path(save_dir).mkdir(exist_ok=True)

    # Initialize dataset and split into train/eval
    dataset = HadamardDataset(data_source, max_matrices=max_matrices, s3_bucket=s3_bucket)

    # Create train/eval split (90/10)
    train_size = int(0.9 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])

    # Create dataloaders
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Get a fixed evaluation batch for consistent monitoring
    eval_batch = next(iter(eval_dataloader)).to(device)

    # Initialize model and graph
    model_kwargs = model_kwargs or {}
    if model_type == 'score':
        model = HadamardScoreModel(matrix_size=matrix_size, **model_kwargs).to(device)
        default_loss_type = 'score_entropy'
    elif model_type == 'radd':
        model = HadamardRADDModel(matrix_size=matrix_size, **model_kwargs).to(device)
        default_loss_type = 't_dce'
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Initialize graph
    if graph_type == 'uniform':
        graph = BinaryUniformGraph()
    elif graph_type == 'absorbing':
        graph = BinaryAbsorbingGraph()
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")

    # Initialize optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = get_loss_fn(graph, loss_type=loss_type or default_loss_type)
    scaler = torch.cuda.amp.GradScaler()
    ema = EMA(model)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(dataloader))

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training batches per epoch: {len(dataloader)}")

    # Initialize wandb logging
    if use_wandb:
        config = {
            'model_type': model_type,
            'loss_type': loss_type or default_loss_type,
            'graph_type': graph_type,
            'matrix_size': matrix_size,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': lr,
            'max_matrices': max_matrices,
            'model_parameters': sum(p.numel() for p in model.parameters()),
            'device': str(device),
            **model_kwargs
        }

        run_name = wandb_run_name or f"{model_type}_{loss_type or default_loss_type}_{matrix_size}x{matrix_size}"
        wandb.init(
            project=wandb_project,
            name=run_name,
            config=config,
            tags=[model_type, loss_type or default_loss_type, f"size_{matrix_size}"]
        )
        wandb.watch(model, log="all", log_freq=log_interval)

    # Training loop
    model.train()
    global_step = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in progress_bar:
            batch = batch.to(device, non_blocking=True)

            optimizer.zero_grad()

            # Forward pass with mixed precision
            with torch.amp.autocast():
                loss = loss_fn(model, batch)

            # Backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Update EMA
            ema.update(model)

            # Logging
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            if global_step % log_interval == 0:
                avg_loss = epoch_loss / num_batches
                lr = scheduler.get_last_lr()[0]
                progress_bar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{lr:.2e}',
                    'step': global_step
                })

                # Log to wandb
                if use_wandb:
                    wandb.log({
                        'train/loss': avg_loss,
                        'train/learning_rate': lr,
                        'train/epoch': epoch + 1,
                        'train/step': global_step
                    }, step=global_step)

        # Evaluation
        if (epoch + 1) % eval_interval == 0:
            print(f"\nRunning evaluation after epoch {epoch + 1}...")

            # Apply EMA weights for evaluation
            ema.apply_shadow(model)

            try:
                eval_metrics = evaluate_model(
                    model=model,
                    eval_data=eval_batch,
                    graph=graph,
                    device=device,
                    matrix_size=matrix_size,
                    num_eval_samples=4,
                    include_random_comparison=True,  # Keep for now as sanity check
                    detailed_analysis=(epoch + 1) % (eval_interval * 2) == 0  # Detailed analysis every other eval
                )

                print(f"Evaluation Results:")
                print(f"  ELBO (Hadamard): {eval_metrics['elbo_hadamard']:.4f}")

                # Print random comparison if available
                if 'elbo_random' in eval_metrics:
                    print(f"  ELBO (Random): {eval_metrics['elbo_random']:.4f}")
                    print(f"  ELBO Difference: {eval_metrics['elbo_difference']:.4f} (should be positive)")

                print(f"  Orthogonal rate: {eval_metrics['orthogonal_rate']:.2%}")
                print(f"  Mean orthogonal error: {eval_metrics['mean_orthogonal_error']:.4f}")
                print(f"  Mean Hadamard energy: {eval_metrics['mean_hadamard_energy']:.4f}")
                print(f"  Value range: {eval_metrics['value_range']}")
                print(f"  Unique values: {len(eval_metrics['unique_values'])} ({eval_metrics['unique_values']})")

                # Print detailed analysis if available
                if 'denoising_analysis' in eval_metrics and eval_metrics['denoising_analysis']:
                    denoising = eval_metrics['denoising_analysis']
                    print(f"  Denoising: {denoising['initial_hadamard_energy']:.2f} → {denoising['final_hadamard_energy']:.2f}")

                # Log evaluation metrics to wandb
                if use_wandb:
                    wandb_metrics = {
                        'eval/elbo_hadamard': eval_metrics['elbo_hadamard'],
                        'eval/orthogonal_rate': eval_metrics['orthogonal_rate'],
                        'eval/mean_orthogonal_error': eval_metrics['mean_orthogonal_error'],
                        'eval/mean_hadamard_energy': eval_metrics['mean_hadamard_energy'],
                        'eval/std_hadamard_energy': eval_metrics['std_hadamard_energy'],
                        'eval/binary_rate': eval_metrics.get('binary_rate', 1.0),
                        'eval/valid_hadamard_rate': eval_metrics.get('valid_hadamard_rate', 0.0),
                        'eval/num_unique_values': len(eval_metrics['unique_values']),
                        'eval/epoch': epoch + 1
                    }

                    # Add optional metrics
                    if 'elbo_random' in eval_metrics:
                        wandb_metrics.update({
                            'eval/elbo_random': eval_metrics['elbo_random'],
                            'eval/elbo_difference': eval_metrics['elbo_difference']
                        })

                    if 'denoising_analysis' in eval_metrics and eval_metrics['denoising_analysis']:
                        denoising = eval_metrics['denoising_analysis']
                        wandb_metrics.update({
                            'eval/initial_hadamard_energy': denoising['initial_hadamard_energy'],
                            'eval/final_hadamard_energy': denoising['final_hadamard_energy']
                        })

                    wandb.log(wandb_metrics, step=global_step)

            except Exception as e:
                print(f"Evaluation failed: {e}")

            finally:
                # Restore training weights
                ema.restore(model)
                model.train()

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'ema_shadow': ema.shadow,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': epoch_loss / num_batches,
                'global_step': global_step,
                'model_kwargs': model_kwargs,
                'model_type': model_type,
                'loss_type': loss_type or default_loss_type
            }
            torch.save(checkpoint, Path(save_dir) / f"checkpoint_epoch_{epoch+1}.pt")

        print(f"Epoch {epoch+1} completed. Average loss: {epoch_loss/num_batches:.4f}")

        # Log epoch summary to wandb
        if use_wandb:
            wandb.log({
                'epoch/avg_loss': epoch_loss / num_batches,
                'epoch/number': epoch + 1
            }, step=global_step)

    # Save final model
    checkpoint = {
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'ema_shadow': ema.shadow,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': epoch_loss / num_batches,
        'global_step': global_step,
        'model_kwargs': model_kwargs,
        'model_type': model_type,
        'loss_type': loss_type or default_loss_type
    }
    torch.save(checkpoint, Path(save_dir) / "final_model.pt")
    print(f"Training completed. Final model saved to {save_dir}/final_model.pt")

    # Finish wandb run
    if use_wandb:
        wandb.finish()

    return model, ema


def train_hadamard_diffusion_preshuffled(
    file_path,
    num_epochs=100,
    batch_size=32,
    lr=1e-4,
    matrix_size=32,
    model_kwargs=None,
    device=None,
    save_dir="checkpoints",
    log_interval=100,
    eval_interval=5,
    eval_batch_size=8,
    model_type='score',  # 'score' or 'radd'
    loss_type=None,  # Auto-select based on model_type if None
    graph_type='uniform',  # 'uniform' or 'absorbing'
    use_wandb=True,
    wandb_project="hadamard-diffusion",
    wandb_run_name=None,
    eval_fraction=0.1,
    train_seed=42,
    eval_seed=43
):
    """
    Train Hadamard diffusion model using pre-shuffled numpy file

    Args:
        file_path: Path to pre-shuffled numpy memmap file
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        matrix_size: Size of Hadamard matrices
        model_kwargs: Additional arguments for HadamardScoreModel/HadamardRADDModel
        device: Device to train on
        save_dir: Directory to save checkpoints
        log_interval: Steps between logging
        eval_interval: Epochs between evaluation
        eval_batch_size: Batch size for evaluation
        model_type: 'score' (time-dependent) or 'radd' (time-independent)
        loss_type: 'score_entropy', 't_dce', or 'lambda_dce' (auto-selected if None)
        graph_type: 'uniform' (uniform diffusion) or 'absorbing' (absorbing diffusion)
        use_wandb: Whether to use Weights & Biases logging
        wandb_project: W&B project name
        wandb_run_name: W&B run name (auto-generated if None)
        eval_fraction: Fraction of data to use for evaluation
        train_seed: Random seed for training transformations
        eval_seed: Random seed for evaluation transformations
    """

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Training on device: {device}")

    # Create directories
    Path(save_dir).mkdir(exist_ok=True)

    # Create train/eval datasets from pre-shuffled file
    train_dataset, eval_dataset = create_train_eval_datasets(
        file_path=file_path,
        batch_size=batch_size,
        train_seed=train_seed,
        eval_seed=eval_seed,
        eval_fraction=eval_fraction
    )

    # Get a fixed evaluation batch for consistent monitoring
    eval_iter = iter(eval_dataset)
    eval_batch = next(eval_iter).to(device)

    # Initialize model and graph
    model_kwargs = model_kwargs or {}
    if model_type == 'score':
        model = HadamardScoreModel(matrix_size=matrix_size, **model_kwargs).to(device)
        default_loss_type = 'score_entropy'
    elif model_type == 'radd':
        model = HadamardRADDModel(matrix_size=matrix_size, **model_kwargs).to(device)
        default_loss_type = 't_dce'
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Initialize graph
    if graph_type == 'uniform':
        graph = BinaryUniformGraph()
    elif graph_type == 'absorbing':
        graph = BinaryAbsorbingGraph()
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")

    # Initialize optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = get_loss_fn(graph, loss_type=loss_type or default_loss_type)
    scaler = torch.cuda.amp.GradScaler()
    ema = EMA(model)

    # Learning rate scheduler - estimate steps per epoch
    estimated_steps_per_epoch = len(train_dataset) // batch_size
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * estimated_steps_per_epoch)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Estimated training batches per epoch: {estimated_steps_per_epoch}")

    # Initialize wandb logging
    if use_wandb:
        config = {
            'model_type': model_type,
            'loss_type': loss_type or default_loss_type,
            'graph_type': graph_type,
            'matrix_size': matrix_size,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': lr,
            'eval_fraction': eval_fraction,
            'train_seed': train_seed,
            'eval_seed': eval_seed,
            'total_train_matrices': len(train_dataset),
            'total_eval_matrices': len(eval_dataset)
        }

        if model_kwargs:
            config.update({f'model_{k}': v for k, v in model_kwargs.items()})

        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config=config
        )

        # Log model architecture
        wandb.watch(model, log_freq=log_interval)

    # Training loop
    global_step = 0
    best_eval_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        # Create fresh iterator for each epoch
        train_iter = iter(train_dataset)

        try:
            while True:
                batch = next(train_iter).to(device)

                optimizer.zero_grad()

                with torch.amp.autocast():
                    loss = loss_fn(model, batch)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                # Update EMA
                ema.update()

                epoch_loss += loss.item()
                num_batches += 1
                global_step += 1

                # Logging
                if global_step % log_interval == 0:
                    avg_loss = epoch_loss / num_batches
                    current_lr = scheduler.get_last_lr()[0]

                    print(f"Epoch {epoch+1}/{num_epochs}, Step {global_step}, "
                          f"Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}, LR: {current_lr:.2e}")

                    if use_wandb:
                        wandb.log({
                            'train/loss': loss.item(),
                            'train/avg_loss': avg_loss,
                            'train/learning_rate': current_lr,
                            'train/epoch': epoch + 1,
                            'train/global_step': global_step
                        })

        except StopIteration:
            # End of epoch
            pass

        # Evaluation
        if (epoch + 1) % eval_interval == 0:
            print("Running evaluation...")
            with ema.ema_scope():
                eval_metrics = evaluate_model(
                    model, eval_batch, graph, device,
                    matrix_size=matrix_size,
                    num_eval_samples=4,
                    compute_denoising_trajectory=True
                )

            current_eval_loss = eval_metrics['elbo/mean']
            print(f"Evaluation - ELBO: {current_eval_loss:.4f}")

            if use_wandb:
                wandb.log({**eval_metrics, 'eval/epoch': epoch + 1})

            # Save best model
            if current_eval_loss < best_eval_loss:
                best_eval_loss = current_eval_loss
                checkpoint_path = Path(save_dir) / f"best_model_epoch_{epoch+1}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'ema_state_dict': ema.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': best_eval_loss,
                    'model_kwargs': model_kwargs,
                    'model_type': model_type,
                    'graph_type': graph_type,
                    'matrix_size': matrix_size
                }, checkpoint_path)
                print(f"Saved best model to {checkpoint_path}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = Path(save_dir) / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'ema_state_dict': ema.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': epoch_loss / max(num_batches, 1),
                'model_kwargs': model_kwargs,
                'model_type': model_type,
                'graph_type': graph_type,
                'matrix_size': matrix_size
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    print("Training completed!")

    # Final evaluation
    print("Running final evaluation...")
    with ema.ema_scope():
        final_metrics = evaluate_model(
            model, eval_batch, graph, device,
            matrix_size=matrix_size,
            num_eval_samples=8,
            compute_denoising_trajectory=True
        )

    if use_wandb:
        wandb.log({**{f'final/{k}': v for k, v in final_metrics.items()}})
        wandb.finish()

    return model, final_metrics


def load_model(checkpoint_path, device=None):
    """Load a trained model from checkpoint"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_kwargs = checkpoint.get('model_kwargs', {})
    model_type = checkpoint.get('model_type', 'score')  # Default to score model

    # Initialize the correct model type
    if model_type == 'score':
        model = HadamardScoreModel(**model_kwargs).to(device)
    elif model_type == 'radd':
        model = HadamardRADDModel(**model_kwargs).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(checkpoint['model_state_dict'])

    # Load EMA weights if available
    if 'ema_shadow' in checkpoint:
        ema = EMA(model)
        ema.shadow = checkpoint['ema_shadow']
        ema.apply_shadow(model)

    return model
