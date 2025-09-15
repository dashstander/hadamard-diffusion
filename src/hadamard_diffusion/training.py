import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from pathlib import Path
from tqdm import tqdm

from .score_model import HadamardScoreModel, BinaryUniformGraph
from .sampling import get_binary_sampler, validate_hadamard_properties


class HadamardDataset(Dataset):
    """Dataset for loading Hadamard matrices from numpy files"""

    def __init__(self, data_dir, max_matrices=None):
        self.data_dir = Path(data_dir)
        self.matrix_files = list(self.data_dir.glob("hadamard_matrices_batch_*.npy"))
        self.max_matrices = max_matrices

        # Load all matrices (or subset)
        self.matrices = []
        total_loaded = 0

        for file_path in sorted(self.matrix_files):
            batch_matrices = np.load(file_path)
            if self.max_matrices and total_loaded + len(batch_matrices) > self.max_matrices:
                # Take only what we need
                remaining = self.max_matrices - total_loaded
                batch_matrices = batch_matrices[:remaining]

            self.matrices.extend(batch_matrices)
            total_loaded += len(batch_matrices)

            if self.max_matrices and total_loaded >= self.max_matrices:
                break

        print(f"Loaded {len(self.matrices)} Hadamard matrices")

    def __len__(self):
        return len(self.matrices)

    def __getitem__(self, idx):
        matrix = self.matrices[idx]
        return torch.from_numpy(matrix).float()


def get_loss_fn(graph, sampling_eps=1e-3):
    """Loss function for discrete diffusion training"""

    def loss_fn(model, batch):
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

    return loss_fn


def compute_elbo(model, x_clean, graph, sampling_eps=1e-3, num_samples=50):
    """
    Compute ELBO (Evidence Lower BOund) for held-out data

    Args:
        model: Trained score model
        x_clean: Clean Hadamard matrices (batch, height, width)
        graph: BinaryUniformGraph instance
        sampling_eps: Small epsilon for numerical stability
        num_samples: Number of timesteps to sample for Monte Carlo estimate

    Returns:
        ELBO estimate (scalar)
    """
    model.eval()

    with torch.no_grad():
        batch_size = x_clean.shape[0]
        device = x_clean.device

        # Sample timesteps for Monte Carlo integration
        t = (1 - sampling_eps) * torch.rand(num_samples, device=device) + sampling_eps

        elbo_estimates = []

        for i in range(num_samples):
            # Expand timestep to batch
            t_batch = t[i].expand(batch_size).unsqueeze(-1)

            # Forward diffusion: x_0 -> x_t
            x_noisy = graph.sample_transition(x_clean, t_batch)

            # Get model scores
            log_scores = model(x_noisy, t_batch)

            # Compute score entropy (the integrand of L_DWDSE)
            score_entropy = graph.score_entropy(log_scores, t_batch, x_noisy, x_clean)

            # Add to estimates
            elbo_estimates.append(score_entropy.mean())

        # Monte Carlo estimate of integral over time
        elbo_estimate = torch.stack(elbo_estimates).mean() * (1 - sampling_eps)

        # KL divergence to base distribution (usually small for uniform)
        # For uniform base distribution, this is approximately 0
        kl_divergence = 0.0

        return elbo_estimate.item() + kl_divergence


def evaluate_model(model, eval_data, graph, device, matrix_size=32, num_eval_samples=4):
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
    elbo_hadamard = compute_elbo(model, eval_data, graph)

    # 2. Generate random non-Hadamard matrices for comparison
    # Create random binary matrices that are unlikely to be Hadamard
    batch_size = eval_data.shape[0]
    random_matrices = graph.sample_limit(batch_size, matrix_size, matrix_size).to(device)

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

    # 4. Compute additional metrics

    # Check value distribution (should be close to {-1, +1})
    unique_vals = torch.unique(generated_samples)
    value_range = (generated_samples.min().item(), generated_samples.max().item())

    # Orthogonality errors for analysis
    orthogonal_errors = []
    for i in range(num_eval_samples):
        H = generated_samples[i]
        HHT = H @ H.T
        expected = matrix_size * torch.eye(matrix_size, device=H.device)
        error = torch.norm(HHT - expected).item()
        orthogonal_errors.append(error)

    return {
        'elbo_hadamard': elbo_hadamard,
        'elbo_random': elbo_random,
        'elbo_difference': elbo_hadamard - elbo_random,  # Should be positive if model learns
        'binary_rate': validation_results['binary_rate'],  # Always 100% due to discrete sampling
        'orthogonal_rate': validation_results['orthogonal_rate'],
        'valid_hadamard_rate': validation_results['valid_rate'],
        'mean_orthogonal_error': np.mean(orthogonal_errors),
        'std_orthogonal_error': np.std(orthogonal_errors),
        'value_range': value_range,
        'unique_values': unique_vals.cpu().tolist(),
        'num_unique_values': len(unique_vals)
    }


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
    data_dir,
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
    eval_batch_size=8
):
    """
    Train Hadamard diffusion model

    Args:
        data_dir: Directory containing hadamard_matrices_batch_*.npy files
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        max_matrices: Maximum number of matrices to load (None for all)
        matrix_size: Size of Hadamard matrices
        model_kwargs: Additional arguments for HadamardScoreModel
        device: Device to train on
        save_dir: Directory to save checkpoints
        log_interval: Steps between logging
        eval_interval: Epochs between evaluation
        eval_batch_size: Batch size for evaluation
    """

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Training on device: {device}")

    # Create directories
    Path(save_dir).mkdir(exist_ok=True)

    # Initialize dataset and split into train/eval
    dataset = HadamardDataset(data_dir, max_matrices=max_matrices)

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
    model = HadamardScoreModel(matrix_size=matrix_size, **model_kwargs).to(device)
    graph = BinaryUniformGraph()

    # Initialize optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = get_loss_fn(graph)
    scaler = GradScaler()
    ema = EMA(model)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(dataloader))

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training batches per epoch: {len(dataloader)}")

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
            with autocast():
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
                    num_eval_samples=4
                )

                print(f"Evaluation Results:")
                print(f"  ELBO (Hadamard): {eval_metrics['elbo_hadamard']:.4f}")
                print(f"  ELBO (Random): {eval_metrics['elbo_random']:.4f}")
                print(f"  ELBO Difference: {eval_metrics['elbo_difference']:.4f} (should be positive)")
                print(f"  Orthogonal rate: {eval_metrics['orthogonal_rate']:.2%}")
                print(f"  Mean orthogonal error: {eval_metrics['mean_orthogonal_error']:.4f}")
                print(f"  Value range: {eval_metrics['value_range']}")
                print(f"  Unique values: {len(eval_metrics['unique_values'])} ({eval_metrics['unique_values']})")
                # Note: Binary rate is always 100% due to discrete sampling architecture

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
                'model_kwargs': model_kwargs
            }
            torch.save(checkpoint, Path(save_dir) / f"checkpoint_epoch_{epoch+1}.pt")

        print(f"Epoch {epoch+1} completed. Average loss: {epoch_loss/num_batches:.4f}")

    # Save final model
    checkpoint = {
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'ema_shadow': ema.shadow,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': epoch_loss / num_batches,
        'global_step': global_step,
        'model_kwargs': model_kwargs
    }
    torch.save(checkpoint, Path(save_dir) / "final_model.pt")
    print(f"Training completed. Final model saved to {save_dir}/final_model.pt")

    return model, ema


def load_model(checkpoint_path, device=None):
    """Load a trained model from checkpoint"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_kwargs = checkpoint.get('model_kwargs', {})

    model = HadamardScoreModel(**model_kwargs).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load EMA weights if available
    if 'ema_shadow' in checkpoint:
        ema = EMA(model)
        ema.shadow = checkpoint['ema_shadow']
        ema.apply_shadow(model)

    return model


# Example usage
if __name__ == "__main__":
    # Small test run
    model, ema = train_hadamard_diffusion(
        data_dir="data/processed",  # Update path as needed
        num_epochs=5,
        batch_size=16,
        max_matrices=1000,  # Small subset for testing
        matrix_size=32,
        model_kwargs={
            'element_dim': 64,
            'pool_dim': 32,
            'num_heads': 4,
            'head_dim': 16,
            'ffn_hidden_dim': 128,
            'num_layers': 3
        }
    )