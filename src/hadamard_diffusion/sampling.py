import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from tqdm import tqdm

from .score_model import BinaryUniformGraph


def sample_categorical(probs, method="hard"):
    """Sample from categorical distribution"""
    if method == "hard":
        return Categorical(probs).sample()
    else:
        # Gumbel-softmax for differentiable sampling
        gumbel = -torch.log(-torch.log(torch.rand_like(probs) + 1e-8) + 1e-8)
        return F.softmax((torch.log(probs + 1e-8) + gumbel) / 0.1, dim=-1)


class BinaryEulerPredictor:
    """Euler predictor for binary uniform diffusion"""

    def __init__(self, graph):
        self.graph = graph

    def update_fn(self, score_fn, x, t, step_size):
        """One Euler update step"""
        sigma = t  # In our parameterization, sigma = t
        dsigma = step_size

        # Get scores from model
        log_scores = score_fn(x, sigma)
        scores = log_scores.exp()

        # For uniform binary diffusion, the reverse rate is:
        # R(x->y) = score(y) * Q^T(x->y) where Q^T is transpose rate
        batch_size = x.shape[0]

        # Convert current state to indices
        x_indices = self.graph.value_to_index(x)  # (batch, height, width)

        # Compute reverse rates
        # For binary uniform: rate from 0->1 and 1->0 is 1/2 each (normalized)
        # The reverse rate incorporates the score
        rate_matrix = torch.zeros_like(scores)  # (batch, height, width, 2)

        # Rate away from current state
        rate_matrix.scatter_(-1, x_indices.unsqueeze(-1), -scores.sum(dim=-1, keepdim=True))

        # Rate to other states (incorporating scores)
        other_indices = 1 - x_indices  # Flip 0<->1
        rate_matrix.scatter_add_(-1, other_indices.unsqueeze(-1), scores)

        # Apply time step
        rate_update = dsigma * rate_matrix

        # Convert to probabilities (adding to one-hot current state)
        x_onehot = F.one_hot(x_indices, num_classes=2).float()
        probs = x_onehot + rate_update

        # Ensure probabilities are valid
        probs = torch.clamp(probs, min=1e-8)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        # Sample new state
        new_indices = sample_categorical(probs)
        return self.graph.index_to_value(new_indices)


class BinaryAnalyticPredictor:
    """Analytic predictor using exact transition probabilities"""

    def __init__(self, graph):
        self.graph = graph

    def update_fn(self, score_fn, x, t, step_size):
        """One analytic update step"""
        curr_sigma = t
        next_sigma = t - step_size
        dsigma = curr_sigma - next_sigma

        # Get scores
        log_scores = score_fn(x, curr_sigma)
        scores = log_scores.exp()

        # For uniform diffusion, staggered score is modified by exp factor
        # This comes from the exact solution of the forward process
        exp_factor = torch.exp(-dsigma)
        staggered_scores = scores * exp_factor.unsqueeze(-1)

        # Add the transition correction
        x_indices = self.graph.value_to_index(x)
        transition_correction = (1 - exp_factor) / 2  # Uniform over 2 states

        # Build transition probabilities
        probs = torch.zeros_like(scores)
        probs.scatter_(-1, x_indices.unsqueeze(-1), exp_factor.unsqueeze(-1))
        probs += transition_correction.unsqueeze(-1) * staggered_scores

        # Normalize
        probs = probs / probs.sum(dim=-1, keepdim=True)

        # Sample
        new_indices = sample_categorical(probs)
        return self.graph.index_to_value(new_indices)


class BinaryDenoiser:
    """Final denoising step for binary diffusion"""

    def __init__(self, graph):
        self.graph = graph

    def update_fn(self, score_fn, x, t):
        """Final denoising step"""
        sigma = t

        # Get scores
        log_scores = score_fn(x, sigma)

        # For denoising, take the argmax of the log scores
        # This gives us the most likely clean state
        new_indices = log_scores.argmax(dim=-1)
        return self.graph.index_to_value(new_indices)


def get_binary_sampler(
    model,
    graph,
    matrix_size=32,
    steps=100,
    predictor="euler",
    denoise=True,
    eps=1e-5,
    device=None,
    show_progress=True
):
    """
    Create a sampling function for binary Hadamard matrices

    Args:
        model: Trained HadamardScoreModel
        graph: BinaryUniformGraph instance
        matrix_size: Size of matrices to generate
        steps: Number of diffusion steps
        predictor: "euler" or "analytic"
        denoise: Whether to apply final denoising step
        eps: Small epsilon for numerical stability
        device: Device to run on
        show_progress: Whether to show progress bar

    Returns:
        Sampling function that takes batch_size and returns generated matrices
    """

    if device is None:
        device = next(model.parameters()).device

    # Choose predictor
    if predictor == "euler":
        predictor_fn = BinaryEulerPredictor(graph)
    elif predictor == "analytic":
        predictor_fn = BinaryAnalyticPredictor(graph)
    else:
        raise ValueError(f"Unknown predictor: {predictor}")

    denoiser = BinaryDenoiser(graph)

    def sample_fn(batch_size):
        """Generate batch_size Hadamard matrices"""
        model.eval()

        with torch.no_grad():
            # Start from pure noise (uniform random)
            x = graph.sample_limit(batch_size, matrix_size, matrix_size).to(device)

            # Create timestep schedule
            timesteps = torch.linspace(1.0, eps, steps + 1, device=device)
            dt = (1.0 - eps) / steps

            # Reverse diffusion process
            iterator = range(steps)
            if show_progress:
                iterator = tqdm(iterator, desc="Sampling")

            for i in iterator:
                t = timesteps[i].expand(batch_size, 1)
                x = predictor_fn.update_fn(model, x, t, dt)

            # Final denoising step
            if denoise:
                t_final = timesteps[-1].expand(batch_size, 1)
                x = denoiser.update_fn(model, x, t_final)

        return x

    return sample_fn


def validate_hadamard_properties(matrices, tolerance=1e-6):
    """
    Validate that generated matrices satisfy Hadamard properties

    Args:
        matrices: (batch, height, width) tensor of matrices
        tolerance: Numerical tolerance for checks

    Returns:
        dict with validation results
    """
    batch_size, height, width = matrices.shape
    results = {
        'orthogonal': [],
        'binary': [],
        'all_valid': []
    }

    for i in range(batch_size):
        H = matrices[i]

        # Check binary values {-1, +1}
        is_binary = torch.all((H == 1) | (H == -1)).item()
        results['binary'].append(is_binary)

        # Check orthogonality: H @ H.T = n * I
        HHT = H @ H.T
        expected = height * torch.eye(height, device=H.device)
        orthogonal_error = torch.norm(HHT - expected).item()
        is_orthogonal = orthogonal_error < tolerance * height
        results['orthogonal'].append(is_orthogonal)

        # Overall validity
        results['all_valid'].append(is_binary and is_orthogonal)

    # Summary statistics
    results['binary_rate'] = np.mean(results['binary'])
    results['orthogonal_rate'] = np.mean(results['orthogonal'])
    results['valid_rate'] = np.mean(results['all_valid'])

    return results


# Example usage and testing
if __name__ == "__main__":
    from .score_model import HadamardScoreModel

    # Create a simple test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    matrix_size = 8  # Small for testing
    batch_size = 4

    # Create untrained model (for testing the sampling mechanics)
    model = HadamardScoreModel(
        matrix_size=matrix_size,
        element_dim=64,
        num_layers=2
    ).to(device)

    graph = BinaryUniformGraph()

    # Create sampler
    sampler = get_binary_sampler(
        model=model,
        graph=graph,
        matrix_size=matrix_size,
        steps=20,  # Few steps for testing
        predictor="euler",
        device=device
    )

    # Generate samples
    print("Generating samples...")
    samples = sampler(batch_size)

    print(f"Generated samples shape: {samples.shape}")
    print(f"Sample values range: [{samples.min():.3f}, {samples.max():.3f}]")
    print(f"Sample matrix (first one):\n{samples[0]}")

    # Validate properties
    results = validate_hadamard_properties(samples)
    print(f"\nValidation results:")
    print(f"Binary rate: {results['binary_rate']:.2%}")
    print(f"Orthogonal rate: {results['orthogonal_rate']:.2%}")
    print(f"Overall valid rate: {results['valid_rate']:.2%}")