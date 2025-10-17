import torch



class BinaryUniformGraph:
    """Simplified uniform graph for binary {-1, +1} states"""

    def __init__(self):
        self.dim = 2  # Two classes: -1 and +1

    def value_to_index(self, x):
        """Convert {-1, +1} values to {0, 1} indices"""
        return torch.where(x == -1, 0, 1).long()

    def index_to_value(self, indices):
        """Convert {0, 1} indices to {-1, +1} values"""
        return torch.where(indices == 0, -1, 1)

    def sample_transition(self, x, sigma):
        """Sample from the uniform transition for binary case"""
        # Convert values to indices
        indices = self.value_to_index(x)

        # Uniform transition: stay with prob exp(-sigma), flip with prob 1-exp(-sigma)
        move_chance = 1 - torch.exp(-sigma)

        # Ensure move_chance broadcasts correctly with x
        while move_chance.dim() < x.dim():
            move_chance = move_chance.unsqueeze(-1)

        should_flip = torch.rand_like(x.float()) < move_chance

        # Flip indices where needed
        new_indices = torch.where(should_flip, 1 - indices, indices)

        # Convert back to values
        return self.index_to_value(new_indices)

    def score_entropy(self, log_score, sigma, x, x0):
        """Score entropy loss for uniform binary diffusion"""
        # Convert to indices
        x_indices = self.value_to_index(x)
        x0_indices = self.value_to_index(x0)

        # Extract scores for current states
        score = log_score.exp()  # Convert from log-space

        # Positive term: E[exp(score)]
        pos_term = score.mean(dim=-1)

        # Negative term: score at current position
        neg_term = torch.gather(score, -1, x_indices.unsqueeze(-1)).squeeze(-1)

        # Constant correction term (simplified for uniform case)
        # This depends on whether we moved or stayed
        moved = (x_indices != x0_indices)
        sigma_flat = sigma.squeeze(-1)
        esigma_minus_1 = torch.expm1(sigma_flat)

        # Ensure proper broadcasting for batch dimensions
        while esigma_minus_1.dim() < moved.dim():
            esigma_minus_1 = esigma_minus_1.unsqueeze(-1)
        while sigma_flat.dim() < moved.dim():
            sigma_flat = sigma_flat.unsqueeze(-1)

        # Correction based on transition probabilities
        const_term = torch.where(
            moved,
            -torch.log(esigma_minus_1) / 2,  # If we moved
            torch.log1p(-torch.exp(-sigma_flat)) / 2  # If we stayed
        )

        return pos_term - neg_term / 2 + const_term

    def sample_limit(self, *batch_dims):
        """Sample from limiting distribution (uniform over {-1, +1})"""
        indices = torch.randint(0, 2, batch_dims)
        return self.index_to_value(indices)
    


class BinaryAbsorbingGraph:
    """Absorbing diffusion graph for binary {-1, +1} values with mask token"""

    def __init__(self, mask_token_id=2):
        self.dim = 3  # Three classes: -1, +1, mask
        self.mask_token_id = mask_token_id
        self.absorb = True

    def value_to_index(self, x):
        """Convert {-1, +1, mask} values to {0, 1, 2} indices"""
        # Handle mask tokens (0 -> 2)
        mask = (x == 0)
        indices = torch.where(x == -1, 0, 1).long()
        indices[mask] = self.mask_token_id
        return indices

    def index_to_value(self, indices):
        """Convert {0, 1, 2} indices to {-1, +1, 0} values"""
        # Handle mask tokens (2 -> 0)
        mask = (indices == self.mask_token_id)
        values = torch.where(indices == 0, -1, 1)
        values[mask] = 0
        return values

    def sample_transition(self, x, sigma):
        """Sample from absorbing transition: x -> mask with rate based on sigma"""
        # Convert values to indices
        indices = self.value_to_index(x)

        # Absorbing transition: transition to mask token with prob 1-exp(-sigma)
        move_chance = 1 - torch.exp(-sigma)

        # Ensure proper broadcasting
        while move_chance.dim() < x.dim():
            move_chance = move_chance.unsqueeze(-1)

        # Only non-mask tokens can transition to mask
        non_mask = (indices != self.mask_token_id)
        should_absorb = torch.rand_like(x.float()) < move_chance

        # Apply absorption only to non-mask positions
        new_indices = indices.clone()
        absorb_mask = non_mask & should_absorb
        new_indices[absorb_mask] = self.mask_token_id

        # Convert back to values
        return self.index_to_value(new_indices)

    def sample_limit(self, *batch_dims):
        """Sample from limiting distribution (uniform over {-1, +1})"""
        indices = torch.randint(0, 2, batch_dims)
        return self.index_to_value(indices)

    def sample_prior(self, *batch_dims):
        """Sample from prior distribution (all mask tokens)"""
        indices = torch.full(batch_dims, self.mask_token_id, dtype=torch.long)
        return self.index_to_value(indices)

