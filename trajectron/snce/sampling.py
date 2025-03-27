import math
import torch

class EventSampler:
    '''
    Implementation of Multi-Modal Safety Critical (MMSC) sampling strategy
    for Safety-aware Noise Contrastive Estimation (SaNCE)
    '''

    def __init__(self, device='cuda'):
        # Safety parameters
        self.noise_local = 0.02                  # ε_perturbations scale
        self.min_separation = 0.2                # Minimum safe distance
        self.max_separation = 2.0                # Maximum sampling radius
        self.n_directions = 20                  # Number of directions (v in θc = π/10 * v)
        
        # Create directional vectors for negative sampling
        angles = torch.linspace(0, 2*math.pi * (19/20), self.n_directions, device=device)  # Fix for older PyTorch
        self.directions = torch.stack([
            torch.cos(angles),
            torch.sin(angles)
        ], dim=1)
        
        self.device = device

    def _valid_check(self, pos_seed, neg_seed):
        '''
        Validate samples based on safety constraints
        '''
        dist = (neg_seed - pos_seed.unsqueeze(1)).norm(dim=-1)
        mask_valid = (dist > self.min_separation) & (dist < self.max_separation)
        
        dmin = torch.where(
            torch.isnan(dist[mask_valid]), 
            torch.full_like(dist[mask_valid], float('inf')), 
            dist[mask_valid]
        ).min()
        assert dmin > self.min_separation

        return mask_valid.unsqueeze(-1)

    def _generate_displacement(self, batch_size, n_samples, radius_scale=1.0):
        '''
        Generate displacement vectors pd = (ρ cos θc, ρ sin θc)
        '''
        # Generate ρ = radius_max * √u with scaling
        u = torch.rand(batch_size, n_samples, device=self.device)
        rho = self.max_separation * radius_scale * torch.sqrt(u)
        
        # Generate θc = π/10 * v
        v = torch.randint(0, self.n_directions, (batch_size, n_samples), device=self.device)
        theta = (math.pi/10) * v
        
        # Calculate pd = (ρ cos θc, ρ sin θc)
        pd = torch.stack([
            rho * torch.cos(theta),
            rho * torch.sin(theta)
        ], dim=-1)
        
        return pd

    def safety_sampling(self, primary_curr, primary_next, neighbors_next):
        '''
        Safety-critical sampling based on neighboring agents' positions
        '''
        batch_size = primary_next.size(0)
        n_neighbors = neighbors_next.size(1)
        
        # Generate positive samples
        sample_pos = primary_next[..., :2] - primary_curr[:, -1:, :2]
        sample_pos += torch.randn_like(sample_pos) * self.noise_local
        
        # Generate negative samples
        # First, calculate base positions of neighbors relative to primary agent
        base_positions = neighbors_next[..., :2] - primary_curr[:, -1:, :2].unsqueeze(1)
        
        # Generate directions correctly
        angles = torch.linspace(0, 2*math.pi * (self.n_directions-1)/self.n_directions, self.n_directions, device=self.device)
        directions = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)  # [n_directions, 2]
        
        # Process each timestep separately to avoid complex reshaping
        all_neg_samples = []
        all_masks = []
        
        n_timesteps = primary_next.size(1)
        for t in range(n_timesteps):
            # Get base positions for this timestep
            base_pos_t = base_positions[:, :, t]  # [batch_size, n_neighbors, 2]
            
            # Generate random radii
            radii = torch.rand(batch_size, n_neighbors, self.n_directions, 1, device=self.device)
            radii = radii * self.max_separation * 0.8 + self.max_separation * 0.2
            
            # Calculate displacements
            displacements = radii * directions.unsqueeze(0).unsqueeze(0)  # [1, 1, n_directions, 2]
            
            # Add displacements to base positions
            sample_neg_t = base_pos_t.unsqueeze(2) + displacements  # [batch_size, n_neighbors, n_directions, 2]
            sample_neg_t = sample_neg_t.reshape(batch_size, n_neighbors * self.n_directions, 2)
            
            # Add noise
            sample_neg_t += torch.randn_like(sample_neg_t) * self.noise_local
            
            # Validate samples
            mask_valid_t = self._valid_check(sample_pos[:, t], sample_neg_t)
            
            # IMPORTANT FIX: Expand the mask to match sample_neg_t dimensions
            mask_valid_t = mask_valid_t.expand(-1, -1, 2)
            
            sample_neg_t.masked_fill_(~mask_valid_t, float('nan'))
            
            # Add dimension for timestep
            sample_neg_t = sample_neg_t.unsqueeze(2)  # [batch_size, n_neighbors*n_directions, 1, 2]
            mask_valid_t = mask_valid_t.unsqueeze(2)  # Also unsqueeze mask to match
            
            all_neg_samples.append(sample_neg_t)
            all_masks.append(mask_valid_t)
        
        # Combine results for all timesteps
        sample_neg = torch.cat(all_neg_samples, dim=2)  # [batch_size, n_neighbors*n_directions, n_timesteps, 2]
        mask_valid = torch.cat(all_masks, dim=2)        # [batch_size, n_neighbors*n_directions, n_timesteps, 2]
        
        return sample_pos, sample_neg, mask_valid

    def local_sampling(self, primary_curr, primary_next, neighbors_next):
        '''
        Draw negative samples centered around the primary agent in the future
        '''
        # positive
        sample_pos = primary_next[..., :2] - primary_curr[..., :2]
        sample_pos += torch.rand(sample_pos.size(), device=self.device).sub(0.5) * self.noise_local

        # neighbor territory
        radius = torch.rand(sample_pos.size(0), 20, device=self.device) * self.max_separation * 0.8 + self.max_separation * 0.2
        theta = torch.rand(sample_pos.size(0), 20, device=self.device) * 2 * math.pi
        dx = radius * torch.cos(theta)
        dy = radius * torch.sin(theta)

        sample_neg = torch.stack([dx, dy], axis=2).unsqueeze(axis=2) + sample_pos.unsqueeze(1)

        # Get mask
        mask_valid = self._valid_check(sample_pos, sample_neg)
        
        # IMPORTANT FIX: Expand the mask to match sample_neg dimensions
        mask_valid = mask_valid.expand_as(sample_neg)
        
        sample_neg.masked_fill_(~mask_valid, float('nan'))

        return sample_pos, sample_neg, mask_valid