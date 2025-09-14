import torch

class Gaussian2D:
    """Single 2D Gaussian primitive."""
    
    def __init__(self, n_channels=3, device='cuda'):
        self.device = device
        # Initialize as Parameters for proper gradient tracking
        self.mu = torch.nn.Parameter(torch.zeros(2, device=device))
        self.theta = torch.nn.Parameter(torch.tensor(0.0, device=device))
        self.s = torch.nn.Parameter(torch.ones(2, device=device) * 5.0)
        self.inv_s = torch.nn.Parameter(torch.ones(2, device=device) / 5.0)
        self.c = torch.nn.Parameter(torch.zeros(n_channels, device=device))
    
    def compute_covariance(self):
        cos_theta = torch.cos(self.theta)
        sin_theta = torch.sin(self.theta)
        R = torch.stack([
            torch.stack([cos_theta, -sin_theta]),
            torch.stack([sin_theta, cos_theta])
        ])
        S = torch.diag(self.s)
        return R @ S @ S.T @ R.T
    
    def evaluate(self, x):
        """Evaluate Gaussian at position x."""
        diff = x - self.mu
        cov = self.compute_covariance()
        inv_cov = torch.inverse(cov + 1e-6 * torch.eye(2, device=self.device))
        return torch.exp(-0.5 * (diff @ inv_cov @ diff))