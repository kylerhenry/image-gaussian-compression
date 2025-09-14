import torch
import numpy as np
from gaussian_2d import Gaussian2D
from gaussian_renderer import GaussianRenderer
from utils import compute_ssim


def content_adaptive_initialization(image, n_gaussians, lambda_init=0.3):
    """Initialize Gaussian positions based on image gradients."""
    device = image.device
    H, W = image.shape[:2]
    total_pixels = H * W
    
    # Compute simple gradients
    if image.shape[2] > 1:
        gray = (image * torch.tensor([0.299, 0.587, 0.114], device=device)).sum(dim=2)
    else:
        gray = image.squeeze(dim=2)
    
    dx = torch.diff(gray, dim=1, prepend=gray[:, :1])
    dy = torch.diff(gray, dim=0, prepend=gray[:1, :])
    gradient_magnitude = torch.sqrt(dx**2 + dy**2)
    
    # Handle large images - PyTorch multinomial has a limit of 2^24 categories
    MAX_CATEGORIES = 2**24 - 1
    
    if total_pixels > MAX_CATEGORIES:
        print(f"Large image detected ({H}x{W} = {total_pixels} pixels). Using chunked sampling.")
        
        # Downsample the probability map for sampling
        downsample_factor = int(np.ceil(np.sqrt(total_pixels / MAX_CATEGORIES)))
        
        # Downsample gradient magnitude
        grad_downsampled = torch.nn.functional.avg_pool2d(
            gradient_magnitude.unsqueeze(0).unsqueeze(0),
            kernel_size=downsample_factor,
            stride=downsample_factor
        ).squeeze()
        
        H_down, W_down = grad_downsampled.shape
        
        # Create sampling probability on downsampled grid
        gradient_prob = grad_downsampled.flatten()
        gradient_prob = gradient_prob / (gradient_prob.sum() + 1e-8)
        uniform_prob = torch.ones_like(gradient_prob) / gradient_prob.numel()
        sampling_prob = (1 - lambda_init) * gradient_prob + lambda_init * uniform_prob
        
        # Sample from downsampled grid
        pixel_indices = torch.multinomial(sampling_prob, n_gaussians, replacement=True)
        
        positions = []
        colors = []
        total_indices = len(pixel_indices)
        for idx, tensor in enumerate(pixel_indices):
            if (idx + 1) % 1000 == 0 or idx == total_indices - 1:
                print(f"  Sampled {idx + 1}/{total_indices} Gaussians", end='\r')
            y_down = (tensor // W_down).item()
            x_down = (tensor % W_down).item()
            
            # Map to original image coordinates
            y = min(y_down * downsample_factor + np.random.randint(0, downsample_factor), H-1)
            x = min(x_down * downsample_factor + np.random.randint(0, downsample_factor), W-1)
            
            positions.append([x/W, y/H])
            colors.append(image[y, x].detach().cpu().numpy())
    
    else:
        # Original method for smaller images
        gradient_prob = gradient_magnitude.flatten()
        gradient_prob = gradient_prob / (gradient_prob.sum() + 1e-8)
        uniform_prob = torch.ones_like(gradient_prob) / gradient_prob.numel()
        sampling_prob = (1 - lambda_init) * gradient_prob + lambda_init * uniform_prob
        
        pixel_indices = torch.multinomial(sampling_prob, n_gaussians, replacement=True)
        
        positions = []
        colors = []
        for idx in pixel_indices:
            y = (idx // W).item()
            x = (idx % W).item()
            positions.append([x/W, y/H])
            colors.append(image[y, x].detach().cpu().numpy())
    
    return positions, colors


def add_gaussians_by_error_large(current_gaussians, target_image, n_add, renderer, device):
    """Add new Gaussians to high-error regions (handles large images)."""
    H, W = target_image.shape[:2]
    n_channels = target_image.shape[2]
    total_pixels = H * W
    
    # Render current state
    rendered = renderer.render(current_gaussians, (H, W), device=device)
    
    # Compute error
    error = torch.abs(rendered - target_image).mean(dim=-1)
    
    MAX_CATEGORIES = 2**24 - 1
    
    new_gaussians = []
    
    if total_pixels > MAX_CATEGORIES:
        # Downsample error map for sampling
        downsample_factor = int(np.ceil(np.sqrt(total_pixels / MAX_CATEGORIES)))
        
        error_downsampled = torch.nn.functional.avg_pool2d(
            error.unsqueeze(0).unsqueeze(0),
            kernel_size=downsample_factor,
            stride=downsample_factor
        ).squeeze()
        
        H_down, W_down = error_downsampled.shape
        error_flat = error_downsampled.flatten()
        error_prob = error_flat / (error_flat.sum() + 1e-8)
        
        pixel_indices = torch.multinomial(error_prob, n_add, replacement=True)
        
        for idx in pixel_indices:
            y_down = (idx // W_down).item()
            x_down = (idx % W_down).item()
            
            y = min(y_down * downsample_factor + np.random.randint(0, downsample_factor), H-1)
            x = min(x_down * downsample_factor + np.random.randint(0, downsample_factor), W-1)
            
            g = Gaussian2D(n_channels, device=device)
            with torch.no_grad():
                g.mu.data[:] = torch.tensor([x/W, y/H], device=device)
                g.c.data[:] = target_image[y, x]
            new_gaussians.append(g)
    
    else:
        error_flat = error.flatten()
        error_prob = error_flat / (error_flat.sum() + 1e-8)
        
        pixel_indices = torch.multinomial(error_prob, n_add, replacement=True)
        
        for idx in pixel_indices:
            y = (idx // W).item()
            x = (idx % W).item()
            
            g = Gaussian2D(n_channels, device=device)
            with torch.no_grad():
                g.mu.data[:] = torch.tensor([x/W, y/H], device=device)
                g.c.data[:] = target_image[y, x]
            new_gaussians.append(g)
    
    return new_gaussians


class ImageGS:
    """Main Image-GS system for image compression and reconstruction."""
    
    def __init__(self, n_gaussians=10000, learning_rates=None, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.n_gaussians = n_gaussians
        self.lr = learning_rates or {
            'position': 5e-4,
            'color': 5e-3,
            'scale': 2e-3,
            'rotation': 2e-3
        }
        self.renderer = GaussianRenderer()
    
    def fit(self, target_image, n_steps=5000, verbose=True):
        """Fit Gaussians to target image."""
        H, W = target_image.shape[:2]
        n_channels = target_image.shape[2]
        
        print(f"Fitting {self.n_gaussians} Gaussians to {H}x{W} image...")
        
        # Initialize Gaussians
        initial_gaussians = self.n_gaussians // 2
        positions, colors = content_adaptive_initialization(
            target_image, initial_gaussians, lambda_init=0.3
        )
        
        # Create Gaussian objects with proper initialization
        gaussians = []
        for i in range(initial_gaussians):
            if i % 1000 == 0 or i == initial_gaussians - 1:
                print(f"  Initialized {i + 1}/{initial_gaussians} Gaussians", end='\r')
            g = Gaussian2D(n_channels, device=self.device)
            # Use torch.no_grad() to avoid creating computation graph
            with torch.no_grad():
                g.mu.data[:] = torch.tensor(positions[i], device=self.device, dtype=torch.float32)
                g.c.data[:] = torch.tensor(colors[i], device=self.device, dtype=torch.float32)
            gaussians.append(g)
        
        # Setup optimizer - use the Parameter objects directly
        params = []
        print("\nStarting optimization...")
        for i, g in enumerate(gaussians):
            if i % 10 == 0 or i == len(gaussians) - 1:
                print(f"  Added {i + 1}/{len(gaussians)} Gaussians to optimizer", end='\r')
            params.extend([
                {'params': g.mu, 'lr': self.lr['position']},
                {'params': g.c, 'lr': self.lr['color']},
                {'params': g.inv_s, 'lr': self.lr['scale']},
                {'params': g.theta, 'lr': self.lr['rotation']}
            ])
        optimizer = torch.optim.Adam(params)
        
        # Training loop
        for step in range(n_steps):
            if step % 2 == 0:
                print(f"\nStep {step}/{n_steps}", end='\r')
            # Update actual scale from inverse
            with torch.no_grad():
                for g in gaussians:
                    g.s.data[:] = 1.0 / (g.inv_s.data + 1e-6)
            
            # Progressive Gaussian addition
            if step > 0 and step % 500 == 0 and len(gaussians) < self.n_gaussians:
                new_gaussians = add_gaussians_by_error_large(
                    gaussians, target_image, 
                    min(self.n_gaussians // 8, self.n_gaussians - len(gaussians)),
                    self.renderer, self.device
                )
                gaussians.extend(new_gaussians)
                
                for g in new_gaussians:
                    print(f"  Added new Gaussian, total now {len(new_gaussians)}", end='\r')
                    optimizer.add_param_group({'params': g.mu, 'lr': self.lr['position']})
                    optimizer.add_param_group({'params': g.c, 'lr': self.lr['color']})
                    optimizer.add_param_group({'params': g.inv_s, 'lr': self.lr['scale']})
                    optimizer.add_param_group({'params': g.theta, 'lr': self.lr['rotation']})
            
            # Render
            print("Rendering...", end='\r')
            rendered = self.renderer.render(gaussians, (H, W), device=self.device)
            
            # Compute loss
            print("Computing loss...", end='\r')
            l1_loss = torch.abs(rendered - target_image).mean()
            ssim_loss = 1 - compute_ssim(rendered, target_image)
            loss = l1_loss + 0.1 * ssim_loss
            
            # Backprop
            print("Back propagating...", end='\r')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Clamp parameters
            print("Clamping parameters...", end='\r')
            with torch.no_grad():
                for g in gaussians:
                    g.mu.data.clamp_(0, 1)
                    g.theta.data.clamp_(0, np.pi)
                    g.inv_s.data.clamp_min_(0.01)
            
            if verbose and step % 100 == 0:
                print(f"Step {step}: Loss = {loss.item():.4f}, Gaussians = {len(gaussians)}")
        
        return gaussians
    
    def reconstruct(self, gaussians, image_size):
        """Reconstruct image from Gaussians."""
        return self.renderer.render(gaussians, image_size, device=self.device)