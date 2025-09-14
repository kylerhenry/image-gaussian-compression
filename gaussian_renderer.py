import torch

class GaussianRenderer:
    """Tile-based renderer for 2D Gaussians."""
    
    def __init__(self, tile_size=16, top_k=10):
        self.tile_size = tile_size
        self.top_k = top_k
    
    def render(self, gaussians, image_size, device='cuda'):
        H, W = image_size
        
        # Get number of channels from first Gaussian or default to 3
        if len(gaussians) > 0:
            n_channels = gaussians[0].c.shape[0]
        else:
            n_channels = 3  # Default to RGB
            
        output = torch.zeros((H, W, n_channels), device=device)
        
        # If no Gaussians, return black image
        if len(gaussians) == 0:
            return output
        
        n_tiles_h = (H + self.tile_size - 1) // self.tile_size
        n_tiles_w = (W + self.tile_size - 1) // self.tile_size
        
        for tile_y in range(n_tiles_h):
            for tile_x in range(n_tiles_w):
                y_start = tile_y * self.tile_size
                y_end = min(y_start + self.tile_size, H)
                x_start = tile_x * self.tile_size
                x_end = min(x_start + self.tile_size, W)
                
                tile_bounds = (x_start/W, y_start/H, x_end/W, y_end/H)
                relevant_gaussians = self._get_relevant_gaussians(
                    gaussians, tile_bounds, (H, W)
                )
                
                for y in range(y_start, y_end):
                    for x in range(x_start, x_end):
                        pixel_pos = torch.tensor([x/W, y/H], device=device)
                        output[y, x] = self._render_pixel(
                            pixel_pos, relevant_gaussians, n_channels
                        )
        
        return output
    
    def _render_pixel(self, pixel_pos, gaussians, n_channels):
        """Render a single pixel using top-K Gaussians."""
        if len(gaussians) == 0:
            return torch.zeros(n_channels, device=pixel_pos.device)
        
        weights = []
        colors = []
        for g in gaussians:
            weight = g.evaluate(pixel_pos)
            weights.append(weight)
            colors.append(g.c)
        
        if len(weights) == 0:
            return torch.zeros(n_channels, device=pixel_pos.device)
            
        weights = torch.stack(weights)
        colors = torch.stack(colors)
        
        if len(weights) > self.top_k:
            top_k_indices = torch.topk(weights, min(self.top_k, len(weights))).indices
            weights = weights[top_k_indices]
            colors = colors[top_k_indices]
        
        if weights.sum() > 0:
            weights = weights / weights.sum()
            pixel_color = (weights.unsqueeze(1) * colors).sum(dim=0)
        else:
            pixel_color = torch.zeros_like(colors[0])
        
        return pixel_color
    
    def _get_relevant_gaussians(self, gaussians, tile_bounds, image_size):
        """Find Gaussians whose 3-sigma range intersects with tile."""
        x_min, y_min, x_max, y_max = tile_bounds
        H, W = image_size
        relevant = []
        
        for g in gaussians:
            # Convert Gaussian scale from pixels to normalized coordinates
            # g.s is in pixels, we need to convert to [0,1] space
            radius_x = 3 * g.s[0].item() / W
            radius_y = 3 * g.s[1].item() / H
            radius = max(radius_x, radius_y)
            
            gx, gy = g.mu[0].item(), g.mu[1].item()
            
            # Check if Gaussian's bounding box intersects with tile
            if (gx + radius >= x_min and gx - radius <= x_max and
                gy + radius >= y_min and gy - radius <= y_max):
                relevant.append(g)
        
        return relevant