import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
import pickle
import json

# Try to import rawpy for RAW image support
try:
    import rawpy
    RAW_SUPPORT = True
except ImportError:
    RAW_SUPPORT = False
    print("Warning: rawpy not installed. RAW image support disabled.")
    print("Install with: pip install rawpy")


def load_image(image_path, target_size=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Load and preprocess an image for Image-GS system.
    Supports standard formats (JPEG, PNG) and RAW formats (ARW, CR2, NEF, DNG, etc.)
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Check if it's a RAW format
    raw_extensions = {'.arw', '.cr2', '.nef', '.dng', '.raf', '.orf', '.rw2', '.pef', '.srw'}
    is_raw = image_path.suffix.lower() in raw_extensions
    
    if is_raw:
        if not RAW_SUPPORT:
            raise ValueError("RAW format detected but rawpy is not installed. Install with: pip install rawpy")
        
        print(f"Loading RAW image: {image_path}")
        
        # Load RAW image using rawpy
        with rawpy.imread(str(image_path)) as raw:
            # Process RAW to RGB
            # use_camera_wb uses camera white balance
            # no_auto_bright prevents automatic brightness adjustment
            rgb = raw.postprocess(
                use_camera_wb=True,
                no_auto_bright=True,
                output_bps=16  # 16-bit output for better precision
            )
        
        # Convert to float32 and normalize to [0, 1]
        if rgb.dtype == np.uint16:
            image_np = rgb.astype(np.float32) / 65535.0
        else:
            image_np = rgb.astype(np.float32) / 255.0
        
        # Create PIL image for resizing if needed
        if target_size is not None:
            # Convert to 8-bit for PIL (PIL doesn't handle 16-bit well)
            img_8bit = (image_np * 255).astype(np.uint8)
            pil_image = Image.fromarray(img_8bit)
        
    else:
        # Standard image loading with PIL
        try:
            pil_image = Image.open(image_path)
        except Exception as e:
            raise ValueError(f"Failed to load image: {e}")
        
        # Handle different image modes
        if pil_image.mode == 'RGBA':
            background = Image.new('RGB', pil_image.size, (255, 255, 255))
            background.paste(pil_image, mask=pil_image.split()[3])
            pil_image = background
        elif pil_image.mode == 'P':
            pil_image = pil_image.convert('RGB')
        elif pil_image.mode not in ['RGB', 'L']:
            pil_image = pil_image.convert('RGB')
        
        # Convert to numpy array immediately if not resizing
        if target_size is None:
            image_np = np.array(pil_image, dtype=np.float32) / 255.0
    
    # Resize if needed
    if target_size is not None:
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        
        if not is_raw or (is_raw and 'pil_image' in locals()):
            pil_image = pil_image.resize((target_size[1], target_size[0]), Image.LANCZOS)
            image_np = np.array(pil_image, dtype=np.float32) / 255.0
        else:
            # For RAW images, resize using numpy/scipy if PIL conversion wasn't done
            from scipy import ndimage
            zoom_factors = [target_size[0] / image_np.shape[0], 
                          target_size[1] / image_np.shape[1], 
                          1]
            image_np = ndimage.zoom(image_np, zoom_factors, order=1)
    
    # Ensure shape is (H, W, C)
    if len(image_np.shape) == 2:
        image_np = np.expand_dims(image_np, axis=-1)
    
    # Convert to torch tensor
    image_tensor = torch.from_numpy(image_np).to(device)
    
    print(f"Loaded image: shape={image_tensor.shape}, range=[{image_tensor.min():.3f}, {image_tensor.max():.3f}]")
    
    return image_tensor.contiguous()


def save_image_as_jpeg(image_tensor, save_path, quality=95):
    """Save tensor as JPEG for comparison."""
    save_path = Path(save_path)
    
    if image_tensor.is_cuda:
        image_tensor = image_tensor.cpu()
    
    image_np = np.clip(image_tensor.numpy(), 0, 1)
    image_np = (image_np * 255).astype(np.uint8)
    
    if image_np.shape[2] == 1:
        image_np = image_np.squeeze(axis=2)
    
    pil_image = Image.fromarray(image_np)
    pil_image.save(save_path, 'JPEG', quality=quality, optimize=True)
    
    # Report file size
    file_size = save_path.stat().st_size / 1024  # KB
    print(f"Saved JPEG to {save_path} ({file_size:.2f} KB, quality={quality})")
    
    return file_size


def save_image(image_tensor, save_path, quality=95):
    """Save a tensor image to file."""
    save_path = Path(save_path)
    
    if image_tensor.is_cuda:
        image_tensor = image_tensor.cpu()
    
    image_np = np.clip(image_tensor.numpy(), 0, 1)
    image_np = (image_np * 255).astype(np.uint8)
    
    if image_np.shape[2] == 1:
        image_np = image_np.squeeze(axis=2)
    
    pil_image = Image.fromarray(image_np)
    
    extension = save_path.suffix.lower()
    if extension in ['.jpg', '.jpeg']:
        pil_image.save(save_path, 'JPEG', quality=quality)
    else:
        pil_image.save(save_path, 'PNG')


def save_gaussians(gaussians, save_path, image_size=None, metadata=None):
    """Save Gaussian parameters to file."""
    save_path = Path(save_path)
    
    # Extract parameters from all Gaussians
    params = {
        'positions': [],
        'colors': [],
        'scales': [],
        'rotations': [],
        'n_channels': gaussians[0].c.shape[0] if gaussians else 3
    }
    
    for g in gaussians:
        params['positions'].append(g.mu.detach().cpu().numpy())
        params['colors'].append(g.c.detach().cpu().numpy())
        params['scales'].append(g.s.detach().cpu().numpy())
        params['rotations'].append(g.theta.detach().cpu().numpy())
    
    # Convert lists to numpy arrays
    params['positions'] = np.array(params['positions'], dtype=np.float16)
    params['colors'] = np.array(params['colors'], dtype=np.float16)
    params['scales'] = np.array(params['scales'], dtype=np.float16)
    params['rotations'] = np.array(params['rotations'], dtype=np.float16)
    
    # Add metadata
    if image_size:
        params['image_size'] = image_size
    if metadata:
        params['metadata'] = metadata
    
    # Save based on extension
    if save_path.suffix == '.npz':
        np.savez_compressed(save_path, **params)
    else:
        with open(save_path.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(params, f)
    
    # Calculate file size
    file_size = save_path.stat().st_size / 1024  # KB
    print(f"Saved {len(gaussians)} Gaussians to {save_path} ({file_size:.2f} KB)")
    
    return file_size


def load_gaussians(load_path, device='cuda'):
    """Load Gaussian parameters from file."""
    from gaussian_2d import Gaussian2D
    
    load_path = Path(load_path)
    
    # Load based on extension
    if load_path.suffix == '.npz':
        data = np.load(load_path, allow_pickle=True)
        params = dict(data)
        if 'metadata' in params and isinstance(params['metadata'], np.ndarray):
            params['metadata'] = params['metadata'].item()
    else:
        with open(load_path, 'rb') as f:
            params = pickle.load(f)
    
    # Reconstruct Gaussian objects
    gaussians = []
    n_gaussians = len(params['positions'])
    n_channels = params['n_channels']
    
    for i in range(n_gaussians):
        g = Gaussian2D(n_channels, device=device)
        g.mu.data = torch.tensor(params['positions'][i], device=device)
        g.c.data = torch.tensor(params['colors'][i], device=device)
        g.s.data = torch.tensor(params['scales'][i], device=device)
        g.theta.data = torch.tensor(params['rotations'][i], device=device)
        g.inv_s.data = 1.0 / (g.s.data + 1e-6)
        gaussians.append(g)
    
    image_size = params.get('image_size', None)
    metadata = params.get('metadata', {})
    
    print(f"Loaded {len(gaussians)} Gaussians from {load_path}")
    
    return gaussians, image_size, metadata


def compute_ssim(img1, img2, window_size=11, sigma=1.5):
    """Compute Structural Similarity Index between two images."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    device = img1.device
    coords = torch.arange(window_size, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g.unsqueeze(0) * g.unsqueeze(1)
    window = window.unsqueeze(0).unsqueeze(0)
    
    ssim_vals = []
    for c in range(img1.shape[2]):
        channel1 = img1[:, :, c].unsqueeze(0).unsqueeze(0)
        channel2 = img2[:, :, c].unsqueeze(0).unsqueeze(0)
        
        mu1 = F.conv2d(channel1, window, padding=window_size//2)
        mu2 = F.conv2d(channel2, window, padding=window_size//2)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(channel1 ** 2, window, padding=window_size//2) - mu1_sq
        sigma2_sq = F.conv2d(channel2 ** 2, window, padding=window_size//2) - mu2_sq
        sigma12 = F.conv2d(channel1 * channel2, window, padding=window_size//2) - mu1_mu2
        
        ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        ssim_vals.append(ssim.mean())
    
    return torch.stack(ssim_vals).mean()