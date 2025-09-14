import torch
import argparse
from pathlib import Path
from optimizer import ImageGS
from gaussian_renderer import GaussianRenderer
from utils import (load_image, save_image, compute_ssim, save_gaussians, 
                   load_gaussians, save_image_as_jpeg)


def compare_compression(image_path, n_gaussians=10000, n_steps=5000, 
                       jpeg_quality=95, output_dir="output"):
    """Compare Image-GS compression with JPEG."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load RAW image
    print(f"\nLoading RAW image: {image_path}")
    image = load_image(image_path)
    print(f"Image shape: {image.shape}")
    
    # Save original as PNG for reference
    original_png = output_dir / f"{Path(image_path).stem}_original.png"
    save_image(image, original_png)
    
    # === JPEG Compression ===
    print("\n=== JPEG Compression ===")
    jpeg_path = output_dir / f"{Path(image_path).stem}_compressed.jpg"
    jpeg_size = save_image_as_jpeg(image, jpeg_path, quality=jpeg_quality)
    
    # Load JPEG back and compute metrics
    jpeg_image = load_image(jpeg_path, device=image.device)
    jpeg_psnr = -10 * torch.log10(torch.mean((image - jpeg_image) ** 2))
    jpeg_ssim = compute_ssim(image, jpeg_image)
    
    # === Image-GS Compression ===
    print("\n=== Image-GS Compression ===")
    model = ImageGS(n_gaussians=n_gaussians)
    gaussians = model.fit(image, n_steps=n_steps)
    
    # Save Gaussians
    print("\nSaving gaussinans...")
    gaussian_path = output_dir / f"{Path(image_path).stem}_gaussians.npz"
    metadata = {
        'source_image': str(image_path),
        'n_gaussians': len(gaussians),
        'n_steps': n_steps
    }
    gs_size = save_gaussians(gaussians, gaussian_path, image_size=image.shape[:2], metadata=metadata)
    
    print("\n=== Image Reconstruction ===")
    # Reconstruct
    reconstructed = model.reconstruct(gaussians, image.shape[:2])
    gs_psnr = -10 * torch.log10(torch.mean((image - reconstructed) ** 2))
    gs_ssim = compute_ssim(image, reconstructed)
    
    # Save reconstructed image
    print("\nSaving reconstructed image...")
    gs_output_path = output_dir / f"{Path(image_path).stem}_reconstructed_gs.png"
    save_image(reconstructed, gs_output_path)
    
    # === Comparison Results ===
    print("\n" + "="*50)
    print("COMPRESSION COMPARISON RESULTS")
    print("="*50)
    
    pixels = image.shape[0] * image.shape[1]
    original_size = pixels * 3  # 3 bytes per pixel for uncompressed
    
    print(f"\nOriginal (uncompressed): {original_size/1024:.2f} KB")
    print(f"Image dimensions: {image.shape[0]}x{image.shape[1]}")
    
    print(f"\nJPEG (quality={jpeg_quality}):")
    print(f"  File size: {jpeg_size:.2f} KB")
    print(f"  Compression ratio: {original_size/1024/jpeg_size:.2f}x")
    print(f"  Bits per pixel: {jpeg_size*1024*8/pixels:.3f}")
    print(f"  PSNR: {jpeg_psnr:.2f} dB")
    print(f"  SSIM: {jpeg_ssim:.4f}")
    
    print(f"\nImage-GS ({len(gaussians)} Gaussians):")
    print(f"  File size: {gs_size:.2f} KB")
    print(f"  Compression ratio: {original_size/1024/gs_size:.2f}x")
    print(f"  Bits per pixel: {gs_size*1024*8/pixels:.3f}")
    print(f"  PSNR: {gs_psnr:.2f} dB")
    print(f"  SSIM: {gs_ssim:.4f}")
    
    print("\n" + "="*50)
    
    # Determine winner
    if gs_psnr > jpeg_psnr and gs_size < jpeg_size:
        print("Image-GS wins: Better quality AND smaller size!")
    elif gs_psnr > jpeg_psnr:
        print(f"Image-GS has better quality (+{gs_psnr-jpeg_psnr:.2f} dB)")
    elif gs_size < jpeg_size:
        print(f"Image-GS has smaller size (-{jpeg_size-gs_size:.2f} KB)")
    else:
        print("JPEG performs better in this case")

'''
Ex. python main.py <path/to/image/from/main> --gaussians 10000 --steps 5000 --jpeg-quality 100
'''
def main():
    parser = argparse.ArgumentParser(description="Compare Image-GS with JPEG compression")
    parser.add_argument("image_path", type=str, help="Path to RAW image (.ARW, .CR2, .NEF, etc.)")
    parser.add_argument("--gaussians", type=int, default=10000, help="Number of Gaussians")
    parser.add_argument("--steps", type=int, default=5000, help="Number of optimization steps")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality (1-100)")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    
    args = parser.parse_args()
    
    compare_compression(
        args.image_path,
        n_gaussians=args.gaussians,
        n_steps=args.steps,
        jpeg_quality=args.jpeg_quality,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()