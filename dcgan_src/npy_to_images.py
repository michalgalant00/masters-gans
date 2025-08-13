#!/usr/bin/env python3
"""
NumPy to Images Converter for DCGAN Samples
===========================================

This script converts saved .npy sample files to image formats (PNG, JPG, etc.).
The .npy files contain normalized arrays with shape (batch, channels, height, width).

Usage:
    python npy_to_images.py input.npy [options]
    python npy_to_images.py --batch-convert samples_dir/ [options]

Examples:
    # Convert single file to individual images
    python npy_to_images.py epoch_001.npy
    
    # Convert single file to grid image
    python npy_to_images.py epoch_001.npy --mode grid --output epoch_001_grid.png
    
    # Convert all .npy files in directory
    python npy_to_images.py --batch-convert output_dcgan/samples/
    
    # Custom output format and quality
    python npy_to_images.py epoch_001.npy --format jpeg --quality 95
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import glob
from typing import List, Tuple, Optional


def load_samples(npy_path: str) -> np.ndarray:
    """Load samples from .npy file"""
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"File not found: {npy_path}")
    
    samples = np.load(npy_path)
    print(f"📁 Loaded: {npy_path}")
    print(f"   Shape: {samples.shape}")
    print(f"   Value range: [{samples.min():.3f}, {samples.max():.3f}]")
    
    return samples


def save_individual_images(samples: np.ndarray, output_dir: str, 
                          base_name: str, format: str = 'png', 
                          quality: int = 95) -> List[str]:
    """Save each sample as individual image file"""
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    
    batch_size = samples.shape[0]
    
    for i in range(batch_size):
        sample = samples[i]
        
        # Handle different channel configurations
        if sample.shape[0] == 1:  # Grayscale (1, H, W)
            image_array = sample[0]  # Remove channel dimension
        elif sample.shape[0] == 3:  # RGB (3, H, W)
            image_array = np.transpose(sample, (1, 2, 0))  # Convert to (H, W, 3)
        else:
            raise ValueError(f"Unsupported channel count: {sample.shape[0]}")
        
        # Convert to 0-255 range
        image_array = (image_array * 255).astype(np.uint8)
        
        # Create PIL Image (mode auto-detected by PIL)
        pil_image = Image.fromarray(image_array)
        
        # Save image
        filename = f"{base_name}_sample_{i+1:02d}.{format.lower()}"
        filepath = os.path.join(output_dir, filename)
        
        if format.lower() == 'jpeg' or format.lower() == 'jpg':
            pil_image.save(filepath, quality=quality, optimize=True)
        else:
            pil_image.save(filepath)
        
        saved_paths.append(filepath)
        print(f"💾 Saved: {filename}")
    
    return saved_paths


def save_grid_image(samples: np.ndarray, output_path: str, 
                   nrow: Optional[int] = None, format: str = 'png', 
                   quality: int = 95) -> str:
    """Save samples as a grid image"""
    batch_size = samples.shape[0]
    
    # Calculate grid dimensions
    if nrow is None:
        nrow = int(np.ceil(np.sqrt(batch_size)))
    ncol = int(np.ceil(batch_size / nrow))
    
    # Get image dimensions
    if samples.shape[1] == 1:  # Grayscale
        h, w = samples.shape[2], samples.shape[3]
        channels = 1
    else:  # RGB
        h, w = samples.shape[2], samples.shape[3]
        channels = 3
    
    # Create grid array
    if channels == 1:
        grid = np.zeros((nrow * h, ncol * w), dtype=np.float32)
    else:
        grid = np.zeros((nrow * h, ncol * w, channels), dtype=np.float32)
    
    # Fill grid
    for i in range(batch_size):
        row = i // ncol
        col = i % ncol
        
        if row >= nrow:
            break
            
        y_start, y_end = row * h, (row + 1) * h
        x_start, x_end = col * w, (col + 1) * w
        
        sample = samples[i]
        if channels == 1:
            grid[y_start:y_end, x_start:x_end] = sample[0]
        else:
            grid[y_start:y_end, x_start:x_end] = np.transpose(sample, (1, 2, 0))
    
    # Convert to 0-255 range
    grid = (grid * 255).astype(np.uint8)
    
    # Create PIL Image (mode auto-detected by PIL)
    pil_image = Image.fromarray(grid)
    
    # Save image
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    if format.lower() == 'jpeg' or format.lower() == 'jpg':
        pil_image.save(output_path, quality=quality, optimize=True)
    else:
        pil_image.save(output_path)
    
    print(f"🎨 Grid saved: {output_path}")
    print(f"   Grid size: {nrow}x{ncol} ({batch_size} samples)")
    
    return output_path


def batch_convert(input_dir: str, **kwargs):
    """Convert all .npy files in directory"""
    npy_files = glob.glob(os.path.join(input_dir, "*.npy"))
    
    if not npy_files:
        print(f"❌ No .npy files found in: {input_dir}")
        return
    
    print(f"📂 Found {len(npy_files)} .npy files in: {input_dir}")
    
    for npy_file in sorted(npy_files):
        print(f"\n🔄 Processing: {os.path.basename(npy_file)}")
        try:
            convert_single_file(npy_file, **kwargs)
        except Exception as e:
            print(f"❌ Error processing {npy_file}: {e}")


def convert_single_file(npy_path: str, mode: str = 'individual', 
                       output: Optional[str] = None, format: str = 'png',
                       quality: int = 95, nrow: Optional[int] = None):
    """Convert single .npy file"""
    samples = load_samples(npy_path)
    
    base_name = os.path.splitext(os.path.basename(npy_path))[0]
    input_dir = os.path.dirname(npy_path)
    
    if mode == 'individual':
        if output is None:
            output_dir = os.path.join(input_dir, f"{base_name}_images")
        else:
            output_dir = output
        
        saved_paths = save_individual_images(
            samples, output_dir, base_name, format, quality
        )
        print(f"✅ Saved {len(saved_paths)} individual images to: {output_dir}")
        
    elif mode == 'grid':
        if output is None:
            output_path = os.path.join(input_dir, f"{base_name}_grid.{format}")
        else:
            output_path = output
        
        save_grid_image(samples, output_path, nrow, format, quality)
        print(f"✅ Saved grid image: {output_path}")
        
    elif mode == 'both':
        # Save individual images
        output_dir = os.path.join(input_dir, f"{base_name}_images")
        saved_paths = save_individual_images(
            samples, output_dir, base_name, format, quality
        )
        
        # Save grid image
        grid_path = os.path.join(input_dir, f"{base_name}_grid.{format}")
        save_grid_image(samples, grid_path, nrow, format, quality)
        
        print(f"✅ Saved {len(saved_paths)} individual images and 1 grid image")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Convert DCGAN .npy sample files to images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Input
    parser.add_argument('input', nargs='?', 
                       help='Input .npy file or directory (for batch convert)')
    parser.add_argument('--batch-convert', type=str, metavar='DIR',
                       help='Convert all .npy files in directory')
    
    # Mode
    parser.add_argument('--mode', choices=['individual', 'grid', 'both'], 
                       default='individual',
                       help='Output mode: individual files, grid image, or both')
    
    # Output
    parser.add_argument('--output', '-o', type=str,
                       help='Output path (file for grid mode, directory for individual mode)')
    
    # Format options
    parser.add_argument('--format', choices=['png', 'jpg', 'jpeg', 'bmp', 'tiff'], 
                       default='png',
                       help='Output image format')
    parser.add_argument('--quality', type=int, default=95, 
                       help='JPEG quality (0-100)')
    
    # Grid options
    parser.add_argument('--nrow', type=int,
                       help='Number of rows in grid (auto-calculated if not specified)')
    
    args = parser.parse_args()
    
    # Validate input
    if not args.input and not args.batch_convert:
        parser.error("Must specify either input file or --batch-convert directory")
    
    if args.input and args.batch_convert:
        parser.error("Cannot specify both input file and --batch-convert directory")
    
    try:
        if args.batch_convert:
            batch_convert(
                args.batch_convert,
                mode=args.mode,
                output=args.output,
                format=args.format,
                quality=args.quality,
                nrow=args.nrow
            )
        else:
            convert_single_file(
                args.input,
                mode=args.mode,
                output=args.output,
                format=args.format,
                quality=args.quality,
                nrow=args.nrow
            )
            
        print("\n🎉 Conversion completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
