"""
Image-Specific Checkpoint Manager
=================================

DCGAN-specific extensions for checkpoint management including
image sample generation and archiving.
"""

import os
import shutil
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Optional, Dict, Any
from ..base.checkpoint_manager import CheckpointManagerBase


class ImageCheckpointManager(CheckpointManagerBase):
    """Checkpoint manager with image-specific extensions for DCGAN"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        super().__init__(checkpoint_dir, model_type="image")
    
    def _generate_samples(self, generator: torch.nn.Module, 
                         device: torch.device, latent_dim: int,
                         num_samples: Optional[int] = None) -> List[str]:
        """Generate and save image samples for checkpoint"""
        if num_samples is None:
            # Use default value instead of importing from config
            num_samples = 5  # Default fallback - consistent with current usage
        
        # Ensure num_samples is not None
        if num_samples is None:
            num_samples = 9
            
        generator.eval()
        sample_paths = []
        
        with torch.no_grad():
            # Generate all samples in batch for numpy saving
            z_batch = torch.randn(num_samples, latent_dim, 1, 1).to(device)
            images_batch = generator(z_batch)
            
            # Save batch as numpy array (.npy format) - use samples directory instead of temp
            npy_filename = f"samples_batch_{num_samples}.npy"
            npy_filepath = os.path.join(os.path.dirname(self.checkpoint_dir), "samples", npy_filename)
            os.makedirs(os.path.dirname(npy_filepath), exist_ok=True)
            
            # Convert to numpy and save
            images_np = images_batch.cpu().detach().numpy()
            # Denormalize from [-1, 1] to [0, 1]
            images_np = (images_np + 1.0) / 2.0
            images_np = np.clip(images_np, 0.0, 1.0)
            np.save(npy_filepath, images_np)
            sample_paths.append(npy_filepath)
            
            # Also save individual PNG files for backward compatibility
            for i in range(num_samples):
                image = images_batch[i:i+1]  # Keep batch dimension
                
                # Denormalize from [-1, 1] to [0, 1]
                image = (image + 1) / 2.0
                image = torch.clamp(image, 0, 1)
                
                # Convert to PIL Image
                if image.shape[1] == 1:  # Grayscale
                    image_np = image.squeeze().cpu().numpy()
                    pil_image = Image.fromarray((image_np * 255).astype(np.uint8), mode='L')
                else:  # RGB
                    image_np = image.squeeze().permute(1, 2, 0).cpu().numpy()
                    pil_image = Image.fromarray((image_np * 255).astype(np.uint8), mode='RGB')
                
                # Save image - use samples directory instead of temp
                filename = f"sample_{i+1}.png"
                filepath = os.path.join(os.path.dirname(self.checkpoint_dir), "samples", filename)
                
                pil_image.save(filepath)
                sample_paths.append(filepath)
        
        # Also create a grid image
        grid_path = self._create_sample_grid(sample_paths)
        if grid_path:
            sample_paths.append(grid_path)
        
        generator.train()
        return sample_paths
    
    def _create_sample_grid(self, sample_paths: List[str]) -> Optional[str]:
        """Create a grid image from individual samples"""
        if not sample_paths:
            return None
        
        try:
            # Load images
            images = []
            for path in sample_paths:
                if os.path.exists(path):
                    img = Image.open(path)
                    images.append(img)
            
            if not images:
                return None
            
            # Calculate grid size
            n = len(images)
            grid_size = int(np.ceil(np.sqrt(n)))
            
            # Get image size
            img_width, img_height = images[0].size
            
            # Create grid
            grid_width = grid_size * img_width
            grid_height = grid_size * img_height
            grid_image = Image.new('RGB', (grid_width, grid_height), color='white')
            
            for i, img in enumerate(images):
                row = i // grid_size
                col = i % grid_size
                x = col * img_width
                y = row * img_height
                grid_image.paste(img, (x, y))
            
            # Save grid - use samples directory instead of temp
            grid_path = os.path.join(os.path.dirname(self.checkpoint_dir), "samples", "sample_grid.png")
            grid_image.save(grid_path)
            return grid_path
            
        except Exception as e:
            print(f"⚠️ Failed to create sample grid: {e}")
            return None
    
    def _copy_samples_to_archive(self, sample_paths: List[str], temp_dir: str):
        """Copy image samples to archive directory"""
        for sample_path in sample_paths:
            if os.path.exists(sample_path):
                filename = os.path.basename(sample_path)
                shutil.copy2(sample_path, os.path.join(temp_dir, filename))
    
    def _get_domain_specific_checkpoint_data(self, **kwargs) -> Dict[str, Any]:
        """Get image-specific checkpoint data"""
        return {
            'image_size': kwargs.get('image_size', 64),
            'num_channels': kwargs.get('num_channels', 3)
        }
    
    def _generate_checkpoint_samples(self, generator, epoch: int, iteration: int,
                                   checkpoint_type: str, output_dir: Optional[str] = None) -> List[str]:
        """Generate 5 sample image files for checkpoint per metrics-checkpoints-how-to.txt
        
        Args:
            generator: Generator model
            epoch: Current epoch
            iteration: Current iteration  
            checkpoint_type: Type of checkpoint
            output_dir: Directory to save samples (if None, uses default)
            
        Returns:
            List of paths to generated sample files
        """
        if output_dir is None:
            output_dir = os.path.join(self.checkpoint_dir, "samples")
        
        os.makedirs(output_dir, exist_ok=True)
        
        sample_paths = []
        generator.eval()
        
        try:
            # Get device and latent_dim from generator
            device = next(generator.parameters()).device
            latent_dim = getattr(generator, 'latent_dim', 100)  # Default to 100
            
            with torch.no_grad():
                for i in range(5):  # Generate exactly 5 samples as per requirements
                    # Generate sample
                    z = torch.randn(1, latent_dim, 1, 1).to(device)
                    image = generator(z)
                    
                    # Denormalize from [-1, 1] to [0, 1]
                    image = (image + 1) / 2.0
                    image = torch.clamp(image, 0, 1)
                    
                    # Convert to numpy for PIL
                    image_np = image.cpu().squeeze().numpy()
                    
                    # Handle grayscale vs RGB
                    if image_np.ndim == 3 and image_np.shape[0] == 1:
                        image_np = image_np.squeeze(0)  # Remove channel dimension for grayscale
                    elif image_np.ndim == 3:
                        image_np = np.transpose(image_np, (1, 2, 0))  # CHW to HWC for RGB
                    
                    # Convert to PIL and save
                    image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
                    sample_path = os.path.join(output_dir, f"sample_{i+1}_epoch_{epoch}_{checkpoint_type}.png")
                    image_pil.save(sample_path)
                    sample_paths.append(sample_path)
                    
            print(f"🎨 Generated {len(sample_paths)} checkpoint samples in {output_dir}")
            
        except Exception as e:
            print(f"⚠️ Failed to generate checkpoint samples: {e}")
            # Return empty paths if generation fails
            sample_paths = []
        
        finally:
            generator.train()  # Restore training mode
            
        return sample_paths
