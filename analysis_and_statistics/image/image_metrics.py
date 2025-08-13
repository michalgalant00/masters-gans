"""
Image Metrics Collector
=======================

DCGAN-specific metrics collection including image quality metrics.
"""

import os
import numpy as np
import torch
from PIL import Image
from typing import Dict, List, Optional, Any, Tuple
import matplotlib.pyplot as plt

# Import base class
from ..base.metrics_collector import MetricsCollectorBase


class ImageMetricsCollector(MetricsCollectorBase):
    """Metrics collector for image-based GANs (DCGAN)"""
    
    def __init__(self, output_dir: str = "output_analysis", config: Optional[Dict] = None):
        super().__init__(output_dir, "image", config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Image-specific tracking
        self.generated_samples_count = 0
        self.last_fid_score = None
        self.last_inception_score = None
    
    def _get_training_headers(self) -> List[str]:
        """Get DCGAN-specific training log headers"""
        base_headers = self._get_base_training_headers()
        
        # Add image-specific headers
        image_headers = [
            "d_real_score", "d_fake_score",  # Discriminator scores
            "image_size",  # Image dimensions
        ]
        
        return base_headers + image_headers
    
    def _get_epoch_headers(self) -> List[str]:
        """Get DCGAN-specific epoch summary headers"""
        base_headers = self._get_base_epoch_headers()
        
        # Add image-specific headers
        image_headers = [
            "avg_d_real_score", "avg_d_fake_score",  # Average discriminator scores
            "d_real_accuracy", "d_fake_accuracy",    # Discriminator accuracies
        ]
        
        return base_headers + image_headers
    
    def _get_checkpoint_headers(self) -> List[str]:
        """Get DCGAN-specific checkpoint metrics headers per metrics-checkpoints-how-to.txt"""
        base_headers = self._get_base_checkpoint_headers()
        
        # Add DCGAN-specific headers according to requirements:
        # "stoi_score, pesq_score, mcd_score" -> adapted for images
        # "audio_sample_paths" -> image_sample_paths  
        # "checkpoint_file_path"
        # "model_performance_score"
        image_headers = [
            "fid_score", "inception_score", "image_quality_score",  # Image quality metrics
            "stoi_score", "pesq_score", "mcd_score",  # Audio metrics (N/A for DCGAN, set to 0)
            "image_sample_paths",  # Sample file paths
            "checkpoint_file_path",  # Path to checkpoint file
            "model_performance_score"  # Overall performance metric
        ]
        
        return base_headers + image_headers
    
    def log_iteration_stats(self, epoch: int, iteration: int,
                          generator_loss: float, discriminator_loss: float,
                          generator_grad_norm: float, discriminator_grad_norm: float,
                          learning_rate_g: float, learning_rate_d: float,
                          batch_size: int, iteration_time: float,
                          d_real_score: float, d_fake_score: float,
                          image_size: int, **kwargs):
        """Log DCGAN iteration statistics using optimized snapshot method"""
        
        # Map to optimized structure - call log_iteration_snapshot
        losses = {'generator_loss': generator_loss, 'discriminator_loss': discriminator_loss}
        grads = {'generator_grad_norm': generator_grad_norm, 'discriminator_grad_norm': discriminator_grad_norm}
        gpu_stats = {'gpu_memory_used': 0.0, 'iteration_time_seconds': iteration_time}
        
        # Use optimized snapshot method according to metrics-checkpoints-how-to.txt
        self.log_iteration_snapshot(epoch, iteration, losses, grads, gpu_stats)
    
    def _get_domain_epoch_data(self, domain_kwargs: Dict) -> List:
        """Get DCGAN-specific epoch summary data"""
        
        # Get domain-specific metrics from kwargs
        avg_d_real = domain_kwargs.get('avg_d_real_score', 0.0)
        avg_d_fake = domain_kwargs.get('avg_d_fake_score', 0.0)
        d_real_accuracy = domain_kwargs.get('d_real_accuracy', 0.0)
        d_fake_accuracy = domain_kwargs.get('d_fake_accuracy', 0.0)
        
        return [avg_d_real, avg_d_fake, d_real_accuracy, d_fake_accuracy]
    
    def calculate_fid_score(self, real_images: torch.Tensor, 
                           fake_images: torch.Tensor) -> float:
        """Calculate Fréchet Inception Distance (FID) score
        
        Args:
            real_images: Tensor of real images
            fake_images: Tensor of generated images
            
        Returns:
            FID score (lower is better)
        """
        try:
            # This is a simplified placeholder implementation
            # In practice, you would use a proper FID implementation
            # that uses InceptionV3 features
            
            # Convert to numpy arrays
            real_np = real_images.detach().cpu().numpy()
            fake_np = fake_images.detach().cpu().numpy()
            
            # Flatten images for simple comparison
            real_flat = real_np.reshape(real_np.shape[0], -1)
            fake_flat = fake_np.reshape(fake_np.shape[0], -1)
            
            # Calculate means and covariances
            mu_real = np.mean(real_flat, axis=0)
            mu_fake = np.mean(fake_flat, axis=0)
            
            # Check if we have enough samples for covariance calculation
            min_samples_needed = real_flat.shape[1] + 1  # Need at least n_features + 1 samples
            
            if real_flat.shape[0] < min_samples_needed or fake_flat.shape[0] < min_samples_needed:
                # Not enough samples for proper covariance - use simplified distance
                diff = mu_real - mu_fake
                fid_score = np.sum(diff ** 2)
                print(f"ℹ️  FID: Using simplified calculation (insufficient samples: real={real_flat.shape[0]}, fake={fake_flat.shape[0]}, need={min_samples_needed})")
            else:
                # Full FID calculation with covariance
                with np.errstate(divide='ignore', invalid='ignore'):
                    sigma_real = np.cov(real_flat, rowvar=False)
                    sigma_fake = np.cov(fake_flat, rowvar=False)
                    
                    # Ensure matrices are positive definite
                    sigma_real = sigma_real + np.eye(sigma_real.shape[0]) * 1e-6
                    sigma_fake = sigma_fake + np.eye(sigma_fake.shape[0]) * 1e-6
                    
                    # Simple FID-like calculation
                    diff = mu_real - mu_fake
                    
                    try:
                        sqrt_product = np.sqrt(sigma_real @ sigma_fake)
                        fid_score = np.sum(diff ** 2) + np.trace(sigma_real + sigma_fake - 2 * sqrt_product)
                    except:
                        # Fallback to simplified calculation
                        fid_score = np.sum(diff ** 2) + np.trace(sigma_real + sigma_fake)
            
            self.last_fid_score = float(fid_score)
            return self.last_fid_score
            
        except Exception as e:
            print(f"⚠️  Error calculating FID score: {e}")
            return 999.0  # High value indicates poor quality
    
    def calculate_inception_score(self, fake_images: torch.Tensor) -> Tuple[float, float]:
        """Calculate Inception Score (IS)
        
        Args:
            fake_images: Tensor of generated images
            
        Returns:
            Tuple of (mean_score, std_score)
        """
        try:
            # This is a simplified placeholder implementation
            # In practice, you would use a proper IS implementation
            # that uses InceptionV3 predictions
            
            # For now, return a mock score based on image statistics
            fake_np = fake_images.detach().cpu().numpy()
            
            # Calculate some basic image statistics as proxy
            mean_intensity = np.mean(fake_np)
            std_intensity = np.std(fake_np)
            
            # Mock IS calculation (higher is better, typically 1-10+ range)
            mock_is = max(1.0, min(10.0, float(5.0 + std_intensity - abs(mean_intensity - 0.5))))
            
            self.last_inception_score = mock_is
            return mock_is, 0.1  # (mean, std)
            
        except Exception as e:
            print(f"⚠️  Error calculating Inception Score: {e}")
            return 1.0, 0.0  # Low score indicates poor quality
    
    def save_image_samples(self, fake_images: torch.Tensor, 
                          sample_dir: str, prefix: str = "sample") -> List[str]:
        """Save generated image samples
        
        Args:
            fake_images: Tensor of generated images
            sample_dir: Directory to save samples
            prefix: Filename prefix
            
        Returns:
            List of saved file paths
        """
        try:
            os.makedirs(sample_dir, exist_ok=True)
            saved_paths = []
            
            # Convert tensor to numpy and denormalize if needed
            images_np = fake_images.detach().cpu().numpy()
            
            # Handle different tensor formats
            if images_np.shape[1] == 3:  # CHW format (channels first)
                images_np = np.transpose(images_np, (0, 2, 3, 1))  # Convert to HWC
            
            # Normalize to [0, 255] range
            if images_np.max() <= 1.0:
                images_np = (images_np * 255).astype(np.uint8)
            else:
                images_np = np.clip(images_np, 0, 255).astype(np.uint8)
            
            # Save individual images
            for i, img_np in enumerate(images_np):
                if len(img_np.shape) == 3 and img_np.shape[2] == 1:  # Grayscale with channel dim
                    img_np = img_np.squeeze(axis=2)  # Remove channel dimension
                    mode = 'L'
                elif len(img_np.shape) == 3 and img_np.shape[2] == 3:  # RGB
                    mode = 'RGB'
                elif len(img_np.shape) == 2:  # Grayscale without channel dimension
                    mode = 'L'
                else:  # Handle unexpected dimensions
                    # Try to squeeze all single dimensions
                    img_np = img_np.squeeze()
                    if len(img_np.shape) == 2:
                        mode = 'L'
                    elif len(img_np.shape) == 3 and img_np.shape[2] == 3:
                        mode = 'RGB'
                    else:
                        print(f"⚠️  Unexpected image shape: {img_np.shape}, skipping sample {i}")
                        continue
                
                img = Image.fromarray(img_np, mode=mode)
                filename = f"{prefix}_{self.generated_samples_count + i}.png"
                file_path = os.path.join(sample_dir, filename)
                img.save(file_path)
                saved_paths.append(file_path)
            
            self.generated_samples_count += len(images_np)
            return saved_paths
            
        except Exception as e:
            print(f"⚠️  Error saving image samples: {e}")
            return []
    
    def create_sample_grid(self, fake_images: torch.Tensor, 
                          grid_path: str, nrow: int = 8) -> str:
        """Create and save a grid of generated samples
        
        Args:
            fake_images: Tensor of generated images
            grid_path: Path to save grid image
            nrow: Number of images per row
            
        Returns:
            Path to saved grid image
        """
        try:
            try:
                import torchvision.utils as vutils
                
                # Create directory if needed
                os.makedirs(os.path.dirname(grid_path), exist_ok=True)
                
                # Create grid
                grid = vutils.make_grid(fake_images, nrow=nrow, normalize=True, padding=2)
                
                # Save grid
                vutils.save_image(grid, grid_path)
                
                return grid_path
                
            except ImportError:
                print("⚠️  torchvision not available, using fallback grid creation")
                # Continue to fallback implementation
                pass
            
            # Fallback: create grid manually
            images_np = fake_images.detach().cpu().numpy()
            
            # Simple grid creation
            n_images = min(len(images_np), nrow * nrow)
            grid_size = int(np.ceil(np.sqrt(n_images)))
            
            if len(images_np.shape) == 4:  # NCHW format
                _, C, H, W = images_np.shape
                grid_img = np.zeros((C, H * grid_size, W * grid_size))
                
                for i in range(n_images):
                    row = i // grid_size
                    col = i % grid_size
                    grid_img[:, row*H:(row+1)*H, col*W:(col+1)*W] = images_np[i]
                
                # Convert to PIL and save
                if C == 1:
                    grid_img = grid_img.squeeze()
                    img = Image.fromarray((grid_img * 255).astype(np.uint8), mode='L')
                else:
                    grid_img = np.transpose(grid_img, (1, 2, 0))
                    img = Image.fromarray((grid_img * 255).astype(np.uint8), mode='RGB')
                
                img.save(grid_path)
                return grid_path
            
            return ""
                
        except Exception as e:
            print(f"⚠️  Error creating sample grid: {e}")
            return ""
    
    def log_checkpoint_metrics(self, checkpoint_type: str, epoch: int, iteration: int,
                             generator, discriminator, real_samples: Optional[torch.Tensor] = None,
                             checkpoint_file_path: str = ""):
        """Log metrics for a checkpoint with image quality assessment
        
        Args:
            checkpoint_type: Type of checkpoint
            epoch: Current epoch
            iteration: Current iteration
            generator: Generator model
            discriminator: Discriminator model
            real_samples: Real image samples for comparison
            checkpoint_file_path: Path to saved checkpoint file
        """
        try:
            import csv
            from datetime import datetime
            
            timestamp = datetime.now().isoformat()
            
            # Generate fake samples for evaluation
            generator.eval()
            with torch.no_grad():
                latent_dim = generator.latent_dim if hasattr(generator, 'latent_dim') else 100
                noise = torch.randn(16, latent_dim, 1, 1)
                if torch.cuda.is_available():
                    noise = noise.cuda()
                fake_samples = generator(noise)
            generator.train()
            
            # Calculate image quality metrics
            fid_score = 999.0  # Default high value
            inception_score = 1.0  # Default low value
            
            if real_samples is not None:
                fid_score = self.calculate_fid_score(real_samples, fake_samples)
            
            inception_score, _ = self.calculate_inception_score(fake_samples)
            
            # Save sample images
            sample_dir = os.path.join(self.output_dir, "samples", checkpoint_type)
            sample_paths = self.save_image_samples(
                fake_samples, sample_dir, 
                f"epoch_{epoch}_{checkpoint_type}"
            )
            
            # Create sample grid
            grid_path = os.path.join(sample_dir, f"grid_epoch_{epoch}_{checkpoint_type}.png")
            self.create_sample_grid(fake_samples, grid_path)
            
            # Calculate model parameters
            gen_params = sum(p.numel() for p in generator.parameters())
            disc_params = sum(p.numel() for p in discriminator.parameters())
            
            # Calculate overall image quality score (composite metric)
            image_quality_score = self._calculate_image_quality_score(fid_score, inception_score)
            
            # Calculate model performance score
            model_performance_score = self._calculate_model_performance_score(
                gen_params, disc_params, fid_score, inception_score
            )
            
            # Prepare checkpoint metrics row per metrics-checkpoints-how-to.txt
            checkpoint_row = [
                checkpoint_type, epoch, iteration, timestamp,
                gen_params, disc_params,  # generator_params_count, discriminator_params_count
                fid_score, inception_score, image_quality_score,  # Image quality metrics
                0.0, 0.0, 0.0,  # stoi_score, pesq_score, mcd_score (N/A for DCGAN)
                str(sample_paths),  # image_sample_paths
                checkpoint_file_path,  # checkpoint_file_path
                model_performance_score  # model_performance_score
            ]
            
            # Write to checkpoint metrics CSV
            with open(self.checkpoint_metrics_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(checkpoint_row)
            
            print(f"📊 Checkpoint metrics logged - FID: {fid_score:.2f}, IS: {inception_score:.2f}")
            
        except Exception as e:
            print(f"❌ Error logging checkpoint metrics: {e}")
    
    def _calculate_image_quality_score(self, fid_score: float, inception_score: float) -> float:
        """Calculate composite image quality score
        
        Args:
            fid_score: FID score (lower is better)
            inception_score: Inception score (higher is better)
            
        Returns:
            Composite quality score (0-100, higher is better)
        """
        # Normalize FID score (typical range 0-300, lower is better)
        fid_normalized = max(0, min(100, 100 - (fid_score / 3)))
        
        # Normalize IS score (typical range 1-10, higher is better)
        is_normalized = max(0, min(100, (inception_score - 1) * 10))
        
        # Weighted combination (FID is generally more reliable)
        quality_score = 0.7 * fid_normalized + 0.3 * is_normalized
        
        return float(quality_score)
    
    def _calculate_model_performance_score(self, gen_params: int, disc_params: int, 
                                         fid_score: float, inception_score: float) -> float:
        """Calculate overall model performance score for checkpoints
        
        Args:
            gen_params: Number of generator parameters
            disc_params: Number of discriminator parameters
            fid_score: FID score (lower is better)
            inception_score: Inception score (higher is better)
            
        Returns:
            Performance score (0-100, higher is better)
        """
        # Image quality component (70% weight)
        quality_score = self._calculate_image_quality_score(fid_score, inception_score)
        
        # Model efficiency component (30% weight)
        total_params = gen_params + disc_params
        # Penalize very large models (>50M params), reward efficient models
        if total_params < 1_000_000:  # <1M params
            efficiency_score = 100.0
        elif total_params < 10_000_000:  # 1-10M params
            efficiency_score = 80.0
        elif total_params < 50_000_000:  # 10-50M params
            efficiency_score = 60.0
        else:  # >50M params
            efficiency_score = 40.0
        
        # Weighted combination
        performance_score = 0.7 * quality_score + 0.3 * efficiency_score
        
        return float(performance_score)
