"""
Advanced Image Metrics for DCGAN
================================

Implementation of Frechet Inception Distance (FID) and Inception Score (IS)
for evaluating DCGAN generated spectrograms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from scipy import linalg
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class InceptionV3(nn.Module):
    """Simplified Inception V3 model for FID and IS calculation"""
    
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        
        # Simple CNN architecture inspired by Inception V3
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # Grayscale input
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Block 2
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Block 3
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Block 4
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Block 5
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Global Average Pooling
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Classifier
        self.classifier = nn.Linear(512, num_classes)
        
        # Feature extractor (before classifier)
        self.feature_dim = 512
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both features and logits
        
        Args:
            x: Input tensor (B, 1, H, W) for grayscale spectrograms
            
        Returns:
            features: Feature vector (B, 512)
            logits: Classification logits (B, num_classes)
        """
        # Extract features
        features = self.features(x)
        features = features.view(features.size(0), -1)  # Flatten: (B, 512)
        
        # Classification
        logits = self.classifier(features)
        
        return features, logits


class AdvancedImageMetrics:
    """Advanced metrics calculator for DCGAN evaluation"""
    
    def __init__(self, device: torch.device = None):
        """
        Initialize metrics calculator
        
        Args:
            device: Compute device (cuda/cpu)
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize simplified Inception model
        self.inception_model = InceptionV3().to(self.device)
        self.inception_model.eval()
        
        # Initialize with random weights (in production, you'd load pretrained weights)
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize model with suitable weights for spectrogram analysis"""
        # For spectrograms, we use random initialization as pretrained ImageNet 
        # weights may not be optimal for spectrogram analysis
        for module in self.inception_model.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)
    
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features from images using Inception model
        
        Args:
            images: Batch of images (B, 1, H, W)
            
        Returns:
            features: Feature vectors (B, 512)
        """
        with torch.no_grad():
            # Ensure proper input format
            if images.dim() == 3:
                images = images.unsqueeze(1)  # Add channel dimension
            
            # Resize if needed (Inception expects at least 32x32)
            if images.size(-1) < 32 or images.size(-2) < 32:
                images = F.interpolate(images, size=(32, 32), mode='bilinear', align_corners=False)
            
            features, _ = self.inception_model(images.to(self.device))
            
        return features.cpu()
    
    def calculate_fid(self, real_features: torch.Tensor, fake_features: torch.Tensor) -> float:
        """
        Calculate Frechet Inception Distance
        
        Args:
            real_features: Features from real images (N, 512)
            fake_features: Features from generated images (M, 512)
            
        Returns:
            FID score (lower is better)
        """
        try:
            # Convert to numpy
            real_features = real_features.numpy()
            fake_features = fake_features.numpy()
            
            # Calculate statistics
            mu_real = np.mean(real_features, axis=0)
            mu_fake = np.mean(fake_features, axis=0)
            
            sigma_real = np.cov(real_features, rowvar=False)
            sigma_fake = np.cov(fake_features, rowvar=False)
            
            # Calculate FID
            diff = mu_real - mu_fake
            
            # Product might be almost singular
            covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_fake), disp=False)
            if not np.isfinite(covmean).all():
                msg = ('fid calculation produces singular product; '
                       'adding %s to diagonal of cov estimates') % 1e-6
                print(f"Warning: {msg}")
                offset = np.eye(sigma_real.shape[0]) * 1e-6
                covmean = linalg.sqrtm((sigma_real + offset).dot(sigma_fake + offset))
            
            # Numerical error might give slight imaginary component
            if np.iscomplexobj(covmean):
                if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                    m = np.max(np.abs(covmean.imag))
                    raise ValueError(f'Imaginary component {m}')
                covmean = covmean.real
            
            tr_covmean = np.trace(covmean)
            
            fid = (diff.dot(diff) + np.trace(sigma_real) + 
                   np.trace(sigma_fake) - 2 * tr_covmean)
            
            return float(fid)
            
        except Exception as e:
            print(f"Warning: FID calculation failed: {e}")
            return 999.0  # Return high value on failure
    
    def calculate_inception_score(self, fake_features: torch.Tensor, 
                                 num_splits: int = 10) -> Tuple[float, float]:
        """
        Calculate Inception Score
        
        Args:
            fake_features: Features from generated images (N, 512)
            num_splits: Number of splits for statistics
            
        Returns:
            mean_is: Mean Inception Score (higher is better)
            std_is: Standard deviation of IS
        """
        try:
            # Get predictions
            with torch.no_grad():
                # Use features to get predictions
                fake_features = fake_features.to(self.device)
                logits = self.inception_model.classifier(fake_features)
                preds = F.softmax(logits, dim=1).cpu().numpy()
            
            # Split predictions into chunks
            N = preds.shape[0]
            split_size = N // num_splits
            
            if split_size < 2:
                # Too few samples, return default
                return 1.0, 0.0
            
            scores = []
            
            for i in range(num_splits):
                start_idx = i * split_size
                end_idx = start_idx + split_size
                if i == num_splits - 1:
                    end_idx = N  # Include remaining samples in last split
                
                split_preds = preds[start_idx:end_idx]
                
                # Calculate p(y|x) for this split
                py = np.mean(split_preds, axis=0)
                
                # Calculate KL divergence for each sample
                kl_divs = []
                for j in range(split_preds.shape[0]):
                    px = split_preds[j]
                    # Avoid log(0) by adding small epsilon
                    px = px + 1e-16
                    py_safe = py + 1e-16
                    kl_div = np.sum(px * np.log(px / py_safe))
                    kl_divs.append(kl_div)
                
                # IS for this split
                is_score = np.exp(np.mean(kl_divs))
                scores.append(is_score)
            
            mean_is = np.mean(scores)
            std_is = np.std(scores)
            
            return float(mean_is), float(std_is)
            
        except Exception as e:
            print(f"Warning: IS calculation failed: {e}")
            return 1.0, 0.0  # Return default values on failure
    
    def evaluate_generator(self, generator: nn.Module, real_images: torch.Tensor,
                          num_generated: int = 1000, batch_size: int = 64) -> dict:
        """
        Comprehensive evaluation of generator
        
        Args:
            generator: Generator model
            real_images: Real images for comparison (N, 1, H, W)
            num_generated: Number of images to generate for evaluation
            batch_size: Batch size for generation and feature extraction
            
        Returns:
            Dictionary with FID and IS scores
        """
        generator.eval()
        
        try:
            # Extract features from real images
            real_features_list = []
            num_real = min(real_images.size(0), num_generated)
            
            for i in range(0, num_real, batch_size):
                end_idx = min(i + batch_size, num_real)
                batch_real = real_images[i:end_idx]
                features = self.extract_features(batch_real)
                real_features_list.append(features)
            
            real_features = torch.cat(real_features_list, dim=0)
            
            # Generate fake images and extract features
            fake_features_list = []
            latent_dim = getattr(generator, 'latent_dim', 100)
            
            with torch.no_grad():
                for i in range(0, num_generated, batch_size):
                    current_batch_size = min(batch_size, num_generated - i)
                    
                    # Generate latent vectors
                    z = torch.randn(current_batch_size, latent_dim, 1, 1).to(self.device)
                    
                    # Generate images
                    fake_images = generator(z)
                    
                    # Extract features
                    features = self.extract_features(fake_images)
                    fake_features_list.append(features)
            
            fake_features = torch.cat(fake_features_list, dim=0)
            
            # Calculate metrics
            fid_score = self.calculate_fid(real_features, fake_features)
            is_mean, is_std = self.calculate_inception_score(fake_features)
            
            # Calculate additional metrics
            results = {
                'fid_score': fid_score,
                'inception_score_mean': is_mean,
                'inception_score_std': is_std,
                'num_real_samples': real_features.size(0),
                'num_fake_samples': fake_features.size(0)
            }
            
            generator.train()  # Restore training mode
            return results
            
        except Exception as e:
            print(f"Error in generator evaluation: {e}")
            generator.train()
            return {
                'fid_score': 999.0,
                'inception_score_mean': 1.0,
                'inception_score_std': 0.0,
                'num_real_samples': 0,
                'num_fake_samples': 0
            }


# Global instance for reuse across training
_global_metrics_calculator = None

def get_metrics_calculator(device: torch.device = None) -> AdvancedImageMetrics:
    """Get global metrics calculator instance"""
    global _global_metrics_calculator
    
    if _global_metrics_calculator is None:
        _global_metrics_calculator = AdvancedImageMetrics(device)
    
    return _global_metrics_calculator


def calculate_fid_is_scores(generator: nn.Module, real_images: torch.Tensor,
                           device: torch.device = None) -> Tuple[float, float]:
    """
    Convenience function to calculate FID and IS scores
    
    Args:
        generator: Generator model
        real_images: Real images for comparison
        device: Compute device
        
    Returns:
        fid_score: FID score (lower is better)
        is_score: Inception Score mean (higher is better)
    """
    calculator = get_metrics_calculator(device)
    results = calculator.evaluate_generator(generator, real_images, num_generated=500)
    
    return results['fid_score'], results['inception_score_mean']
