"""
Audio Metrics Collector
=======================

WaveGAN-specific metrics collection including audio quality metrics.
"""

import os
import numpy as np
import torch
from typing import Dict, List, Optional, Any, Tuple
import soundfile as sf

# Import base class
from ..base.metrics_collector import MetricsCollectorBase


class AudioMetricsCollector(MetricsCollectorBase):
    """Metrics collector for audio-based GANs (WaveGAN)"""
    
    def __init__(self, output_dir: str = "output_wavegan", config: Optional[Dict] = None):
        super().__init__(output_dir, model_type="audio", config=config)
        
        # Audio-specific tracking
        self.generated_samples_count = 0
        self.last_fad_score = None
        self.last_snr_score = None
    
    def _get_training_headers(self) -> List[str]:
        """Get WaveGAN-specific training log headers"""
        base_headers = self._get_base_training_headers()
        
        # Add audio-specific headers
        audio_headers = [
            "d_real_score", "d_fake_score",  # Discriminator scores
            "sample_rate", "audio_length",   # Audio characteristics
        ]
        
        return base_headers + audio_headers
    
    def _get_epoch_headers(self) -> List[str]:
        """Get WaveGAN-specific epoch summary headers"""
        base_headers = self._get_base_epoch_headers()
        
        # Add audio-specific headers
        audio_headers = [
            "avg_d_real_score", "avg_d_fake_score",  # Average discriminator scores
            "d_real_accuracy", "d_fake_accuracy",    # Discriminator accuracies
        ]
        
        return base_headers + audio_headers
    
    def _get_checkpoint_headers(self) -> List[str]:
        """Get WaveGAN-specific checkpoint metrics headers"""
        base_headers = self._get_base_checkpoint_headers()
        
        # Add audio-specific headers
        audio_headers = [
            "fad_score", "snr_score", "audio_quality_score",
            "audio_sample_paths"
        ]
        
        return base_headers + audio_headers
    
    # Compatibility methods for WaveGAN API
    def log_iteration_metrics(self, epoch: int, iteration: int, d_loss: float, g_loss: float,
                            d_grad_norm: float, g_grad_norm: float, gradient_penalty: float = 0.0,
                            **kwargs):
        """Compatibility method for WaveGAN - maps to log_iteration_snapshot"""
        losses = {'generator_loss': g_loss, 'discriminator_loss': d_loss}
        grads = {'generator_grad_norm': g_grad_norm, 'discriminator_grad_norm': d_grad_norm}
        gpu_stats = {'gpu_memory_used': 0.0, 'iteration_time_seconds': 0.1}  # Default values
        
        # Call the optimized method according to metrics-checkpoints-how-to.txt
        self.log_iteration_snapshot(epoch, iteration, losses, grads, gpu_stats)
    
    def log_epoch_metrics(self, epoch: int, avg_d_loss: float, avg_g_loss: float,
                         epoch_duration_minutes: float, **kwargs):
        """Compatibility method for WaveGAN - maps to log_epoch_stats"""
        epoch_losses = {
            'avg_g': avg_g_loss, 'avg_d': avg_d_loss,
            'min_g': avg_g_loss, 'min_d': avg_d_loss,  # Default values
            'max_g': avg_g_loss, 'max_d': avg_d_loss   # Default values
        }
        epoch_grads = {'avg_g': 0.0, 'avg_d': 0.0, 'max_g': 0.0, 'max_d': 0.0}  # Default values
        epoch_lr = {'lr_g': 0.0002, 'lr_d': 0.0002}  # Default values
        epoch_gpu_stats = {'gpu_memory': 0.0, 'gpu_util': 0.0, 'cpu_usage': 0.0, 'ram_usage': 0.0}
        
        # Call the optimized method according to metrics-checkpoints-how-to.txt
        self.log_epoch_stats(epoch, epoch_losses, epoch_grads, epoch_lr, epoch_gpu_stats, epoch_duration_minutes)
    
    def log_iteration_stats(self, epoch: int, iteration: int,
                          generator_loss: float, discriminator_loss: float,
                          generator_grad_norm: float, discriminator_grad_norm: float,
                          learning_rate_g: float, learning_rate_d: float,
                          batch_size: int, iteration_time: float,
                          d_real_score: float, d_fake_score: float,
                          sample_rate: int, audio_length: int, **kwargs):
        """Log WaveGAN iteration statistics using optimized snapshot method"""
        
        # Map to optimized structure - call log_iteration_snapshot
        losses = {'generator_loss': generator_loss, 'discriminator_loss': discriminator_loss}
        grads = {'generator_grad_norm': generator_grad_norm, 'discriminator_grad_norm': discriminator_grad_norm}
        gpu_stats = {'gpu_memory_used': 0.0, 'iteration_time_seconds': iteration_time}
        
        # Use optimized snapshot method according to metrics-checkpoints-how-to.txt
        self.log_iteration_snapshot(epoch, iteration, losses, grads, gpu_stats)
    
    def _get_domain_epoch_data(self, domain_kwargs: Dict) -> List:
        """Get WaveGAN-specific epoch summary data"""
        
        # Get domain-specific metrics from kwargs
        avg_d_real = domain_kwargs.get('avg_d_real_score', 0.0)
        avg_d_fake = domain_kwargs.get('avg_d_fake_score', 0.0)
        d_real_accuracy = domain_kwargs.get('d_real_accuracy', 0.0)
        d_fake_accuracy = domain_kwargs.get('d_fake_accuracy', 0.0)
        
        return [avg_d_real, avg_d_fake, d_real_accuracy, d_fake_accuracy]
    
    def calculate_frechet_audio_distance(self, real_audio: torch.Tensor, 
                                       fake_audio: torch.Tensor) -> float:
        """Calculate Fréchet Audio Distance (FAD) score
        
        Args:
            real_audio: Tensor of real audio samples
            fake_audio: Tensor of generated audio samples
            
        Returns:
            FAD score (lower is better)
        """
        try:
            # This is a simplified placeholder implementation
            # In practice, you would use a proper FAD implementation
            # that uses VGGish or similar audio features
            
            # Convert to numpy arrays
            real_np = real_audio.detach().cpu().numpy()
            fake_np = fake_audio.detach().cpu().numpy()
            
            # Calculate spectral features as proxy for VGGish features
            real_features = self._extract_spectral_features(real_np)
            fake_features = self._extract_spectral_features(fake_np)
            
            # Calculate means and covariances
            mu_real = np.mean(real_features, axis=0)
            mu_fake = np.mean(fake_features, axis=0)
            
            # Check if we have enough samples for covariance calculation
            min_samples_needed = real_features.shape[1] + 1  # Need at least n_features + 1 samples
            
            if real_features.shape[0] < min_samples_needed or fake_features.shape[0] < min_samples_needed:
                # Not enough samples for proper covariance - use simplified distance
                diff = mu_real - mu_fake
                fad_score = np.sum(diff ** 2)
                print(f"ℹ️  FAD: Using simplified calculation (insufficient samples: real={real_features.shape[0]}, fake={fake_features.shape[0]}, need={min_samples_needed})")
            else:
                # Full FAD calculation with covariance
                with np.errstate(divide='ignore', invalid='ignore'):
                    sigma_real = np.cov(real_features, rowvar=False)
                    sigma_fake = np.cov(fake_features, rowvar=False)
                    
                    # Ensure matrices are positive definite
                    sigma_real = sigma_real + np.eye(sigma_real.shape[0]) * 1e-6
                    sigma_fake = sigma_fake + np.eye(sigma_fake.shape[0]) * 1e-6
                    
                    # FAD calculation
                    diff = mu_real - mu_fake
                    
                    try:
                        sqrt_product = np.sqrt(sigma_real @ sigma_fake)
                        fad_score = np.sum(diff ** 2) + np.trace(sigma_real + sigma_fake - 2 * sqrt_product)
                    except np.linalg.LinAlgError:
                        # Fallback to simplified calculation if matrix operations fail
                        fad_score = np.sum(diff ** 2)
                        print("ℹ️  FAD: Using simplified calculation (matrix operation failed)")
            
            self.last_fad_score = float(fad_score)
            return self.last_fad_score
            
        except Exception as e:
            print(f"⚠️  Error calculating FAD score: {e}")
            return 999.0  # High value indicates poor quality
    
    def calculate_signal_to_noise_ratio(self, audio_signal: torch.Tensor) -> float:
        """Calculate Signal-to-Noise Ratio (SNR)
        
        Args:
            audio_signal: Audio signal tensor
            
        Returns:
            SNR in dB (higher is better)
        """
        try:
            audio_np = audio_signal.detach().cpu().numpy()
            
            # Calculate signal power
            signal_power = np.mean(audio_np ** 2)
            
            # Estimate noise power (using high-frequency components as proxy)
            # Apply high-pass filter approximation
            diff_signal = np.diff(audio_np, axis=-1)
            noise_power = np.mean(diff_signal ** 2)
            
            # Calculate SNR in dB
            if noise_power > 0:
                snr_db = 10 * np.log10(signal_power / noise_power)
            else:
                snr_db = 100.0  # Very high SNR if no noise detected
            
            self.last_snr_score = float(snr_db)
            return self.last_snr_score
            
        except Exception as e:
            print(f"⚠️  Error calculating SNR: {e}")
            return 0.0  # Low SNR indicates poor quality
    
    def _extract_spectral_features(self, audio_np: np.ndarray) -> np.ndarray:
        """Extract spectral features from audio for FAD calculation
        
        Args:
            audio_np: Audio samples as numpy array
            
        Returns:
            Feature matrix
        """
        try:
            # Simple spectral features
            features_list = []
            
            for audio_sample in audio_np:
                # FFT-based features
                fft = np.fft.fft(audio_sample)
                magnitude = np.abs(fft)
                
                # Extract features from different frequency bands
                n_samples = len(magnitude)
                low_freq = magnitude[:n_samples//4]
                mid_freq = magnitude[n_samples//4:n_samples//2]
                high_freq = magnitude[n_samples//2:3*n_samples//4]
                
                features = np.array([
                    np.mean(low_freq),   # Low frequency energy
                    np.std(low_freq),    # Low frequency variance
                    np.mean(mid_freq),   # Mid frequency energy
                    np.std(mid_freq),    # Mid frequency variance
                    np.mean(high_freq),  # High frequency energy
                    np.std(high_freq),   # High frequency variance
                    np.mean(magnitude),  # Overall spectral centroid
                    np.std(magnitude),   # Spectral spread
                ])
                
                features_list.append(features)
            
            return np.array(features_list)
            
        except Exception as e:
            print(f"⚠️  Error extracting spectral features: {e}")
            # Return dummy features
            return np.random.randn(len(audio_np), 8)
    
    def save_audio_samples(self, fake_audio: torch.Tensor, 
                          sample_dir: str, sample_rate: int = 16000,
                          prefix: str = "sample") -> List[str]:
        """Save generated audio samples
        
        Args:
            fake_audio: Tensor of generated audio
            sample_dir: Directory to save samples
            sample_rate: Audio sample rate
            prefix: Filename prefix
            
        Returns:
            List of saved file paths
        """
        try:
            os.makedirs(sample_dir, exist_ok=True)
            saved_paths = []
            
            # Convert tensor to numpy
            audio_np = fake_audio.detach().cpu().numpy()
            
            # Normalize audio to [-1, 1] range
            if audio_np.max() > 1.0 or audio_np.min() < -1.0:
                audio_np = audio_np / np.max(np.abs(audio_np))
            
            # Save individual audio files
            for i, audio_sample in enumerate(audio_np):
                filename = f"{prefix}_{self.generated_samples_count + i}.wav"
                file_path = os.path.join(sample_dir, filename)
                
                # Ensure audio_sample is 1D
                if len(audio_sample.shape) > 1:
                    audio_sample = audio_sample.flatten()
                
                sf.write(file_path, audio_sample, sample_rate)
                saved_paths.append(file_path)
            
            self.generated_samples_count += len(audio_np)
            return saved_paths
            
        except Exception as e:
            print(f"⚠️  Error saving audio samples: {e}")
            return []
    
    def log_checkpoint_metrics(self, checkpoint_type: str, epoch: int, iteration: int,
                             generator, discriminator, real_samples: Optional[torch.Tensor] = None,
                             sample_rate: int = 16000):
        """Log metrics for a checkpoint with audio quality assessment
        
        Args:
            checkpoint_type: Type of checkpoint
            epoch: Current epoch
            iteration: Current iteration
            generator: Generator model
            discriminator: Discriminator model
            real_samples: Real audio samples for comparison
            sample_rate: Audio sample rate
        """
        try:
            import csv
            from datetime import datetime
            
            timestamp = datetime.now().isoformat()
            
            # Generate fake samples for evaluation
            generator.eval()
            with torch.no_grad():
                noise = torch.randn(8, generator.latent_dim if hasattr(generator, 'latent_dim') else 100)
                if torch.cuda.is_available():
                    noise = noise.cuda()
                fake_samples = generator(noise)
            generator.train()
            
            # Calculate audio quality metrics
            fad_score = 999.0  # Default high value
            snr_score = 0.0    # Default low value
            
            if real_samples is not None:
                fad_score = self.calculate_frechet_audio_distance(real_samples, fake_samples)
            
            snr_score = self.calculate_signal_to_noise_ratio(fake_samples)
            
            # Save sample audio files
            sample_dir = os.path.join(self.output_dir, "samples", checkpoint_type)
            sample_paths = self.save_audio_samples(
                fake_samples, sample_dir, sample_rate,
                f"epoch_{epoch}_{checkpoint_type}"
            )
            
            # Calculate model parameters
            gen_params = sum(p.numel() for p in generator.parameters())
            disc_params = sum(p.numel() for p in discriminator.parameters())
            
            # Calculate overall audio quality score (composite metric)
            audio_quality_score = self._calculate_audio_quality_score(fad_score, snr_score)
            
            # Prepare checkpoint metrics row
            checkpoint_row = [
                checkpoint_type, epoch, iteration, timestamp,
                gen_params, disc_params,
                fad_score, snr_score, audio_quality_score,
                str(sample_paths)  # Convert list to string for CSV
            ]
            
            # Write to checkpoint metrics CSV
            with open(self.checkpoint_metrics_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(checkpoint_row)
            
            print(f"📊 Checkpoint metrics logged - FAD: {fad_score:.2f}, SNR: {snr_score:.1f}dB")
            
        except Exception as e:
            print(f"❌ Error logging checkpoint metrics: {e}")
    
    def _calculate_audio_quality_score(self, fad_score: float, snr_score: float) -> float:
        """Calculate composite audio quality score
        
        Args:
            fad_score: FAD score (lower is better)
            snr_score: SNR score (higher is better)
            
        Returns:
            Composite quality score (0-100, higher is better)
        """
        # Normalize FAD score (typical range 0-100, lower is better)
        fad_normalized = max(0, min(100, 100 - fad_score))
        
        # Normalize SNR score (typical range 0-60 dB, higher is better)
        snr_normalized = max(0, min(100, (snr_score / 60) * 100))
        
        # Weighted combination
        quality_score = 0.6 * fad_normalized + 0.4 * snr_normalized
        
        return float(quality_score)
