"""
WaveGAN Training Pipeline
========================

Complete training pipeline including WGAN-GP loss, gradient penalty,
model saving, and sample generation utilities with comprehensive
metrics collection and checkpoint management.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import matplotlib.pyplot as plt
import time
import logging
import warnings

# Suppress torchaudio warnings that break progress bars
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
from tqdm import tqdm
import os
from torch.utils.data import DataLoader

from .config_loader import WaveGANConfig
from .dataset import AudioDataset
from .generator import WaveGANGenerator
from .discriminator import WaveGANDiscriminator
from .metrics import MetricsCollector
from .checkpoints import CheckpointManager
from .analytics import TrainingAnalyzer
from .convergence_detector import ConvergenceDetector, ConvergenceStatus
from .advanced_monitoring import get_resource_monitor, cleanup_resource_monitor

# Configure logger for training
logger = logging.getLogger(__name__)

class WaveGANTrainer:
    """Complete training pipeline for WaveGAN with metrics and checkpoints"""
    
    def __init__(self, config: WaveGANConfig):
        """Initialize trainer with configuration object"""
        self.config = config
        
        # Extract frequently used values
        self.device = config.device
        self.latent_dim = int(config.get('model', 'latent_dim'))
        self.audio_length = int(config.get('audio', 'audio_length_samples'))
        self.batch_size = int(config.get('training', 'batch_size'))
        self.learning_rate = float(config.get('training', 'learning_rate'))
        self.epochs = int(config.get('training', 'epochs'))
        self.n_critic = int(config.get('training', 'n_critic'))
        self.gp_weight = float(config.get('training', 'gp_weight'))
        self.sample_rate = int(config.get('audio', 'sample_rate'))
        
        # Extract model configuration
        self.model_dim = int(config.get('model', 'model_dim'))
        self.kernel_len = int(config.get('model', 'kernel_len'))
        
        # Initialize models
        self.generator = WaveGANGenerator(
            latent_dim=self.latent_dim, 
            output_length=self.audio_length,
            model_dim=self.model_dim,
            kernel_len=self.kernel_len
        ).to(self.device)
        self.discriminator = WaveGANDiscriminator(
            input_length=self.audio_length,
            model_dim=self.model_dim,
            kernel_len=self.kernel_len
        ).to(self.device)
        
        # Initialize optimizers
        self.g_optimizer = optim.Adam(
            self.generator.parameters(), 
            lr=self.learning_rate, 
            betas=(0.5, 0.9)
        )
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(), 
            lr=self.learning_rate, 
            betas=(0.5, 0.9)
        )
        
        # Initialize metrics and checkpoint systems
        # WaveGANConfig stores the actual config in .config attribute
        config_dict = config.config if hasattr(config, 'config') else (config.to_dict() if hasattr(config, 'to_dict') else config.__dict__)
        self.metrics = MetricsCollector(output_dir=config.output_dir, config=config_dict)
        self.checkpoint_manager = CheckpointManager(output_dir=config.output_dir)
        self.analyzer = TrainingAnalyzer(output_dir=config.output_dir)
        
        # Initialize advanced monitoring and convergence detection
        self.resource_monitor = get_resource_monitor(start_monitoring=True)
        
        # Initialize convergence detector with configuration
        from .convergence_detector import ConvergenceConfig, ConvergenceDetector
        
        # Create custom convergence config from settings
        if hasattr(config, 'config') and config.config:
            convergence_config = config.config.get('convergence', {})
            
            if convergence_config.get('enabled', True):
                conv_config = ConvergenceConfig(
                    enabled=convergence_config.get('enabled', True),
                    min_iterations=convergence_config.get('min_iterations', 300),
                    check_frequency=convergence_config.get('check_frequency', 75),
                    patience=convergence_config.get('patience', 200),
                    min_improvement=convergence_config.get('min_improvement', 0.0003),
                    loss_explosion_threshold=convergence_config.get('loss_explosion_threshold', 500.0),
                    grad_norm_threshold=convergence_config.get('grad_norm_threshold', 5000.0)
                )
            else:
                # Create detector with convergence disabled
                conv_config = ConvergenceConfig(enabled=False)
        else:
            # Default conservative configuration for WaveGAN
            conv_config = ConvergenceConfig(
                enabled=True,
                min_iterations=300,
                check_frequency=75,
                patience=200,
                min_improvement=0.0003,
                loss_explosion_threshold=500.0,
                grad_norm_threshold=5000.0
            )
        
        self.convergence_detector = ConvergenceDetector(conv_config, config.output_dir)
        
        # Training metrics (legacy for compatibility)
        self.d_losses = []
        self.g_losses = []
        
        # Iteration counter
        self.global_iteration = 0
        
        # Log model info
        self.g_params = sum(p.numel() for p in self.generator.parameters())
        self.d_params = sum(p.numel() for p in self.discriminator.parameters())
        config.logger.info(f"Generator parameters: {self.g_params:,}")
        config.logger.info(f"Discriminator parameters: {self.d_params:,}")
        print(f"🏗️  Generator parameters: {self.g_params:,}")
        print(f"🏗️  Discriminator parameters: {self.d_params:,}")
        
        # Save experiment configuration
        self._save_experiment_config()
    
    def _save_experiment_config(self):
        """Save experiment configuration and model details"""
        config_data = {
            'mode': os.path.basename(self.config.config_path),
            'hyperparameters': {
                'latent_dim': self.latent_dim,
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'learning_rate': self.learning_rate,
                'n_critic': self.n_critic,
                'gp_weight': self.gp_weight,
                'audio_length': self.audio_length,
                'sample_rate': self.sample_rate
            },
            'model_architecture': {
                'generator_params': self.g_params,
                'discriminator_params': self.d_params,
                'total_params': self.g_params + self.d_params
            },
            'dataset_info': {
                'dataset_files': self.config.dataset_files,
                'dataset_metadata': self.config.dataset_metadata
            },
            'system_info': {
                'device': str(self.device),
                'cuda_available': torch.cuda.is_available(),
                'cuda_device_name': torch.cuda.get_device_name() if torch.cuda.is_available() else None
            }
        }
        
        self.metrics.save_experiment_config(config_data)
    
    def _calculate_gradient_norms(self):
        """Calculate gradient norms for both models"""
        g_grad_norm = 0.0
        d_grad_norm = 0.0
        
        # Generator gradients
        for param in self.generator.parameters():
            if param.grad is not None:
                g_grad_norm += param.grad.data.norm(2).item() ** 2
        g_grad_norm = g_grad_norm ** 0.5
        
        # Discriminator gradients
        for param in self.discriminator.parameters():
            if param.grad is not None:
                d_grad_norm += param.grad.data.norm(2).item() ** 2
        d_grad_norm = d_grad_norm ** 0.5
        
        return g_grad_norm, d_grad_norm
    
    def gradient_penalty(self, real_samples, fake_samples):
        """Calculate gradient penalty for WGAN-GP"""
        batch_size = real_samples.size(0)
        
        # Random interpolation factor - ensure it matches audio dimensions
        epsilon = torch.rand(batch_size, 1, 1).to(self.device)
        epsilon = epsilon.expand_as(real_samples)
        
        # Interpolated samples
        interpolated = epsilon * real_samples + (1 - epsilon) * fake_samples
        interpolated.requires_grad_(True)
        
        # Get discriminator output for interpolated samples
        d_interpolated = self.discriminator(interpolated)
        
        # Calculate gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones(d_interpolated.size()).to(self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Calculate penalty
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        penalty = ((gradient_norm - 1) ** 2).mean()
        
        return penalty
    
    def save_audio_samples(self, epoch, num_samples=5):
        """Save generated audio samples with debugging"""
        self.config.logger.info(f"=== SAVING AUDIO SAMPLES - EPOCH {epoch} ===")
        self.generator.eval()
        
        with torch.no_grad():
            # Generate samples
            z = torch.randn(num_samples, self.latent_dim).to(self.device)
            fake_audio = self.generator(z)
            
            # Log generated audio stats
            self.config.log_tensor_stats(f"Generated_audio_epoch_{epoch}", fake_audio)
            
            # Normalize audio properly
            fake_audio_cpu = fake_audio.cpu()
            
            # Create samples directory
            samples_dir = self.config.samples_dir
            os.makedirs(samples_dir, exist_ok=True)
            
            # Save samples with proper normalization
            for i, audio_sample in enumerate(fake_audio_cpu):
                self.config.logger.info(f"Sample {i} - min: {audio_sample.min():.6f}, max: {audio_sample.max():.6f}")
                
                # Normalize to [-0.9, 0.9] to avoid clipping
                if audio_sample.abs().max() > 1e-6:
                    audio_sample = audio_sample / audio_sample.abs().max() * 0.9
                    self.config.logger.info(f"Sample {i} after normalization - min: {audio_sample.min():.6f}, max: {audio_sample.max():.6f}")
                else:
                    self.config.logger.warning(f"Sample {i} is silent, keeping as zeros")
                
                # Save as wav file in 16-bit format
                # audio_sample has shape [1, length] - keep the channel dimension
                audio_int16 = (audio_sample * 32767).clamp(-32768, 32767).to(torch.int16)
                filename = os.path.join(samples_dir, f'epoch_{epoch:03d}_sample_{i+1:02d}.wav')
                
                try:
                    # audio_int16 already has correct shape [1, length] for torchaudio.save
                    torchaudio.save(filename, audio_int16, self.sample_rate, encoding="PCM_S", bits_per_sample=16)
                    
                    # Log file size for verification
                    if os.path.exists(filename):
                        file_size = os.path.getsize(filename)
                        self.config.logger.info(f"Saved {filename} - file size: {file_size} bytes")
                    else:
                        self.config.logger.error(f"Failed to save {filename}")
                except Exception as e:
                    self.config.logger.error(f"Error saving {filename}: {e}")
            
            self.config.logger.info("=== AUDIO SAMPLES SAVED ===")
        
        self.generator.train()
    
    def _handle_epoch_checkpoints(self, epoch: int, avg_g_loss: float, avg_d_loss: float, metrics_config: dict):
        """Handle all checkpoint and sample generation logic at the end of each epoch"""
        # Save audio samples
        if epoch % metrics_config.get('audio_samples_per_checkpoint', 5) == 0:
            self.save_audio_samples(epoch)
        
        # Milestone epoch checkpoint
        if epoch % metrics_config['milestone_epoch_frequency'] == 0:
            self.checkpoint_manager.save_checkpoint(
                epoch=epoch,
                iteration=self.global_iteration,
                generator=self.generator,
                discriminator=self.discriminator,
                optimizer_g=self.g_optimizer,
                optimizer_d=self.d_optimizer,
                generator_loss=avg_g_loss,
                discriminator_loss=avg_d_loss
            )
            # Log audio quality metrics for epoch checkpoint
            sample_rate = self.config.get('audio', 'sample_rate') or 22050
            self.metrics.log_checkpoint_metrics(
                checkpoint_type="epoch",
                epoch=epoch,
                iteration=self.global_iteration,
                generator=self.generator,
                discriminator=self.discriminator,
                real_samples=None,  # Use None since we don't have batch here
                sample_rate=sample_rate
            )
    
    def train(self):
        """Main training loop with comprehensive metrics"""
        self.config.logger.info(f"Using device: {self.device}")
        self.config.logger.info(f"Audio length: {self.audio_length} samples ({self.audio_length/self.sample_rate:.2f} seconds)")
        
        # Initialize dataset
        max_files_per_class = self.config.get('dataset', 'max_files_per_class', None)
        dataset = AudioDataset(
            self.config.dataset_files, 
            self.config.dataset_metadata, 
            target_length=self.audio_length,
            max_files_per_class=max_files_per_class
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.config.get('hardware', 'num_workers', 2))
        
        print(f"📊 Dataset loaded: {len(dataset)} samples")
        print(f"🚀 Starting training for {self.epochs} epochs...")
        
        # Get metrics config for checkpoints and samples
        metrics_config = self.config.get('metrics_and_checkpoints')
        if metrics_config is None:
            # Default checkpoint frequencies if config is missing
            metrics_config = {
                'detailed_logging_frequency': 100,
                'emergency_checkpoint_frequency': 100,
                'milestone_checkpoint_frequency': 1000,
                'milestone_epoch_frequency': 10,
                'audio_samples_per_checkpoint': 5
            }
        
        # Main training loop
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            self.config.logger.info(f"=== EPOCH {epoch+1}/{self.epochs} STARTED ===")
            
            pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{self.epochs}')
            epoch_d_losses = []
            epoch_g_losses = []
            
            for iteration, real_audio in enumerate(pbar):
                # Check and rotate log if needed
                if iteration % 1000 == 0:
                    self.config.check_and_rotate_log()
                
                real_audio = real_audio.to(self.device)
                current_batch_size = real_audio.size(0)
                
                # ========================
                # Train Discriminator
                # ========================
                for _ in range(self.n_critic):
                    self.d_optimizer.zero_grad()
                    
                    # Generate fake samples
                    z = torch.randn(current_batch_size, self.latent_dim).to(self.device)
                    fake_audio = self.generator(z).detach()
                    
                    # Discriminator outputs
                    d_real = self.discriminator(real_audio)
                    d_fake = self.discriminator(fake_audio)
                    
                    # Gradient penalty
                    gp = self.gradient_penalty(real_audio, fake_audio)
                    
                    # Discriminator loss (WGAN-GP)
                    d_loss = d_fake.mean() - d_real.mean() + self.gp_weight * gp
                    
                    d_loss.backward()
                    self.d_optimizer.step()
                
                # ========================
                # Train Generator
                # ========================
                self.g_optimizer.zero_grad()
                
                # Generate fake samples
                z = torch.randn(current_batch_size, self.latent_dim).to(self.device)
                fake_audio = self.generator(z)
                
                # Generator loss (WGAN-GP)
                d_fake_g = self.discriminator(fake_audio)
                g_loss = -d_fake_g.mean()
                
                g_loss.backward()
                self.g_optimizer.step()
                
                # Calculate gradient norms BEFORE using them
                g_grad_norm, d_grad_norm = self._calculate_gradient_norms()
                
                # Update metrics
                epoch_d_losses.append(d_loss.item())
                epoch_g_losses.append(g_loss.item())
                
                # Update convergence detector
                convergence_status = self.convergence_detector.update(
                    epoch=epoch + 1,
                    generator_loss=g_loss.item(),
                    discriminator_loss=d_loss.item(),
                    grad_norm_g=g_grad_norm,
                    grad_norm_d=d_grad_norm,
                    learning_rate=self.learning_rate
                )
                
                # Check convergence status
                if convergence_status in [ConvergenceStatus.DIVERGED, ConvergenceStatus.EARLY_STOP]:
                    # Log to console with pbar.write for progress bar compatibility
                    pbar.write(f"⏹️  Training stopped due to convergence: {convergence_status.value}")
                    # Log to file
                    self.config.logger.warning(f"Training stopped due to convergence: {convergence_status.value}")
                    pbar.close()
                    break
                
                # Update progress bar
                pbar.set_postfix({
                    'D_loss': f'{d_loss.item():.4f}',
                    'G_loss': f'{g_loss.item():.4f}',
                    'GP': f'{gp.item():.4f}'
                })
                
                # Detailed logging and metrics collection
                if iteration % metrics_config['detailed_logging_frequency'] == 0:
                    # Log detailed stats to file only (console shows progress bar)
                    self.config.logger.info(f"Epoch {epoch+1}/{self.epochs}, Iter {iteration}: D_loss={d_loss.item():.6f}, G_loss={g_loss.item():.6f}")
                    self.config.logger.info(f"Gradient norms - G: {g_grad_norm:.6f}, D: {d_grad_norm:.6f}")
                    
                    # Log tensor stats for debugging
                    if hasattr(self.config, 'log_tensor_stats'):
                        # Log real and fake audio tensor statistics
                        self.config.log_tensor_stats(f"real_audio_epoch_{epoch}_iter_{iteration}", real_audio)
                        self.config.log_tensor_stats(f"fake_audio_epoch_{epoch}_iter_{iteration}", fake_audio)
                    
                    # Collect detailed metrics
                    self.metrics.log_iteration_metrics(
                        epoch=epoch,
                        iteration=iteration,
                        d_loss=d_loss.item(),
                        g_loss=g_loss.item(),
                        d_grad_norm=d_grad_norm,
                        g_grad_norm=g_grad_norm,
                        gradient_penalty=gp.item()
                    )
                
                # Log rotation check
                if hasattr(self.config, 'check_and_rotate_log'):
                    self.config.check_and_rotate_log()
                
                # Note: Checkpoint logic moved to end of epoch to match requirements
                
                self.global_iteration += 1
            
            # End of epoch processing
            epoch_duration = (time.time() - epoch_start_time) / 60  # minutes
            
            avg_d_loss = sum(epoch_d_losses) / len(epoch_d_losses)
            avg_g_loss = sum(epoch_g_losses) / len(epoch_g_losses)
            
            # Store losses for legacy compatibility
            self.d_losses.append(avg_d_loss)
            self.g_losses.append(avg_g_loss)
            
            self.config.logger.info(f"Epoch {epoch+1}/{self.epochs} completed in {epoch_duration:.2f} minutes")
            self.config.logger.info(f"Average losses - D: {avg_d_loss:.6f}, G: {avg_g_loss:.6f}")
            
            # Collect epoch metrics
            self.metrics.log_epoch_metrics(
                epoch=epoch,
                avg_d_loss=avg_d_loss,
                avg_g_loss=avg_g_loss,
                epoch_duration_minutes=epoch_duration
            )
            
            # Handle checkpoint and sample generation at end of epoch
            self._handle_epoch_checkpoints(epoch, avg_g_loss, avg_d_loss, metrics_config)
        
        print("\n🎯 Training completed!")
        
        # Stop resource monitoring and save data
        self.resource_monitor.stop_monitoring()
        monitoring_path = os.path.join(self.config.output_dir, "resource_monitoring.json")
        self.resource_monitor.save_monitoring_data(monitoring_path)
        
        # Print resource summary
        resource_summary = self.resource_monitor.get_training_summary()
        print(f"📊 Resource usage summary: {resource_summary.get('alerts', {}).get('total_count', 0)} alerts")
        
        # Print bottleneck analysis
        bottleneck_analysis = self.resource_monitor.get_bottleneck_analysis()
        if bottleneck_analysis.get('bottlenecks_detected', 0) > 0:
            print(f"⚠️ Performance bottlenecks detected: {bottleneck_analysis['bottlenecks_detected']}")
        
        # Save final models
        self.save_models("final")
        
        # Generate final samples
        self.save_audio_samples(self.epochs - 1, num_samples=10)
        
        # Generate analytics
        self.analyzer.generate_all_plots()
        
        print("✅ All training artifacts saved!")
    
    def save_checkpoint(self, suffix="checkpoint"):
        """Save current training state"""
        checkpoint_path = os.path.join(self.config.models_dir, f'wavegan_{suffix}.pth')
        
        checkpoint = {
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'g_optimizer': self.g_optimizer.state_dict(),
            'd_optimizer': self.d_optimizer.state_dict(),
            'global_iteration': self.global_iteration,
            'd_losses': self.d_losses,
            'g_losses': self.g_losses,
            'config_path': self.config.config_path
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.config.logger.info(f"Checkpoint saved: {checkpoint_path}")
        print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load training state from checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer'])
        self.global_iteration = checkpoint.get('global_iteration', 0)
        self.d_losses = checkpoint.get('d_losses', [])
        self.g_losses = checkpoint.get('g_losses', [])
        
        self.config.logger.info(f"Checkpoint loaded: {checkpoint_path}")
        print(f"💾 Checkpoint loaded: {checkpoint_path}")
    
    def save_models(self, suffix="final"):
        """Save generator and discriminator models"""
        gen_path = os.path.join(self.config.models_dir, f'wavegan_generator_{suffix}.pth')
        disc_path = os.path.join(self.config.models_dir, f'wavegan_discriminator_{suffix}.pth')
        
        torch.save(self.generator.state_dict(), gen_path)
        torch.save(self.discriminator.state_dict(), disc_path)
        
        self.config.logger.info(f"Models saved: {gen_path}, {disc_path}")
        print(f"💾 Models saved with suffix: {suffix}")
    
    def plot_losses(self):
        """Plot and save training losses"""
        if not self.d_losses or not self.g_losses:
            print("⚠️  No loss data to plot")
            return
            
        plt.figure(figsize=(10, 5))
        plt.plot(self.d_losses, label='Discriminator Loss', alpha=0.8)
        plt.plot(self.g_losses, label='Generator Loss', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('WaveGAN Training Losses')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = os.path.join(self.config.metrics_dir, 'training_losses.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Loss plot saved: {plot_path}")
        plt.close()
    
    def generate_samples(self, num_samples=10, model_path=None):
        """Generate audio samples using trained model"""
        if model_path is None:
            model_path = os.path.join(self.config.models_dir, 'wavegan_generator_final.pth')
        
        if not os.path.exists(model_path):
            # Use current generator if no saved model exists
            generator = self.generator
            print(f"🎵 Using current generator (no saved model found at {model_path})")
        else:
            # Load trained generator with correct parameters from config
            generator = WaveGANGenerator(
                latent_dim=self.latent_dim, 
                output_length=self.audio_length,
                model_dim=self.model_dim,
                kernel_len=self.kernel_len
            ).to(self.device)
            generator.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"🎵 Loaded generator from {model_path}")
        
        generator.eval()
        
        print(f"🎵 Generating {num_samples} audio samples...")
        
        final_samples_dir = os.path.join(self.config.samples_dir, 'final_samples')
        os.makedirs(final_samples_dir, exist_ok=True)
        
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(self.device)
            fake_audio = generator(z)
            
            # Save samples
            for i, audio in enumerate(fake_audio):
                # Normalize and convert to 16-bit
                if audio.abs().max() > 1e-6:
                    audio_normalized = audio / audio.abs().max() * 0.9  # Prevent clipping
                else:
                    audio_normalized = audio
                    
                audio_int16 = (audio_normalized * 32767).clamp(-32768, 32767).to(torch.int16)
                
                # Save as wav file in 16-bit format
                filename = os.path.join(final_samples_dir, f'generated_sample_{i+1:02d}.wav')
                # audio_int16 already has correct shape [1, length] for torchaudio.save
                torchaudio.save(filename, audio_int16, self.sample_rate, 
                              encoding="PCM_S", bits_per_sample=16)
        
        print(f"✅ Generated samples saved to '{final_samples_dir}' directory")
