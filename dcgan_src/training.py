"""
DCGAN Training Pipeline
======================

Complete training pipeline for DCGAN including BCE loss, model saving,
and sample generation utilities with comprehensive metrics collection
and checkpoint management.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import os
from torch.utils.data import DataLoader

from .config import (
    DEVICE, LATENT_DIM, BATCH_SIZE, EPOCHS, LEARNING_RATE, BETA1,
    IMAGE_SIZE, CHANNELS, logger, log_tensor_stats, check_and_rotate_log,
    CONFIG_MODE, DETAILED_LOGGING_FREQUENCY, DATASET_SPECTROGRAMS,
    DATASET_SPECTROGRAMS_METADATA
)
from .dataset import SpectrogramDataset, DummySpectrogramDataset, create_dataloader
from .generator import DCGANGenerator, weights_init_generator
from .discriminator import DCGANDiscriminator, weights_init_discriminator
from .metrics import MetricsCollector
from .checkpoints import CheckpointManager
from .analytics import TrainingAnalyzer
from .convergence_detector import ConvergenceDetector, ConvergenceStatus
from .advanced_metrics import calculate_fid_is_scores, get_metrics_calculator
from .advanced_monitoring import get_resource_monitor, cleanup_resource_monitor

class DCGANTrainer:
    """Complete training pipeline for DCGAN with metrics and checkpoints"""
    
    def __init__(self, dataset_spectrograms, dataset_metadata, config=None):
        """Initialize trainer with dataset paths and config"""
        self.dataset_spectrograms = dataset_spectrograms
        self.dataset_metadata = dataset_metadata
        self.config = config
        
        # Extract configuration values (use passed config if available, otherwise use defaults)
        if config:
            model_config = config.get_model_config()
            training_config = config.get_training_config()
            
            self.latent_dim = model_config['latent_dim']
            self.image_size = model_config['image_size']
            self.channels = model_config['channels']
            self.features_g = model_config['features_g']
            self.features_d = model_config['features_d']
            
            self.batch_size = training_config['batch_size']
            self.epochs = training_config['epochs']
            self.learning_rate = training_config['learning_rate']
            self.beta1 = training_config['beta1']
            
            self.device = config.device
        else:
            # Fallback to global constants
            self.latent_dim = LATENT_DIM
            self.image_size = IMAGE_SIZE
            self.channels = CHANNELS
            self.batch_size = BATCH_SIZE
            self.epochs = EPOCHS
            self.learning_rate = LEARNING_RATE
            self.beta1 = BETA1
            self.device = DEVICE
        
        # Initialize models with configuration values
        self.generator = DCGANGenerator(
            latent_dim=self.latent_dim, 
            channels=self.channels,
            image_size=self.image_size
        ).to(self.device)
        
        self.discriminator = DCGANDiscriminator(
            channels=self.channels,
            image_size=self.image_size
        ).to(self.device)
        
        # Apply weight initialization
        self.generator.apply(weights_init_generator)
        self.discriminator.apply(weights_init_discriminator)
        
        # Initialize optimizers
        self.g_optimizer = optim.Adam(
            self.generator.parameters(), 
            lr=self.learning_rate, 
            betas=(self.beta1, 0.999)
        )
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(), 
            lr=self.learning_rate, 
            betas=(self.beta1, 0.999)
        )
        
        # Initialize loss function
        self.criterion = nn.BCELoss()
        
        # Initialize metrics and checkpoint systems - FIXED: use consistent output_dir like WaveGAN
        if config and hasattr(config, 'output_dir'):
            # Use config-based paths (consistent with WaveGAN)
            # DCGANConfig stores the actual config in .config attribute
            config_dict = config.config if hasattr(config, 'config') else (config.to_dict() if hasattr(config, 'to_dict') else config.__dict__)
            self.metrics = MetricsCollector(output_dir=config.output_dir, config=config_dict)
            self.checkpoint_manager = CheckpointManager(checkpoint_dir=os.path.join(config.output_dir, "checkpoints"))
            self.analyzer = TrainingAnalyzer(output_dir=config.output_dir)
        else:
            # Fallback to legacy mode - use output_dcgan
            self.metrics = MetricsCollector(output_dir="output_dcgan")
            self.checkpoint_manager = CheckpointManager(checkpoint_dir="output_dcgan/checkpoints")
            self.analyzer = TrainingAnalyzer(output_dir="output_dcgan")
        
        # Initialize advanced monitoring and convergence detection
        self.resource_monitor = get_resource_monitor(start_monitoring=True)
        
        # Initialize convergence detector with configuration
        from .convergence_detector import ConvergenceConfig, ConvergenceDetector
        
        # Create custom convergence config from settings
        if self.config and hasattr(self.config, 'config') and self.config.config:
            convergence_config = self.config.config.get('convergence', {})
            
            if convergence_config.get('enabled', True):
                conv_config = ConvergenceConfig(
                    enabled=convergence_config.get('enabled', True),
                    min_iterations=convergence_config.get('min_iterations', 200),
                    check_frequency=convergence_config.get('check_frequency', 50),
                    patience=convergence_config.get('patience', 150),
                    min_improvement=convergence_config.get('min_improvement', 0.0005),
                    loss_explosion_threshold=convergence_config.get('loss_explosion_threshold', 200.0),
                    grad_norm_threshold=convergence_config.get('grad_norm_threshold', 2000.0)
                )
            else:
                # Create detector with convergence disabled
                conv_config = ConvergenceConfig(enabled=False)
        else:
            # Default conservative configuration
            conv_config = ConvergenceConfig(
                enabled=True,
                min_iterations=200,
                check_frequency=50,
                patience=150,
                min_improvement=0.0005,
                loss_explosion_threshold=200.0,
                grad_norm_threshold=2000.0
            )
        
        # Get output directory
        output_dir = getattr(self.config, 'output_dir', 'output_dcgan') if self.config else 'output_dcgan'
        self.convergence_detector = ConvergenceDetector(conv_config, output_dir)
        
        # Training metrics (legacy for compatibility)
        self.d_losses = []
        self.g_losses = []
        
        # Tracking variables for enhanced metrics
        self._recent_grad_norms = []  # Track recent gradient norms for stability
        
        # Iteration counter
        self.global_iteration = 0
        
        # Log model info
        self.g_params = sum(p.numel() for p in self.generator.parameters())
        self.d_params = sum(p.numel() for p in self.discriminator.parameters())
        logger.info(f"Generator parameters: {self.g_params:,}")
        logger.info(f"Discriminator parameters: {self.d_params:,}")
        print(f"🏗️  Generator parameters: {self.g_params:,}")
        print(f"🏗️  Discriminator parameters: {self.d_params:,}")
        
        # Save experiment configuration
        self.save_experiment_config()
    
    def save_experiment_config(self):
        """Save experiment configuration and model details"""
        config_data = {
            'mode': CONFIG_MODE,
            'hyperparameters': {
                'latent_dim': self.latent_dim,
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'learning_rate': self.learning_rate,
                'beta1': self.beta1,
                'image_size': self.image_size,
                'channels': self.channels
            },
            'model_architecture': {
                'generator_params': self.g_params,
                'discriminator_params': self.d_params,
                'total_params': self.g_params + self.d_params,
                'generator_info': self.generator.get_model_info(),
                'discriminator_info': self.discriminator.get_model_info()
            },
            'dataset_info': {
                'dataset_spectrograms': self.dataset_spectrograms,
                'dataset_metadata': self.dataset_metadata
            },
            'system_info': {
                'device': str(DEVICE),
                'cuda_available': torch.cuda.is_available(),
                'cuda_device_name': torch.cuda.get_device_name() if torch.cuda.is_available() else None
            }
        }
        
        self.metrics.save_experiment_config(config_data)
    
    def create_dataloader(self):
        """Create dataloader for training"""
        # Check if dummy data is explicitly requested
        if self.config and hasattr(self.config, 'config'):
            config_dict = self.config.config
            use_dummy = config_dict.get('dataset', {}).get('use_dummy_data', False)
            dummy_size = config_dict.get('dataset', {}).get('dummy_dataset_size', 1000)
        else:
            use_dummy = False
            dummy_size = 1000
        
        if use_dummy:
            print(f"🔄 Using dummy dataset (size: {dummy_size}) as configured...")
            dataset = DummySpectrogramDataset(
                size=dummy_size,
                image_size=self.image_size,
                channels=self.channels
            )
        else:
            try:
                # Try to load real spectrogram dataset
                dataset = SpectrogramDataset(
                    self.dataset_spectrograms,
                    self.dataset_metadata,
                    image_size=self.image_size
                )
                print(f"✅ Loaded real spectrogram dataset with {len(dataset)} images")
                
            except Exception as e:
                print(f"⚠️  Could not load real dataset: {e}")
                print("🔄 Using dummy dataset for testing...")
                dataset = DummySpectrogramDataset(
                    size=100 if CONFIG_MODE == "TESTING" else 1000,
                    image_size=self.image_size,
                    channels=self.channels
                )
        
        return create_dataloader(dataset, self.batch_size, shuffle=True)
    
    def calculate_gradient_norm(self, model):
        """Calculate gradient norm for a model"""
        total_norm = 0.0
        param_count = 0
        
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        return (total_norm ** 0.5) if param_count > 0 else 0.0
    
    def train(self):
        """Main training loop"""
        print(f"🚀 Starting DCGAN training - {CONFIG_MODE} mode")
        print(f"🎯 Target: {self.epochs} epochs, {self.batch_size} batch size")
        print("="*60)
        
        # Create dataloader
        dataloader = self.create_dataloader()
        
        # Fixed noise for progress tracking
        fixed_noise = torch.randn(16, self.latent_dim, 1, 1, device=self.device)
        
        # Training loop
        self.generator.train()
        self.discriminator.train()
        
        total_start_time = time.time()
        
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            
            # Epoch progress
            epoch_iterator = tqdm(
                dataloader,
                desc=f"Epoch {epoch+1}/{self.epochs}",
                leave=False
            )
            
            for i, real_data in enumerate(epoch_iterator):
                iteration_start_time = time.time()
                
                self.global_iteration += 1
                batch_size = real_data.size(0)
                real_data = real_data.to(self.device)
                
                # Create labels
                real_label = torch.ones(batch_size, device=self.device)
                fake_label = torch.zeros(batch_size, device=self.device)
                
                # ============================================
                # Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
                # ============================================
                self.discriminator.zero_grad()
                
                # Train with real data
                output_real = self.discriminator(real_data)
                loss_d_real = self.criterion(output_real, real_label)
                loss_d_real.backward()
                
                # Train with fake data
                noise = torch.randn(batch_size, self.latent_dim, 1, 1, device=self.device)
                fake_data = self.generator(noise)
                output_fake = self.discriminator(fake_data.detach())
                loss_d_fake = self.criterion(output_fake, fake_label)
                loss_d_fake.backward()
                
                loss_d = loss_d_real + loss_d_fake
                self.d_optimizer.step()
                
                # ============================================
                # Train Generator: max log(D(G(z)))
                # ============================================
                self.generator.zero_grad()
                
                # Since we just updated D, perform another forward pass
                output_fake = self.discriminator(fake_data)
                loss_g = self.criterion(output_fake, real_label)  # We want D to think fake is real
                loss_g.backward()
                self.g_optimizer.step()
                
                # Calculate gradient norms
                g_grad_norm = self.calculate_gradient_norm(self.generator)
                d_grad_norm = self.calculate_gradient_norm(self.discriminator)
                
                # Track gradient norms for stability score calculation
                avg_grad_norm = (g_grad_norm + d_grad_norm) / 2
                self._recent_grad_norms.append(avg_grad_norm)
                if len(self._recent_grad_norms) > 10:  # Keep only recent 10 values
                    self._recent_grad_norms.pop(0)
                
                # Record losses
                self.g_losses.append(loss_g.item())
                self.d_losses.append(loss_d.item())
                
                # Calculate scores
                d_real_score = output_real.mean().item()
                d_fake_score = output_fake.mean().item()
                
                iteration_time = time.time() - iteration_start_time
                
                # Update convergence detector
                convergence_status = self.convergence_detector.update(
                    epoch=epoch + 1,
                    generator_loss=loss_g.item(),
                    discriminator_loss=loss_d.item(),
                    grad_norm_g=g_grad_norm,
                    grad_norm_d=d_grad_norm,
                    learning_rate=self.g_optimizer.param_groups[0]['lr']
                )
                
                # Check convergence status
                if convergence_status in [ConvergenceStatus.DIVERGED, ConvergenceStatus.EARLY_STOP]:
                    logger.warning(f"Training stopped due to convergence: {convergence_status.value}")
                    epoch_iterator.close()
                    break
                
                # Log iteration metrics
                self.metrics.log_iteration_stats(
                    epoch=epoch + 1,
                    iteration=self.global_iteration,
                    generator_loss=loss_g.item(),
                    discriminator_loss=loss_d.item(),
                    generator_grad_norm=g_grad_norm,
                    discriminator_grad_norm=d_grad_norm,
                    learning_rate_g=self.g_optimizer.param_groups[0]['lr'],
                    learning_rate_d=self.d_optimizer.param_groups[0]['lr'],
                    d_real_score=d_real_score,
                    d_fake_score=d_fake_score,
                    batch_size=batch_size,
                    image_size=IMAGE_SIZE,
                    iteration_time=iteration_time
                )
                
                # Update progress bar
                epoch_iterator.set_postfix({
                    'G_loss': f'{loss_g.item():.4f}',
                    'D_loss': f'{loss_d.item():.4f}',
                    'D(real)': f'{d_real_score:.3f}',
                    'D(fake)': f'{d_fake_score:.3f}'
                })
                
                # Detailed logging
                if self.global_iteration % DETAILED_LOGGING_FREQUENCY == 0:
                    logger.info(f"Iter {self.global_iteration}: G_loss={loss_g.item():.4f}, "
                               f"D_loss={loss_d.item():.4f}, D(real)={d_real_score:.3f}, "
                               f"D(fake)={d_fake_score:.3f}")
                    
                    # Log tensor stats for debugging
                    log_tensor_stats("real_data", real_data, self.global_iteration)
                    log_tensor_stats("fake_data", fake_data, self.global_iteration)
                
                # NOTE: Checkpoints are handled per EPOCH, not per iteration
                # See handle_checkpoints() call after epoch completion
                
                # Rotate log if needed
                check_and_rotate_log()
            
            # End of epoch
            epoch_duration = (time.time() - epoch_start_time) / 60  # minutes
            
            # Calculate epoch averages for logging
            epoch_losses = {
                'avg_g': sum(self.g_losses[-len(dataloader):]) / len(dataloader),
                'avg_d': sum(self.d_losses[-len(dataloader):]) / len(dataloader),
                'min_g': min(self.g_losses[-len(dataloader):]),
                'min_d': min(self.d_losses[-len(dataloader):]),
                'max_g': max(self.g_losses[-len(dataloader):]),
                'max_d': max(self.d_losses[-len(dataloader):])
            }
            
            # Aggregate and log epoch metrics using new API
            epoch_metrics = self.metrics.aggregate_epoch_metrics()
            
            # Log epoch stats (EXTENDED with missing metrics per metrics-checkpoints-how-to.txt)
            resource_stats = self.resource_monitor.get_current_stats()
            
            # Extract GPU stats in expected format
            gpu_stats = {'gpu_memory': 0, 'gpu_util': 0, 'cpu_usage': 0, 'ram_usage': 0}
            if 'gpu' in resource_stats and resource_stats['gpu']['count'] > 0:
                gpu_device = resource_stats['gpu']['devices'][0]  # Use first GPU
                gpu_stats['gpu_memory'] = gpu_device['memory']['used_mb'] / 1024  # Convert to GB
                gpu_stats['gpu_util'] = gpu_device['utilization_percent']
            
            if 'cpu' in resource_stats:
                gpu_stats['cpu_usage'] = resource_stats['cpu']['usage_percent']
            
            if 'memory' in resource_stats:
                gpu_stats['ram_usage'] = resource_stats['memory']['usage_percent']
            
            self.metrics.log_epoch_stats(
                epoch=epoch + 1,
                epoch_losses=epoch_losses,
                epoch_grads=epoch_metrics.get('epoch_grads', {'avg_g': 0.0, 'avg_d': 0.0, 'max_g': 0.0, 'max_d': 0.0}),
                epoch_lr={'lr_g': self.g_optimizer.param_groups[0]['lr'], 'lr_d': self.d_optimizer.param_groups[0]['lr']},
                epoch_gpu_stats=gpu_stats,
                epoch_timer=epoch_duration,
                # ADDED MISSING METRICS for compliance with metrics-checkpoints-how-to.txt:
                gradient_penalty_value=0.0,  # DCGAN doesn't use gradient penalty (WGAN-GP feature)
                wasserstein_distance_estimate=0.0,  # DCGAN uses BCE loss, not Wasserstein
                frechet_inception_distance=0.0,  # TODO: Implement FID calculation
                signal_to_noise_ratio=0.0,  # TODO: Implement for spectrograms if needed  
                inception_score=0.0,  # TODO: Implement IS calculation
                convergence_indicator=self._calculate_convergence_indicator(),
                training_stability_score=self._calculate_stability_score(),
                batch_size=BATCH_SIZE,
                sequence_length=IMAGE_SIZE,  # Using image_size as sequence_length for DCGAN
                total_iterations_in_epoch=len(dataloader),
                # ADDED RESOURCE MONITORING:
                resource_stats=resource_stats
            )
            
            # Generate samples for visualization
            if (epoch + 1) % 5 == 0 or epoch == 0:
                self.save_progress_samples(fixed_noise, epoch + 1)
            
            # Calculate epoch average losses
            avg_g_loss = sum(self.g_losses[-len(dataloader):]) / len(dataloader)
            avg_d_loss = sum(self.d_losses[-len(dataloader):]) / len(dataloader)
            
            # Handle checkpoints per EPOCH according to metrics-checkpoints-how-to.txt
            # This should be called once per epoch with average losses
            self.handle_checkpoints(epoch + 1, avg_g_loss, avg_d_loss)
            
            # Print epoch summary
            print(f"Epoch [{epoch+1}/{self.epochs}] completed in {epoch_duration:.2f}min - "
                  f"Avg G_loss: {avg_g_loss:.4f}, Avg D_loss: {avg_d_loss:.4f}")
        
        # Training completed
        total_duration = (time.time() - total_start_time) / 3600  # hours
        print(f"\n🎉 Training completed in {total_duration:.2f} hours!")
        
        # Stop resource monitoring and save data
        self.resource_monitor.stop_monitoring()
        monitoring_path = os.path.join(self.checkpoint_manager.checkpoint_dir, "resource_monitoring.json")
        self.resource_monitor.save_monitoring_data(monitoring_path)
        
        # Print resource summary
        resource_summary = self.resource_monitor.get_training_summary()
        print(f"📊 Resource usage summary: {resource_summary.get('alerts', {}).get('total_count', 0)} alerts")
        
        # Print bottleneck analysis
        bottleneck_analysis = self.resource_monitor.get_bottleneck_analysis()
        if bottleneck_analysis.get('bottlenecks_detected', 0) > 0:
            print(f"⚠️ Performance bottlenecks detected: {bottleneck_analysis['bottlenecks_detected']}")
        
        # Save final checkpoint using new API
        final_checkpoint = self.checkpoint_manager.save_final_checkpoint(
            self.generator, self.discriminator,
            self.g_optimizer, self.d_optimizer,
            self.epochs, self.global_iteration,
            self.g_losses[-1], self.d_losses[-1]
        )
        
        print(f"💾 Final checkpoint saved: {final_checkpoint}")
        
        # Generate comprehensive post-training analysis
        self.post_training_analysis()
    
    def handle_checkpoints(self, epoch, generator_loss, discriminator_loss):
        """Handle checkpoint saving using new rolling buffer strategy"""
        
        # Use new checkpoint API - save_checkpoint handles the logic internally
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            self.generator, self.discriminator,
            self.g_optimizer, self.d_optimizer,
            epoch, self.global_iteration,
            generator_loss, discriminator_loss
        )
        
        # If checkpoint was saved, log metrics
        if checkpoint_path:
            print(f"💾 Checkpoint saved: {checkpoint_path}")
            
            # Check if this was a best model checkpoint
            if generator_loss < self.checkpoint_manager.best_generator_loss:
                print(f"🏆 New best model saved! Generator loss: {generator_loss:.4f}")
            
            # LOG CHECKPOINT METRICS per metrics-checkpoints-how-to.txt requirements
            self.metrics.log_checkpoint_metrics(
                checkpoint_type="regular" if epoch % self.checkpoint_manager.checkpoint_frequency == 0 else "best",
                epoch=epoch, 
                iteration=self.global_iteration,
                generator=self.generator,
                discriminator=self.discriminator,
                real_samples=None,  # Optional parameter for image quality assessment
                checkpoint_file_path=checkpoint_path  # Path to saved checkpoint
            )
    
    def save_progress_samples(self, fixed_noise, epoch):
        """Save generated samples for progress tracking"""
        self.generator.eval()
        
        with torch.no_grad():
            fake_samples = self.generator(fixed_noise)
            
            # Create output directory - FIXED: use config-based path
            if self.config and hasattr(self.config, 'samples_dir'):
                samples_dir = self.config.samples_dir
            elif self.config and hasattr(self.config, 'output_dir'):
                samples_dir = os.path.join(self.config.output_dir, "samples")
            else:
                samples_dir = "output_dcgan/samples"  # Fallback
            os.makedirs(samples_dir, exist_ok=True)
            
            # Save samples as numpy arrays (.npy format)
            self.save_samples_npy(
                fake_samples, 
                os.path.join(samples_dir, f"epoch_{epoch:03d}.npy"),
                epoch=epoch
            )
        
        self.generator.train()
    
    def save_samples_npy(self, samples, save_path, epoch=None):
        """Save generated samples as numpy arrays in .npy format"""
        import numpy as np
        
        # Convert samples to numpy format
        # Shape: (batch_size, channels, height, width)
        samples_np = samples.cpu().detach().numpy()
        
        # Denormalize from [-1, 1] to [0, 1] 
        samples_np = (samples_np + 1.0) / 2.0
        samples_np = np.clip(samples_np, 0.0, 1.0)
        
        # Save the numpy array
        np.save(save_path, samples_np)
        
        # Print info about saved samples
        batch_size, channels, height, width = samples_np.shape
        print(f"💾 Saved {batch_size} samples to: {save_path}")
        print(f"   Shape: {samples_np.shape} (batch, channels, height, width)")
        print(f"   Value range: [{samples_np.min():.3f}, {samples_np.max():.3f}]")
        if epoch is not None:
            print(f"   Epoch: {epoch}")
    
    def save_image_grid(self, images, save_path, title="Generated Images", nrow=4):
        """Save a grid of images"""
        import math
        
        batch_size = images.size(0)
        ncol = min(nrow, batch_size)
        nrow_actual = math.ceil(batch_size / ncol)
        
        fig, axes = plt.subplots(nrow_actual, ncol, figsize=(ncol*2, nrow_actual*2))
        
        # Handle different axes structures
        if nrow_actual == 1 and ncol == 1:
            axes = [axes]
        elif nrow_actual == 1:
            axes = axes.tolist() if hasattr(axes, 'tolist') else list(axes)
        else:
            axes = axes.flatten().tolist() if hasattr(axes, 'flatten') else axes.flatten()
        
        for i in range(batch_size):
            img = images[i].cpu().squeeze()
            img = (img + 1) / 2  # Denormalize from [-1,1] to [0,1]
            img = torch.clamp(img, 0, 1)
            
            # Handle different image formats
            if img.dim() == 3:  # Color image (C, H, W)
                if img.shape[0] == 3:  # RGB
                    img = img.permute(1, 2, 0)  # Convert to (H, W, C)
                elif img.shape[0] == 1:  # Grayscale with channel dim
                    img = img.squeeze(0)  # Remove channel dimension
            elif img.dim() == 2:  # Already grayscale (H, W)
                pass
            else:
                raise ValueError(f"Unexpected image dimensions: {img.shape}")
            
            # Get the correct axis
            if isinstance(axes, list):
                ax = axes[i]
            else:
                ax = axes
                
            # Use appropriate colormap
            if img.dim() == 3:  # Color image
                ax.imshow(img.numpy())
            else:  # Grayscale
                ax.imshow(img.numpy(), cmap='gray')
            ax.axis('off')
        
        # Hide unused subplots
        if isinstance(axes, list):
            for i in range(batch_size, len(axes)):
                axes[i].axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def post_training_analysis(self):
        """Generate comprehensive post-training analysis using new module"""
        print("\n📈 Generating comprehensive training analysis...")
        
        try:
            # Import new post-training analyzer
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
            from analysis_and_statistics.base.post_training_analyzer import generate_training_report
            
            # Use the convenient function for full analysis - FIXED: use config-based path
            # This automatically generates all plots and reports
            if self.config and hasattr(self.config, 'output_dir'):
                output_analysis_dir = self.config.output_dir
            else:
                output_analysis_dir = "output_dcgan"  # Fallback
                
            analysis_results = generate_training_report(
                output_analysis_dir=output_analysis_dir
            )
            
            print(f"📊 Post-training analysis completed successfully!")
            print(f"📈 All plots and reports generated in '{output_analysis_dir}/plots'")
            print(f"📝 Training summary saved as JSON report")
            
            if analysis_results and 'performance_score' in analysis_results:
                print(f"🏆 Training Performance Score: {analysis_results['performance_score']}")
                print(f"📉 Best Generator Loss: {analysis_results.get('best_losses', {}).get('generator', 'N/A')}")
            
            return analysis_results
            
        except Exception as e:
            print(f"❌ Error in post-training analysis: {e}")
            print("💡 Falling back to basic analysis...")
            
            # Fallback: basic summary
            total_epochs = self.epochs
            final_g_loss = self.g_losses[-1] if self.g_losses else 0
            final_d_loss = self.d_losses[-1] if self.d_losses else 0
            
            print(f"📊 Basic Training Summary:")
            print(f"   Total epochs: {total_epochs}")
            print(f"   Final Generator Loss: {final_g_loss:.4f}")
            print(f"   Final Discriminator Loss: {final_d_loss:.4f}")
            print(f"   Total iterations: {self.global_iteration}")
            
            return None
    
    def generate_samples(self, num_samples=16, save_path=None):
        """Generate and optionally save sample images"""
        self.generator.eval()
        
        with torch.no_grad():
            noise = torch.randn(num_samples, self.latent_dim, 1, 1, device=self.device)
            generated_samples = self.generator(noise)
            
            if save_path:
                # Determine format based on file extension
                if save_path.endswith('.npy'):
                    self.save_samples_npy(
                        generated_samples, 
                        save_path
                    )
                else:
                    # Default to image grid for non-.npy extensions
                    self.save_image_grid(
                        generated_samples, 
                        save_path,
                        title=f"Generated Samples ({num_samples} images)"
                    )
                print(f"💾 Samples saved to: {save_path}")
            
            return generated_samples
    
    def load_checkpoint(self, checkpoint_path):
        """Load a checkpoint and resume training state"""
        context = self.checkpoint_manager.load_checkpoint(
            checkpoint_path,
            self.generator,
            self.discriminator,
            self.g_optimizer,
            self.d_optimizer
        )
        
        # Update training state
        self.global_iteration = context.get('iteration', 0)
        
        print(f"📥 Checkpoint loaded successfully!")
        print(f"   Resumed from epoch {context['epoch']}, iteration {context['iteration']}")
        
        return context
    
    def get_training_info(self):
        """Get current training information"""
        return {
            'global_iteration': self.global_iteration,
            'generator_params': self.g_params,
            'discriminator_params': self.d_params,
            'total_params': self.g_params + self.d_params,
            'current_losses': {
                'generator': self.g_losses[-1] if self.g_losses else None,
                'discriminator': self.d_losses[-1] if self.d_losses else None
            },
            'checkpoints_info': self.checkpoint_manager.get_checkpoint_info()
        }
    
    def _calculate_convergence_indicator(self):
        """Calculate convergence indicator based on loss variance"""
        if len(self.g_losses) < 10:
            return 0.0
        
        # Calculate variance of last 10 generator losses
        recent_g_losses = self.g_losses[-10:]
        variance = torch.var(torch.tensor(recent_g_losses)).item()
        
        # Normalize by mean loss to get relative variance
        mean_loss = torch.mean(torch.tensor(recent_g_losses)).item()
        if mean_loss != 0:
            convergence_indicator = variance / abs(mean_loss)
        else:
            convergence_indicator = variance
            
        return convergence_indicator
    
    def _calculate_stability_score(self):
        """Calculate training stability score based on gradient consistency"""
        if not hasattr(self, '_recent_grad_norms') or len(self._recent_grad_norms) < 5:
            return 0.0
        
        # Use coefficient of variation for gradient norm consistency
        recent_norms = self._recent_grad_norms[-5:]
        if len(recent_norms) < 2:
            return 0.0
            
        mean_norm = sum(recent_norms) / len(recent_norms)
        if mean_norm == 0:
            return 0.0
            
        variance = sum((norm - mean_norm) ** 2 for norm in recent_norms) / len(recent_norms)
        std_dev = variance ** 0.5
        
        # Lower coefficient of variation = higher stability
        coefficient_of_variation = std_dev / mean_norm
        stability_score = max(0.0, 1.0 - coefficient_of_variation)
        
        return stability_score
    
    def _calculate_performance_score(self, g_loss, d_loss):
        """Calculate model performance score for checkpoints"""
        # Simple performance metric based on loss balance
        # Good performance when losses are balanced and relatively low
        
        if g_loss <= 0 or d_loss <= 0:
            return 0.0
        
        # Ideal balance is when losses are close to each other
        loss_ratio = min(g_loss, d_loss) / max(g_loss, d_loss)
        
        # Performance decreases with higher losses
        avg_loss = (g_loss + d_loss) / 2
        loss_penalty = min(1.0, 2.0 / (1 + avg_loss))  # Higher loss = lower score
        
        performance_score = loss_ratio * loss_penalty
        return performance_score
    
    def _get_latest_sample_paths(self):
        """Get paths to latest generated sample files (both .npy and .png)"""
        import glob
        
        # Look for sample files in samples directory using config
        from .config import SAMPLES_DIR
        if not os.path.exists(SAMPLES_DIR):
            return []
        
        # Find latest sample files - prioritize .npy files
        patterns = [
            os.path.join(SAMPLES_DIR, "epoch_*.npy"),  # Primary: numpy arrays
            os.path.join(SAMPLES_DIR, "epoch_*.png")   # Secondary: images
        ]
        
        sample_files = []
        for pattern in patterns:
            files = glob.glob(pattern)
            sample_files.extend(files)
        
        # Return up to 5 most recent files
        sample_files.sort(reverse=True)
        return sample_files[:5]
