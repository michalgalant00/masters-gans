"""
Base Metrics Collector
======================

Domain-agnostic metrics collection and logging functionality.
Common to all GAN architectures (WaveGAN, DCGAN, etc.)

Implements optimized structure from metrics-checkpoints-how-to.txt:
- Per epoch statistics (epoch_statistics.csv)
- Per checkpoint metrics (checkpoint_metrics.csv) 
- Detailed iteration snapshots (single_epochs_statistics/epoch_N/)
- Experiment configuration (experiment_config.json)
"""

import os
import csv
import json
import time
import gzip
import torch
import numpy as np
from datetime import datetime
from collections import deque
from typing import Dict, List, Optional, Any, Tuple
from abc import ABC, abstractmethod

# Try to import psutil for system monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False
    print("⚠️  psutil not available - system monitoring will be limited")


class MetricsCollectorBase(ABC):
    """
    Base class for metrics collection across different GAN architectures
    
    Optimized structure:
    output_analysis/
      epochs_statistics/
        epoch_statistics.csv (główne statystyki per epoka)
        checkpoint_metrics.csv (jakość modelu per checkpoint)
        experiment_config.json (konfiguracja + model info)
      single_epochs_statistics/
        epoch_1/epoch_1_iterations.csv
        epoch_2/epoch_2_iterations.csv
        ...
    """
    
    def __init__(self, output_dir: str = "output_analysis", model_type: str = "base", config: Optional[Dict] = None):
        self.output_dir = output_dir
        self.model_type = model_type
        self.config = config or {}
        
        # Optimized file paths according to new structure
        self.epochs_stats_dir = os.path.join(output_dir, "epochs_statistics")
        self.single_epochs_dir = os.path.join(output_dir, "single_epochs_statistics")
        
        # Setup directories after paths are defined
        self.setup_directories()
        
        self.epoch_statistics_path = os.path.join(self.epochs_stats_dir, "epoch_statistics.csv")
        self.checkpoint_metrics_path = os.path.join(self.epochs_stats_dir, "checkpoint_metrics.csv")
        self.experiment_config_path = os.path.join(self.epochs_stats_dir, "experiment_config.json")
        
        # Buffers for statistical calculations
        self.loss_buffer = deque(maxlen=10)  # Last 10 epochs for convergence analysis
        self.grad_buffer = deque(maxlen=10)  # Last 10 iterations for stability
        self.current_epoch_iterations = []  # Current epoch iteration stats
        
        # Performance tracking
        self.best_generator_loss = float('inf')
        
        # Configuration-based frequencies (zgodnie z metrics-checkpoints-how-to.txt)
        metrics_config = self.config.get('metrics_and_checkpoints', {})
        self.iteration_snapshot_frequency = metrics_config.get('csv_write_frequency', 100)  # Snapshot co N iteracji
        self.epoch_cleanup_threshold = 50  # Auto-cleanup epok starszych niż 50
        self.start_time = time.time()
        
        # Initialize CSV files with domain-specific headers
        self._initialize_csv_files()
        
        print(f"📊 {model_type.upper()} Metrics collector initialized - output: {output_dir}")
    
    def setup_directories(self):
        """Create optimized directory structure according to metrics-checkpoints-how-to.txt"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.epochs_stats_dir, exist_ok=True)
        os.makedirs(self.single_epochs_dir, exist_ok=True)
        
        print(f"📁 Metrics directories created: epochs_statistics, single_epochs_statistics")
    
    @abstractmethod
    def _get_training_headers(self) -> List[str]:
        """Get domain-specific training log headers - DEPRECATED in new structure"""
        # This method is kept for backward compatibility but not used in optimized structure
        pass
    
    @abstractmethod 
    def _get_epoch_headers(self) -> List[str]:
        """Get domain-specific epoch summary headers (should combine base + domain-specific)"""
        pass
    
    @abstractmethod
    def _get_checkpoint_headers(self) -> List[str]:
        """Get domain-specific checkpoint metrics headers (should combine base + domain-specific)"""
        pass
    
    @abstractmethod
    def _get_domain_epoch_data(self, domain_kwargs: Dict) -> List:
        """Get domain-specific epoch summary data (e.g., FAD for audio, FID for images)"""
        pass
    
    def _get_base_epoch_headers(self) -> List[str]:
        """
        Get optimized epoch statistics headers according to metrics-checkpoints-how-to.txt
        PER EPOKA (EPOCH_STATISTICS.CSV)
        """
        return [
            "epoch", "timestamp",
            "avg_generator_loss", "avg_discriminator_loss",
            "min_generator_loss", "min_discriminator_loss", 
            "max_generator_loss", "max_discriminator_loss",
            "avg_grad_norm_g", "avg_grad_norm_d",
            "max_grad_norm_g", "max_grad_norm_d",
            "learning_rate_g", "learning_rate_d",
            "gradient_penalty_value", "wasserstein_distance_estimate",
            "gpu_memory_used", "gpu_utilization", "cpu_usage", "ram_usage",
            "epoch_duration_minutes",
            "convergence_indicator", "training_stability_score",
            "batch_size", "sequence_length", "total_iterations_in_epoch"
        ]
    
    def _get_base_checkpoint_headers(self) -> List[str]:
        """
        Get optimized checkpoint metrics headers according to metrics-checkpoints-how-to.txt
        PER CHECKPOINT (CHECKPOINT_METRICS.CSV)
        """
        return [
            "checkpoint_type", "epoch", "iteration", "timestamp",
            "generator_params_count", "discriminator_params_count"
        ]
    
    def _get_base_iteration_headers(self) -> List[str]:
        """
        Get iteration snapshot headers for single_epochs_statistics
        SINGLE EPOCH DETAILS (PER EPOCH FOLDER)
        """
        return [
            "iteration", "generator_loss", "discriminator_loss",
            "generator_grad_norm", "discriminator_grad_norm",
            "gpu_memory_used", "iteration_time_seconds", "timestamp"
        ]
    
    def _get_base_training_headers(self) -> List[str]:
        """Get common training headers used by all domains"""
        return [
            "epoch", "iteration", "timestamp",
            "generator_loss", "discriminator_loss",
            "generator_grad_norm", "discriminator_grad_norm",
            "learning_rate_g", "learning_rate_d",
            "gpu_memory_used", "gpu_utilization", "cpu_usage", "ram_usage",
            "batch_size", "iteration_time_seconds",
            "loss_variance_last_10", "gradient_variance_last_10"
        ]
    
    def _initialize_csv_files(self):
        """Initialize CSV files with optimized headers according to metrics-checkpoints-how-to.txt"""
        # Get domain-specific headers (combined with base headers)
        epoch_headers = self._get_epoch_headers()  # Should include base + domain-specific
        checkpoint_headers = self._get_checkpoint_headers()  # Should include base + domain-specific
        
        # Initialize epoch statistics CSV
        if not os.path.exists(self.epoch_statistics_path):
            with open(self.epoch_statistics_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(epoch_headers)
        
        # Initialize checkpoint metrics CSV
        if not os.path.exists(self.checkpoint_metrics_path):
            with open(self.checkpoint_metrics_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(checkpoint_headers)
    
    def _get_system_metrics(self) -> Tuple[float, float, float, float]:
        """Get current system resource usage"""
        gpu_memory = 0.0
        gpu_util = 0.0
        cpu_usage = 0.0
        ram_usage = 0.0
        
        try:
            # GPU metrics
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
                gpu_util = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0.0
            
            # CPU and RAM metrics
            if PSUTIL_AVAILABLE and psutil is not None:
                cpu_usage = psutil.cpu_percent()
                ram_usage = psutil.virtual_memory().percent
                
        except Exception as e:
            print(f"⚠️  Error getting system metrics: {e}")
        
        return gpu_memory, gpu_util, cpu_usage, ram_usage
    
    def _calculate_loss_variance(self) -> float:
        """Calculate variance of recent losses"""
        if len(self.loss_buffer) < 2:
            return 0.0
        
        losses = [sum(loss_pair) for loss_pair in self.loss_buffer]
        return float(np.var(losses))
    
    def _calculate_gradient_variance(self) -> float:
        """Calculate variance of recent gradient norms"""
        if len(self.grad_buffer) < 2:
            return 0.0
        
        grads = [sum(grad_pair) for grad_pair in self.grad_buffer]
        return float(np.var(grads))
    
    def _calculate_convergence_indicator(self, epoch_stats: List[Dict]) -> float:
        """Calculate convergence indicator based on loss stability"""
        if len(epoch_stats) < 5:  # Need at least 5 iterations
            return 0.0
        
        recent_gen_losses = [stat['generator_loss'] for stat in epoch_stats[-5:]]
        loss_trend = np.polyfit(range(len(recent_gen_losses)), recent_gen_losses, 1)[0]
        
        # Convergence indicator: negative trend (decreasing loss) is good
        return max(0.0, min(1.0, -loss_trend))
    
    def _calculate_training_stability(self, epoch_stats: List[Dict]) -> float:
        """Calculate training stability score"""
        if len(epoch_stats) < 10:
            return 0.5  # Neutral score for early training
        
        # Calculate coefficient of variation for losses
        gen_losses = [stat['generator_loss'] for stat in epoch_stats]
        disc_losses = [stat['discriminator_loss'] for stat in epoch_stats]
        
        gen_cv = np.std(gen_losses) / (np.mean(gen_losses) + 1e-8)
        disc_cv = np.std(disc_losses) / (np.mean(disc_losses) + 1e-8)
        
        # Lower coefficient of variation = higher stability
        stability = 1.0 / (1.0 + (gen_cv + disc_cv) / 2.0)
        return min(1.0, max(0.0, float(stability)))
    
    def _calculate_convergence_indicator_from_buffer(self) -> float:
        """Calculate convergence indicator from loss buffer (last 10 epochs)"""
        if len(self.loss_buffer) < 5:
            return 0.0
        
        # Use variance of last losses - lower variance indicates convergence
        loss_variance = np.var(list(self.loss_buffer))
        return max(0.0, min(1.0, float(1.0 / (1.0 + loss_variance))))
    
    def _calculate_training_stability_from_buffer(self) -> float:
        """Calculate training stability from gradient buffer"""
        if len(self.grad_buffer) < 5:
            return 0.5
        
        # Use gradient consistency - lower variance indicates stability
        grad_variance = np.var(list(self.grad_buffer))
        return max(0.0, min(1.0, float(1.0 / (1.0 + grad_variance))))
    
    def log_base_iteration_stats(self, epoch: int, iteration: int,
                                generator_loss: float, discriminator_loss: float,
                                generator_grad_norm: float, discriminator_grad_norm: float,
                                learning_rate_g: float, learning_rate_d: float,
                                batch_size: int, iteration_time: float,
                                **domain_specific_kwargs) -> List:
        """Log base iteration statistics common to all domains"""
        
        timestamp = datetime.now().isoformat()
        
        # Update buffers for variance calculations
        self.loss_buffer.append((generator_loss, discriminator_loss))
        self.grad_buffer.append((generator_grad_norm, discriminator_grad_norm))
        
        # Calculate metrics
        loss_variance = self._calculate_loss_variance()
        gradient_variance = self._calculate_gradient_variance()
        
        # Get system metrics
        gpu_memory, gpu_util, cpu_usage, ram_usage = self._get_system_metrics()
        
        # Prepare base row data
        base_row_data = [
            epoch, iteration, timestamp,
            generator_loss, discriminator_loss,
            generator_grad_norm, discriminator_grad_norm,
            learning_rate_g, learning_rate_d,
            gpu_memory, gpu_util, cpu_usage, ram_usage,
            batch_size, iteration_time,
            loss_variance, gradient_variance
        ]
        
        # Store stats for epoch summary
        epoch_stat = {
            'generator_loss': generator_loss,
            'discriminator_loss': discriminator_loss,
            'generator_grad_norm': generator_grad_norm,
            'discriminator_grad_norm': discriminator_grad_norm,
            **domain_specific_kwargs
        }
        
        return base_row_data
    def log_epoch_stats(self, epoch: int, epoch_losses: Dict, epoch_grads: Dict, 
                       epoch_lr: Dict, epoch_gpu_stats: Dict, epoch_timer: float,
                       **domain_specific_kwargs):
        """
        Log głównych statystyk per epoka do epochs_statistics/epoch_statistics.csv
        Append mode - jedna linijka per epoka
        
        Args:
            epoch: Current epoch number
            epoch_losses: {'avg_g': float, 'avg_d': float, 'min_g': float, 'min_d': float, 'max_g': float, 'max_d': float}
            epoch_grads: {'avg_g': float, 'avg_d': float, 'max_g': float, 'max_d': float}
            epoch_lr: {'lr_g': float, 'lr_d': float}
            epoch_gpu_stats: {'gpu_memory': float, 'gpu_util': float, 'cpu_usage': float, 'ram_usage': float}
            epoch_timer: Duration in minutes
            **domain_specific_kwargs: Domain-specific metrics (e.g., FAD for audio, FID for images)
        """
        timestamp = datetime.now().isoformat()
        
        # Calculate convergence and stability indicators
        convergence_indicator = self._calculate_convergence_indicator_from_buffer()
        stability_score = self._calculate_training_stability_from_buffer()
        
        # Base epoch data according to metrics-checkpoints-how-to.txt
        base_epoch_data = [
            epoch, timestamp,
            epoch_losses['avg_g'], epoch_losses['avg_d'],
            epoch_losses['min_g'], epoch_losses['min_d'], 
            epoch_losses['max_g'], epoch_losses['max_d'],
            epoch_grads['avg_g'], epoch_grads['avg_d'],
            epoch_grads['max_g'], epoch_grads['max_d'],
            epoch_lr['lr_g'], epoch_lr['lr_d'],
            domain_specific_kwargs.get('gradient_penalty_value', 0.0),
            domain_specific_kwargs.get('wasserstein_distance_estimate', 0.0),
            epoch_gpu_stats['gpu_memory'], epoch_gpu_stats['gpu_util'],
            epoch_gpu_stats['cpu_usage'], epoch_gpu_stats['ram_usage'],
            epoch_timer,
            convergence_indicator, stability_score,
            domain_specific_kwargs.get('batch_size', 0),
            domain_specific_kwargs.get('sequence_length', 0),
            domain_specific_kwargs.get('total_iterations', 0)
        ]
        
        # Get domain-specific epoch data
        domain_epoch_data = self._get_domain_epoch_data(domain_specific_kwargs)
        
        # Combine and write to CSV
        epoch_row_data = base_epoch_data + domain_epoch_data
        
        with open(self.epoch_statistics_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(epoch_row_data)
        
        # Update best loss tracking
        if epoch_losses['avg_g'] < self.best_generator_loss:
            self.best_generator_loss = epoch_losses['avg_g']
        
        print(f"📊 Epoch {epoch} stats logged - avg G loss: {epoch_losses['avg_g']:.4f}")
        
    def log_iteration_snapshot(self, epoch: int, iteration: int, losses: Dict, grads: Dict, 
                              gpu_stats: Dict, timestamp: Optional[str] = None):
        """
        Log snapshot co 100 iteracji do single_epochs_statistics/epoch_N/epoch_N_iterations.csv
        Tworzenie katalogu per epoka i CSV z szczegółami iteracji
        
        Args:
            epoch: Current epoch number  
            iteration: Current iteration number
            losses: {'generator_loss': float, 'discriminator_loss': float}
            grads: {'generator_grad_norm': float, 'discriminator_grad_norm': float}
            gpu_stats: {'gpu_memory_used': float, 'iteration_time_seconds': float}
            timestamp: Optional timestamp, if not provided will use current time
        """
        if iteration % self.iteration_snapshot_frequency != 0:
            return  # Only snapshot every N iterations
            
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        # Create epoch-specific directory
        epoch_dir = os.path.join(self.single_epochs_dir, f"epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        
        # Epoch-specific CSV file
        epoch_iterations_csv = os.path.join(epoch_dir, f"epoch_{epoch}_iterations.csv")
        
        # Initialize CSV with headers if it doesn't exist
        if not os.path.exists(epoch_iterations_csv):
            with open(epoch_iterations_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self._get_base_iteration_headers())
        
        # Write iteration data
        iteration_data = [
            iteration,
            losses['generator_loss'], losses['discriminator_loss'],
            grads['generator_grad_norm'], grads['discriminator_grad_norm'],
            gpu_stats['gpu_memory_used'], gpu_stats['iteration_time_seconds'],
            timestamp
        ]
        
        with open(epoch_iterations_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(iteration_data)
            
        # Also store in current epoch buffer for aggregation
        self.current_epoch_iterations.append({
            'iteration': iteration,
            'generator_loss': losses['generator_loss'],
            'discriminator_loss': losses['discriminator_loss'],
            'generator_grad_norm': grads['generator_grad_norm'],
            'discriminator_grad_norm': grads['discriminator_grad_norm'],
            'gpu_memory_used': gpu_stats['gpu_memory_used'],
            'iteration_time_seconds': gpu_stats['iteration_time_seconds'],
            'timestamp': timestamp
        })
    
    def aggregate_epoch_metrics(self) -> Dict:
        """
        Agregacja metryk z iteracji w epoce (avg, min, max)
        Kalkulacja convergence indicators i stability scores
        
        Returns:
            Dict containing aggregated metrics for the epoch
        """
        if not self.current_epoch_iterations:
            return {}
        
        # Extract metrics from iterations
        g_losses = [item['generator_loss'] for item in self.current_epoch_iterations]
        d_losses = [item['discriminator_loss'] for item in self.current_epoch_iterations]
        g_grads = [item['generator_grad_norm'] for item in self.current_epoch_iterations]
        d_grads = [item['discriminator_grad_norm'] for item in self.current_epoch_iterations]
        
        aggregated = {
            'avg_g': np.mean(g_losses), 'avg_d': np.mean(d_losses),
            'min_g': np.min(g_losses), 'min_d': np.min(d_losses),
            'max_g': np.max(g_losses), 'max_d': np.max(d_losses),
            'avg_g_grad': np.mean(g_grads), 'avg_d_grad': np.mean(d_grads),
            'max_g_grad': np.max(g_grads), 'max_d_grad': np.max(d_grads),
            'total_iterations': len(self.current_epoch_iterations)
        }
        
        # Clear current epoch buffer for next epoch
        self.current_epoch_iterations.clear()
        
        return aggregated
    
    def cleanup_old_epochs(self, current_epoch: int):
        """Auto-cleanup pojedynczych epok starszych niż threshold (configurable)"""
        if current_epoch < self.epoch_cleanup_threshold:
            return
            
        cleanup_epoch = current_epoch - self.epoch_cleanup_threshold
        cleanup_dir = os.path.join(self.single_epochs_dir, f"epoch_{cleanup_epoch}")
        
        if os.path.exists(cleanup_dir):
            import shutil
            shutil.rmtree(cleanup_dir)
            print(f"🧹 Cleaned up old epoch {cleanup_epoch} statistics")
    
    def save_experiment_config(self, config_dict: Dict[str, Any]):
        """Save experiment configuration to JSON"""
        config_dict['experiment_start_time'] = datetime.now().isoformat()
        config_dict['model_type'] = self.model_type
        
        with open(self.experiment_config_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        print(f"💾 Experiment config saved to {self.experiment_config_path}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training progress from epoch statistics"""
        try:
            import pandas as pd
            df = pd.read_csv(self.epoch_statistics_path)
            
            return {
                'total_epochs': len(df),
                'current_epoch': df['epoch'].max() if not df.empty else 0,
                'best_generator_loss': df['avg_generator_loss'].min() if not df.empty else float('inf'),
                'average_epoch_duration': df['epoch_duration_minutes'].mean() if not df.empty else 0,
                'training_duration_hours': (time.time() - self.start_time) / 3600
            }
        except Exception as e:
            print(f"⚠️  Error generating training summary: {e}")
            return {}
