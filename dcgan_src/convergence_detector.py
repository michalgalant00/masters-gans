"""
Real-time Convergence Detection for GANs
========================================

Early stopping based on training metrics, loss patterns, and convergence indicators.
Supports both DCGAN and WaveGAN architectures.
"""

import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
from collections import deque
import pickle
import os
from dataclasses import dataclass
from enum import Enum


class ConvergenceStatus(Enum):
    """Convergence status enumeration"""
    TRAINING = "training"
    CONVERGED = "converged"
    DIVERGED = "diverged"
    OSCILLATING = "oscillating"
    EARLY_STOP = "early_stop"


@dataclass
class ConvergenceConfig:
    """Configuration for convergence detection"""
    # Core settings
    enabled: bool = True  # Enable/disable convergence detection
    min_iterations: int = 100  # Minimum iterations before checking convergence
    check_frequency: int = 25  # Check convergence every N iterations
    
    # Early stopping
    patience: int = 100  # Epochs to wait for improvement (increased from 20)
    min_improvement: float = 0.0005  # Minimum improvement threshold (reduced from 0.001)
    
    # Loss monitoring
    loss_window_size: int = 15  # Window size for loss analysis (increased from 10)
    loss_variance_threshold: float = 0.5  # Maximum loss variance (increased from 0.1)
    loss_trend_threshold: float = 0.1  # Trend detection threshold (increased from 0.05)
    
    # Gradient monitoring
    grad_norm_threshold: float = 1000.0  # Maximum gradient norm (increased from 100.0)
    grad_variance_threshold: float = 10.0  # Gradient variance threshold (increased from 1.0)
    
    # Divergence detection
    loss_explosion_threshold: float = 100.0  # Loss explosion threshold (increased from 10.0)
    nan_detection: bool = True  # Detect NaN/Inf values
    
    # Oscillation detection
    oscillation_window: int = 12  # Window for oscillation detection (increased from 8)
    oscillation_threshold: float = 0.6  # Oscillation detection threshold (increased from 0.3)
    
    # Convergence criteria
    convergence_window: int = 25  # Window for convergence detection (increased from 15)
    convergence_threshold: float = 0.005  # Convergence threshold (reduced from 0.02)
    
    # Save/load
    save_state: bool = True  # Save convergence state
    state_file: str = "convergence_state.pkl"  # State file name


class ConvergenceDetector:
    """Real-time convergence detection for GAN training"""
    
    def __init__(self, config: Optional[ConvergenceConfig] = None, output_dir: str = ""):
        """
        Initialize convergence detector
        
        Args:
            config: Convergence configuration
            output_dir: Directory for saving state
        """
        self.config = config or ConvergenceConfig()
        self.output_dir = output_dir
        
        # Training history
        self.generator_losses: deque = deque(maxlen=200)  # Increased buffer
        self.discriminator_losses: deque = deque(maxlen=200)  # Increased buffer
        self.gradient_norms: deque = deque(maxlen=100)  # Increased buffer
        self.learning_rates: deque = deque(maxlen=100)  # Increased buffer
        
        # Convergence tracking
        self.best_loss: float = float('inf')
        self.best_epoch: int = 0
        self.patience_counter: int = 0
        self.convergence_status: ConvergenceStatus = ConvergenceStatus.TRAINING
        
        # Additional metrics
        self.fid_scores: deque = deque(maxlen=50)
        self.inception_scores: deque = deque(maxlen=50)
        self.stability_scores: deque = deque(maxlen=50)
        
        # State management
        self.epoch_count: int = 0
        self.iteration_count: int = 0  # Track iterations for min_iterations check
        self.last_check_epoch: int = 0
        
        # Load previous state if exists
        if self.config.save_state:
            self._load_state()
    
    def update(self, epoch: int, generator_loss: float, discriminator_loss: float,
               grad_norm_g: float = 0.0, grad_norm_d: float = 0.0,
               learning_rate: float = 0.0, **kwargs) -> ConvergenceStatus:
        """
        Update convergence detector with new metrics
        
        Args:
            epoch: Current epoch
            generator_loss: Generator loss
            discriminator_loss: Discriminator loss
            grad_norm_g: Generator gradient norm
            grad_norm_d: Discriminator gradient norm
            learning_rate: Current learning rate
            **kwargs: Additional metrics (fid_score, inception_score, etc.)
            
        Returns:
            Current convergence status
        """
        self.epoch_count = epoch
        self.iteration_count += 1
        
        # Return early if convergence detection is disabled
        if not self.config.enabled:
            return ConvergenceStatus.TRAINING
        
        # Skip convergence detection in early iterations
        if self.iteration_count < self.config.min_iterations:
            self.generator_losses.append(generator_loss)
            self.discriminator_losses.append(discriminator_loss)
            self.gradient_norms.append((grad_norm_g + grad_norm_d) / 2)
            self.learning_rates.append(learning_rate)
            return ConvergenceStatus.TRAINING
        
        # Check convergence only at specified intervals to reduce overhead
        if self.iteration_count % self.config.check_frequency != 0:
            self.generator_losses.append(generator_loss)
            self.discriminator_losses.append(discriminator_loss)
            self.gradient_norms.append((grad_norm_g + grad_norm_d) / 2)
            self.learning_rates.append(learning_rate)
            return self.convergence_status
        
        # Store metrics
        self.generator_losses.append(generator_loss)
        self.discriminator_losses.append(discriminator_loss)
        self.gradient_norms.append((grad_norm_g + grad_norm_d) / 2)
        self.learning_rates.append(learning_rate)
        
        # Store additional metrics if provided
        if 'fid_score' in kwargs:
            self.fid_scores.append(kwargs['fid_score'])
        if 'inception_score' in kwargs:
            self.inception_scores.append(kwargs['inception_score'])
        if 'stability_score' in kwargs:
            self.stability_scores.append(kwargs['stability_score'])
        
        # Check for immediate problems
        if self._check_divergence(generator_loss, discriminator_loss):
            self.convergence_status = ConvergenceStatus.DIVERGED
            return self.convergence_status
        
        # Check for NaN/Inf
        if self.config.nan_detection and self._check_nan_inf(generator_loss, discriminator_loss):
            self.convergence_status = ConvergenceStatus.DIVERGED
            return self.convergence_status
        
        # Periodic convergence checks (every few epochs to avoid overhead)
        if epoch - self.last_check_epoch >= 5:  # Increased from 3 to 5
            self._perform_convergence_analysis()
            self.last_check_epoch = epoch
        
        # Save state periodically
        if self.config.save_state and epoch % 20 == 0:  # Increased from 10 to 20
            self._save_state()
        
        return self.convergence_status
    
    def _check_divergence(self, g_loss: float, d_loss: float) -> bool:
        """Check for immediate divergence indicators"""
        # Loss explosion
        if (g_loss > self.config.loss_explosion_threshold or 
            d_loss > self.config.loss_explosion_threshold):
            return True
        
        # Gradient explosion (if we have recent gradients)
        if (len(self.gradient_norms) > 0 and 
            self.gradient_norms[-1] > self.config.grad_norm_threshold):
            return True
        
        return False
    
    def _check_nan_inf(self, g_loss: float, d_loss: float) -> bool:
        """Check for NaN or Inf values"""
        return (np.isnan(g_loss) or np.isinf(g_loss) or 
                np.isnan(d_loss) or np.isinf(d_loss))
    
    def _perform_convergence_analysis(self):
        """Perform comprehensive convergence analysis"""
        if len(self.generator_losses) < self.config.loss_window_size:
            return  # Not enough data yet
        
        # Calculate combined loss for convergence detection
        recent_g_losses = list(self.generator_losses)[-self.config.loss_window_size:]
        recent_d_losses = list(self.discriminator_losses)[-self.config.loss_window_size:]
        combined_losses = [(g + d) / 2 for g, d in zip(recent_g_losses, recent_d_losses)]
        
        # Check for convergence
        if self._check_convergence(combined_losses):
            self.convergence_status = ConvergenceStatus.CONVERGED
            return
        
        # Check for oscillation
        if self._check_oscillation(combined_losses):
            self.convergence_status = ConvergenceStatus.OSCILLATING
            return
        
        # Check for early stopping
        if self._check_early_stopping(combined_losses):
            self.convergence_status = ConvergenceStatus.EARLY_STOP
            return
        
        # Default to training
        self.convergence_status = ConvergenceStatus.TRAINING
    
    def _check_convergence(self, losses: List[float]) -> bool:
        """Check if training has converged"""
        if len(losses) < self.config.convergence_window:
            return False
        
        recent_losses = losses[-self.config.convergence_window:]
        
        # Check if loss variance is low
        loss_std = np.std(recent_losses)
        if loss_std < self.config.convergence_threshold:
            # Check if trend is flat
            trend = self._calculate_trend(recent_losses)
            if abs(trend) < self.config.loss_trend_threshold:
                return True
        
        return False
    
    def _check_oscillation(self, losses: List[float]) -> bool:
        """Check for oscillating behavior"""
        if len(losses) < self.config.oscillation_window:
            return False
        
        recent_losses = losses[-self.config.oscillation_window:]
        
        # Count direction changes
        direction_changes = 0
        for i in range(1, len(recent_losses) - 1):
            if ((recent_losses[i] > recent_losses[i-1] and recent_losses[i] > recent_losses[i+1]) or
                (recent_losses[i] < recent_losses[i-1] and recent_losses[i] < recent_losses[i+1])):
                direction_changes += 1
        
        # High number of direction changes indicates oscillation
        oscillation_ratio = direction_changes / (len(recent_losses) - 2)
        return oscillation_ratio > self.config.oscillation_threshold
    
    def _check_early_stopping(self, losses: List[float]) -> bool:
        """Check early stopping criteria"""
        if len(losses) < 5:
            return False
        
        current_loss = losses[-1]
        
        # Update best loss
        if current_loss < self.best_loss - self.config.min_improvement:
            self.best_loss = current_loss
            self.best_epoch = self.epoch_count
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        # Check patience
        return self.patience_counter >= self.config.patience
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend (slope) of values"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # Linear regression slope
        slope = np.polyfit(x, y, 1)[0]
        return slope
    
    def get_convergence_report(self) -> Dict:
        """Get detailed convergence report"""
        if len(self.generator_losses) == 0:
            return {"status": "no_data"}
        
        # Recent statistics
        recent_window = min(self.config.loss_window_size, len(self.generator_losses))
        recent_g_losses = list(self.generator_losses)[-recent_window:]
        recent_d_losses = list(self.discriminator_losses)[-recent_window:]
        
        report = {
            "convergence_status": self.convergence_status.value,
            "epoch": self.epoch_count,
            "best_loss": self.best_loss,
            "best_epoch": self.best_epoch,
            "patience_counter": self.patience_counter,
            "patience_limit": self.config.patience,
            
            # Loss statistics
            "recent_generator_loss": {
                "mean": np.mean(recent_g_losses),
                "std": np.std(recent_g_losses),
                "trend": self._calculate_trend(recent_g_losses)
            },
            "recent_discriminator_loss": {
                "mean": np.mean(recent_d_losses),
                "std": np.std(recent_d_losses),
                "trend": self._calculate_trend(recent_d_losses)
            },
            
            # Training stability
            "stability_metrics": {
                "loss_variance": np.var(recent_g_losses + recent_d_losses),
                "gradient_stability": np.std(list(self.gradient_norms)[-10:]) if len(self.gradient_norms) >= 10 else 0.0
            }
        }
        
        # Add quality metrics if available
        if len(self.fid_scores) > 0:
            report["quality_metrics"] = {
                "latest_fid": self.fid_scores[-1],
                "fid_trend": self._calculate_trend(list(self.fid_scores)[-5:])
            }
        
        if len(self.inception_scores) > 0:
            report["quality_metrics"] = report.get("quality_metrics", {})
            report["quality_metrics"]["latest_is"] = self.inception_scores[-1]
            report["quality_metrics"]["is_trend"] = self._calculate_trend(list(self.inception_scores)[-5:])
        
        return report
    
    def should_stop_training(self) -> bool:
        """Check if training should be stopped"""
        return self.convergence_status in [
            ConvergenceStatus.CONVERGED,
            ConvergenceStatus.DIVERGED,
            ConvergenceStatus.EARLY_STOP
        ]
    
    def get_recommendation(self) -> str:
        """Get training recommendation based on current status"""
        status = self.convergence_status
        
        if status == ConvergenceStatus.TRAINING:
            if self.patience_counter > self.config.patience // 2:
                return "Training is progressing but improvement is slow. Consider reducing learning rate."
            return "Training is progressing normally."
        
        elif status == ConvergenceStatus.CONVERGED:
            return "Training has converged. Consider stopping or reducing learning rate for fine-tuning."
        
        elif status == ConvergenceStatus.DIVERGED:
            return "Training has diverged. Stop training and reduce learning rate or check model architecture."
        
        elif status == ConvergenceStatus.OSCILLATING:
            return "Training is oscillating. Consider reducing learning rate or adjusting training parameters."
        
        elif status == ConvergenceStatus.EARLY_STOP:
            return "No improvement for extended period. Consider stopping or adjusting hyperparameters."
        
        return "Status unknown."
    
    def _save_state(self):
        """Save convergence detector state"""
        if not self.output_dir:
            return
        
        state_path = os.path.join(self.output_dir, self.config.state_file)
        
        try:
            state = {
                'generator_losses': list(self.generator_losses),
                'discriminator_losses': list(self.discriminator_losses),
                'gradient_norms': list(self.gradient_norms),
                'learning_rates': list(self.learning_rates),
                'fid_scores': list(self.fid_scores),
                'inception_scores': list(self.inception_scores),
                'stability_scores': list(self.stability_scores),
                'best_loss': self.best_loss,
                'best_epoch': self.best_epoch,
                'patience_counter': self.patience_counter,
                'epoch_count': self.epoch_count,
                'iteration_count': self.iteration_count,  # Added
                'convergence_status': self.convergence_status.value
            }
            
            with open(state_path, 'wb') as f:
                pickle.dump(state, f)
                
        except Exception as e:
            print(f"Warning: Could not save convergence state: {e}")
    
    def _load_state(self):
        """Load convergence detector state"""
        if not self.output_dir:
            return
        
        state_path = os.path.join(self.output_dir, self.config.state_file)
        
        if not os.path.exists(state_path):
            return
        
        try:
            with open(state_path, 'rb') as f:
                state = pickle.load(f)
            
            self.generator_losses = deque(state['generator_losses'], maxlen=200)
            self.discriminator_losses = deque(state['discriminator_losses'], maxlen=200)
            self.gradient_norms = deque(state['gradient_norms'], maxlen=100)
            self.learning_rates = deque(state['learning_rates'], maxlen=100)
            self.fid_scores = deque(state['fid_scores'], maxlen=50)
            self.inception_scores = deque(state['inception_scores'], maxlen=50)
            self.stability_scores = deque(state['stability_scores'], maxlen=50)
            self.best_loss = state['best_loss']
            self.best_epoch = state['best_epoch']
            self.patience_counter = state['patience_counter']
            self.epoch_count = state['epoch_count']
            self.iteration_count = state.get('iteration_count', 0)  # Added with default
            self.convergence_status = ConvergenceStatus(state['convergence_status'])
            
            print(f"📈 Convergence state loaded from {state_path}")
            
        except Exception as e:
            print(f"Warning: Could not load convergence state: {e}")


def create_convergence_detector(architecture: str = "dcgan", output_dir: str = "") -> ConvergenceDetector:
    """
    Create convergence detector with architecture-specific configuration
    
    Args:
        architecture: Architecture type ("dcgan" or "wavegan")
        output_dir: Output directory for state saving
        
    Returns:
        Configured convergence detector
    """
    if architecture.lower() == "dcgan":
        config = ConvergenceConfig(
            enabled=True,
            min_iterations=200,  # Increased from not specified
            check_frequency=50,  # Increased from not specified
            patience=150,  # Increased from 25 - DCGAN typically needs more patience
            min_improvement=0.0005,  # Reduced from 0.002
            loss_window_size=20,  # Increased from 12
            convergence_threshold=0.003,  # Reduced from 0.015
            oscillation_threshold=0.4,  # Increased from 0.25
            loss_explosion_threshold=200.0,  # Increased threshold
            grad_norm_threshold=2000.0  # Increased threshold
        )
    elif architecture.lower() == "wavegan":
        config = ConvergenceConfig(
            enabled=True,
            min_iterations=300,  # Increased from not specified  
            check_frequency=75,  # Increased from not specified
            patience=200,  # Increased from 30 - WaveGAN may need even more patience
            min_improvement=0.0003,  # Reduced from 0.001
            loss_window_size=25,  # Increased from 15
            convergence_threshold=0.005,  # Reduced from 0.02
            oscillation_threshold=0.5,  # Increased from 0.3
            loss_explosion_threshold=500.0,  # Increased threshold
            grad_norm_threshold=5000.0  # Increased threshold
        )
    else:
        config = ConvergenceConfig()  # Default
    
    return ConvergenceDetector(config, output_dir)
