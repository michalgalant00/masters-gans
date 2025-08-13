"""
WaveGAN - Generative Adversarial Network for Audio Generation
============================================================

A modular implementation of WaveGAN for generating engine sound samples.

Modules:
- config_loader: JSON-based configuration management
- dataset: Audio dataset loading and preprocessing
- generator: WaveGAN Generator network
- discriminator: WaveGAN Discriminator network
- training: Training pipeline and utilities
- metrics: Training metrics collection and logging
- checkpoints: Checkpoint management and disaster recovery
- analytics: Post-training analysis and reporting
"""

from .config_loader import WaveGANConfig, load_config, get_available_configs
from .dataset import AudioDataset
from .generator import WaveGANGenerator
from .discriminator import WaveGANDiscriminator
from .training import WaveGANTrainer
from .metrics import MetricsCollector
from .checkpoints import CheckpointManager
from .analytics import TrainingAnalyzer

__version__ = "1.0.0"
__author__ = "Engine Sound Generation Research"
