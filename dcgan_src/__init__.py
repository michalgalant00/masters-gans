"""
DCGAN Source Package
===================

Modular implementation of DCGAN for spectrogram generation.
"""

# Import key components for easy access
# Removed config import to prevent automatic directory creation
from .dataset import SpectrogramDataset
from .generator import DCGANGenerator
from .discriminator import DCGANDiscriminator
from .training import DCGANTrainer
from .metrics import MetricsCollector
from .checkpoints import CheckpointManager
from .analytics import TrainingAnalyzer
