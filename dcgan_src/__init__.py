"""
DCGAN Source Package
===================

Modular implementation of DCGAN for spectrogram generation.
"""

# Version info
__version__ = "1.0.0"
__author__ = "DCGAN Implementation"

# Import key components for easy access
from .config import *
from .dataset import SpectrogramDataset
from .generator import DCGANGenerator
from .discriminator import DCGANDiscriminator
from .training import DCGANTrainer
from .metrics import MetricsCollector
from .checkpoints import CheckpointManager
from .analytics import TrainingAnalyzer
