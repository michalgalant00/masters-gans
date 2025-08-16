"""
DCGAN Configuration (DISABLED)
==============================

This module has been disabled to prevent automatic configuration loading during imports.
All configuration loading is now handled explicitly by config_loader.py.
"""

import os
import logging

# All configuration variables are set to None to indicate they are not loaded
_config = None
DATASET_BASE_PATH = None
DATASET_SPECTROGRAMS = None
DATASET_SPECTROGRAMS_METADATA = None
DEVICE = None
LATENT_DIM = None
FEATURES_G = None
FEATURES_D = None
IMAGE_SIZE = None
CHANNELS = None
BATCH_SIZE = None
EPOCHS = None
LEARNING_RATE = None
BETA1 = None
CSV_WRITE_FREQUENCY = None
DETAILED_LOGGING_FREQUENCY = None
EMERGENCY_CHECKPOINT_FREQUENCY = None
MILESTONE_CHECKPOINT_FREQUENCY = None
MILESTONE_EPOCH_FREQUENCY = None
EMERGENCY_BUFFER_SIZE = None
IMAGE_SAMPLES_PER_CHECKPOINT = None
BEST_MODEL_IMPROVEMENT_THRESHOLD = None
KEY_EPOCHS_FIRST = None
KEY_EPOCHS_LAST = None
NUM_WORKERS = None
MODELS_DIR = None
SAMPLES_DIR = None
METRICS_DIR = None
BASE_OUTPUT_DIR = None
logger = None
log_filename = None
CONFIG_MODE = "DISABLED"

def check_and_rotate_log():
    """Disabled function"""
    pass

def log_tensor_stats(tensor_name, tensor, step=None):
    """Disabled function"""
    pass
