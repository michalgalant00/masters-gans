"""
DCGAN Configuration
==================

Configuration module that uses JSON-based configuration system.
This module provides backward compatibility while using the new JSON config loader.

Usage:
    from dcgan_src.config import *
    
All configuration variables are loaded from JSON files and exposed as module-level constants.
"""

import os
import logging
from .config_loader import load_config, DCGANConfig

# Determine which config to load based on environment or default
CONFIG_MODE = os.getenv('DCGAN_CONFIG_MODE', 'config')  # Default to main config.json

# Load configuration
if CONFIG_MODE.endswith('.json'):
    # Full path provided
    config_path = CONFIG_MODE
else:
    # Config name provided, build path
    config_dir = os.path.dirname(__file__)
    config_path = os.path.join(config_dir, f'{CONFIG_MODE}.json')

try:
    # Load the configuration
    _config = load_config(config_path)

    # Export all configuration as module-level constants for backward compatibility
    # Dataset paths
    DATASET_BASE_PATH = _config.dataset_base_path
    DATASET_SPECTROGRAMS = _config.dataset_spectrograms
    DATASET_SPECTROGRAMS_METADATA = _config.dataset_spectrograms_metadata

    # Hardware
    DEVICE = _config.device

    # Model configuration
    model_config = _config.get_model_config()
    LATENT_DIM = model_config['latent_dim']
    FEATURES_G = model_config['features_g']
    FEATURES_D = model_config['features_d']
    IMAGE_SIZE = model_config['image_size']
    CHANNELS = model_config['channels']

    # Training configuration
    training_config = _config.get_training_config()
    BATCH_SIZE = training_config['batch_size']
    EPOCHS = training_config['epochs']
    LEARNING_RATE = training_config['learning_rate']
    BETA1 = training_config['beta1']

    # Metrics and checkpoints configuration
    metrics_config = _config.get_metrics_config()
    CSV_WRITE_FREQUENCY = metrics_config['csv_write_frequency']
    DETAILED_LOGGING_FREQUENCY = metrics_config['detailed_logging_frequency']
    EMERGENCY_CHECKPOINT_FREQUENCY = metrics_config['emergency_checkpoint_frequency']
    MILESTONE_CHECKPOINT_FREQUENCY = metrics_config['milestone_checkpoint_frequency']
    MILESTONE_EPOCH_FREQUENCY = metrics_config['milestone_epoch_frequency']
    EMERGENCY_BUFFER_SIZE = metrics_config['emergency_buffer_size']
    IMAGE_SAMPLES_PER_CHECKPOINT = metrics_config['image_samples_per_checkpoint']
    BEST_MODEL_IMPROVEMENT_THRESHOLD = metrics_config['best_model_improvement_threshold']
    KEY_EPOCHS_FIRST = metrics_config['key_epochs_first']
    KEY_EPOCHS_LAST = metrics_config['key_epochs_last']

    # Hardware configuration
    hardware_config = _config.get_hardware_config()
    NUM_WORKERS = hardware_config['num_workers']

    # Output directories
    MODELS_DIR = _config.models_dir
    SAMPLES_DIR = _config.samples_dir
    METRICS_DIR = _config.metrics_dir
    BASE_OUTPUT_DIR = _config.base_output_dir

    # Logging functions - expose from config object
    logger = _config.logger
    log_filename = _config.log_filename
    check_and_rotate_log = _config.check_and_rotate_log
    log_tensor_stats = _config.log_tensor_stats

    # Print configuration summary
    _config.print_config_summary()

    # Backward compatibility - determine CONFIG_MODE based on loaded config
    config_filename = os.path.basename(_config.config_path)
    if 'test' in config_filename:
        CONFIG_MODE = "TESTING"
        print("🚀 TESTING MODE ACTIVATED")
        print("   - Small model size for fast computation")
        print(f"   - {IMAGE_SIZE}x{IMAGE_SIZE} image size")
        print(f"   - {EPOCHS} epochs for quick testing")
        print(f"   - Small batch size ({BATCH_SIZE})")
        print("   - OPTIMIZED: Reduced checkpoint/metrics frequency")
        print("⚡ Expected training time: ~5-10 minutes")
    else:
        CONFIG_MODE = "PRODUCTION"
        print("🏭 PRODUCTION MODE ACTIVATED")
        print("   - Full model optimized for high-quality spectrograms")
        print(f"   - {IMAGE_SIZE}x{IMAGE_SIZE} high-resolution spectrograms")
        print(f"   - {EPOCHS} epochs for professional results")
        print(f"   - Large batch size ({BATCH_SIZE})")
        print("🎯 Expected training time: ~4-8 hours on RTX 4090")

    print("\n📋 System Requirements Check:")
    print("  ✅ PyTorch available")

    # Check optional dependencies
    try:
        import psutil
        print("  ✅ psutil available (system monitoring enabled)")
    except ImportError:
        print("  ⚠️  psutil not found - install with: pip install psutil")
        print("     (System resource monitoring will be limited)")

    try:
        import pandas
        print("  ✅ pandas available")
    except ImportError:
        print("  ❌ pandas required - install with: pip install pandas")

    try:
        import matplotlib
        print("  ✅ matplotlib available")
    except ImportError:
        print("  ❌ matplotlib required - install with: pip install matplotlib")

    try:
        from PIL import Image
        print("  ✅ PIL available")
    except ImportError:
        print("  ❌ PIL required - install with: pip install Pillow")

    print("=" * 60)

except Exception as e:
    # Fallback configuration if JSON loading fails
    print(f"⚠️  Warning: Could not load JSON config ({e}), using fallback defaults")
    
    import torch
    
    # Fallback values
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CONFIG_MODE = "TESTING"
    
    # Dataset paths (fallback)
    DATASET_BASE_PATH = '../01_dataset_prep/dataset-processed/final'
    DATASET_SPECTROGRAMS = f'{DATASET_BASE_PATH}/spectrograms/'
    DATASET_SPECTROGRAMS_METADATA = f'{DATASET_BASE_PATH}/spectrograms-metadata.csv'
    
    # Model config (testing defaults)
    LATENT_DIM = 50
    FEATURES_G = 32
    FEATURES_D = 32
    IMAGE_SIZE = 64
    CHANNELS = 1
    
    # Training config (testing defaults)
    BATCH_SIZE = 8
    EPOCHS = 5
    LEARNING_RATE = 0.0002
    BETA1 = 0.5
    
    # Metrics config (testing defaults)
    CSV_WRITE_FREQUENCY = 50
    DETAILED_LOGGING_FREQUENCY = 50
    EMERGENCY_CHECKPOINT_FREQUENCY = 50
    MILESTONE_CHECKPOINT_FREQUENCY = 200
    MILESTONE_EPOCH_FREQUENCY = 1
    EMERGENCY_BUFFER_SIZE = 3
    IMAGE_SAMPLES_PER_CHECKPOINT = 5
    BEST_MODEL_IMPROVEMENT_THRESHOLD = 0.05
    KEY_EPOCHS_FIRST = 1
    KEY_EPOCHS_LAST = 1
    
    # Hardware config
    NUM_WORKERS = 2
    
    # Output directories
    BASE_OUTPUT_DIR = "output_dcgan_fallback"
    MODELS_DIR = os.path.join(BASE_OUTPUT_DIR, "models")
    SAMPLES_DIR = os.path.join(BASE_OUTPUT_DIR, "samples")
    METRICS_DIR = os.path.join(BASE_OUTPUT_DIR, "metrics")
    
    # Create output directories
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    
    # Fallback logging
    logger = logging.getLogger(__name__)
    log_filename = os.path.join(METRICS_DIR, "dcgan_fallback.log")
    
    def check_and_rotate_log():
        pass
    
    def log_tensor_stats(tensor_name, tensor, step=None):
        pass
