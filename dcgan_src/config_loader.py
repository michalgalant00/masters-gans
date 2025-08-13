"""
Configuration Loader for DCGAN
===============================

Ładuje konfigurację z plików JSON i ustawia wszystkie parametry treningu.
Zastępuje poprzedni config.py zgodnie z wytycznymi projektu.
"""

import json
import os
import torch
import logging
from datetime import datetime
from typing import Dict, Any, Optional


class DCGANConfig:
    """Klasa do ładowania i zarządzania konfiguracją DCGAN"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Inicjalizacja konfiguracji
        
        Args:
            config_path: Ścieżka do pliku konfiguracyjnego JSON
        """
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        
        self.config_path = config_path
        self.config = self._load_config()
        self._setup_computed_values()
        self._setup_logging()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Ładuje konfigurację z pliku JSON"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Plik konfiguracyjny nie znaleziony: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        print(f"✅ Konfiguracja DCGAN załadowana z: {self.config_path}")
        return config
    
    def _setup_computed_values(self):
        """Ustawia wartości obliczane na podstawie konfiguracji"""
        # Device configuration
        device_config = self.config['hardware']['device']
        if device_config == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device_config)
        
        # Compute dataset paths
        dataset_config = self.config['dataset']
        self.dataset_base_path = dataset_config['base_path']
        self.dataset_spectrograms = os.path.join(self.dataset_base_path, dataset_config['spectrograms_path'])
        self.dataset_spectrograms_metadata = os.path.join(self.dataset_base_path, dataset_config['spectrograms_metadata'])
        
        # Setup output directories - FIXED: use config value or fallback to output_dcgan
        output_config = self.config['output']
        self.base_output_dir = output_config.get('base_output_dir', "output_dcgan")  # Use config value or fallback
        self.output_dir = self.base_output_dir  # ADDED: compatibility with WaveGAN interface
        self.models_dir = os.path.join(self.base_output_dir, output_config['models_dir'])
        self.samples_dir = os.path.join(self.base_output_dir, output_config['samples_dir'])
        self.metrics_dir = os.path.join(self.base_output_dir, output_config['metrics_dir'])
        
        # Create output directories
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.samples_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
    
    def _setup_logging(self):
        """Konfiguruje system logowania"""
        logging_config = self.config['logging']
        
        # Determine config mode from filename for log naming
        config_name = os.path.basename(self.config_path).replace('.json', '')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.log_filename = os.path.join(self.metrics_dir, f'dcgan_training_{config_name}_{timestamp}.log')
        
        # Configure logging
        log_level = getattr(logging, logging_config['log_level'].upper())
        
        handlers = []
        if logging_config['log_to_file']:
            handlers.append(logging.FileHandler(self.log_filename, mode='w'))
        if logging_config['log_to_console']:
            handlers.append(logging.StreamHandler())
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=handlers
        )
        
        self.logger = logging.getLogger(__name__)
        
        # Log initial configuration
        self.logger.info("=== DCGAN TRAINING SESSION STARTED ===")
        self.logger.info(f"Configuration file: {self.config_path}")
        self.logger.info(f"Log file: {self.log_filename}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Output directory: {self.base_output_dir}")
    
    def _validate_config(self):
        """Waliduje konfigurację"""
        model_config = self.config['model']
        
        # Check if image size is power of 2
        image_size = model_config['image_size']
        if image_size & (image_size - 1) != 0:
            raise ValueError(f"Image size must be power of 2, got {image_size}")
        
        # Check dataset paths
        if not os.path.exists(self.dataset_spectrograms):
            print(f"⚠️  Warning: Dataset path does not exist: {self.dataset_spectrograms}")
        
        if not os.path.exists(self.dataset_spectrograms_metadata):
            print(f"⚠️  Warning: Metadata file does not exist: {self.dataset_spectrograms_metadata}")
    
    def get_model_config(self) -> Dict[str, Any]:
        """Zwraca konfigurację modelu"""
        return self.config['model']
    
    def get_training_config(self) -> Dict[str, Any]:
        """Zwraca konfigurację treningu"""
        return self.config['training']
    
    def get_metrics_config(self) -> Dict[str, Any]:
        """Zwraca konfigurację metryk i checkpointów"""
        return self.config['metrics_and_checkpoints']
    
    def get_hardware_config(self) -> Dict[str, Any]:
        """Zwraca konfigurację sprzętową"""
        return self.config['hardware']
    
    def check_and_rotate_log(self):
        """Sprawdza rozmiar loga i rotuje jeśli > max_log_size_mb"""
        try:
            max_size_mb = self.config['logging']['max_log_size_mb']
            if os.path.exists(self.log_filename):
                file_size = os.path.getsize(self.log_filename)
                if file_size > max_size_mb * 1024 * 1024:  # Convert MB to bytes
                    # Clear the log file
                    with open(self.log_filename, 'w') as f:
                        f.write('')
                    self.logger.info(f"=== LOG ROTATED - Previous log exceeded {max_size_mb}MB ===")
                    self.logger.info(f"=== CONTINUING TRAINING SESSION ===")
        except Exception as e:
            pass  # Ignore errors in log rotation
    
    def log_tensor_stats(self, tensor_name: str, tensor: torch.Tensor, step: Optional[int] = None):
        """Loguje szczegółowe statystyki tensora do wykrywania NaN/Inf"""
        if step is not None:
            prefix = f"Step {step} - {tensor_name}"
        else:
            prefix = tensor_name
        
        self.logger.info(f"{prefix}: shape={tensor.shape}")
        self.logger.info(f"{prefix}: min={tensor.min().item():.6f}, max={tensor.max().item():.6f}")
        self.logger.info(f"{prefix}: mean={tensor.mean().item():.6f}, std={tensor.std().item():.6f}")
        self.logger.info(f"{prefix}: has_nan={torch.isnan(tensor).any().item()}")
        self.logger.info(f"{prefix}: has_inf={torch.isinf(tensor).any().item()}")
        self.logger.info(f"{prefix}: all_zero={(tensor.abs() < 1e-7).all().item()}")
        self.logger.info(f"{prefix}: norm={tensor.norm().item():.6f}")
    
    def print_config_summary(self):
        """Wyświetla podsumowanie konfiguracji"""
        model_config = self.get_model_config()
        training_config = self.get_training_config()
        metrics_config = self.get_metrics_config()
        
        print("=" * 60)
        print("DCGAN Configuration Summary:")
        print(f"  - Config file: {os.path.basename(self.config_path)}")
        print(f"  - Device: {self.device}")
        print(f"  - Batch size: {training_config['batch_size']}")
        print(f"  - Image size: {model_config['image_size']}x{model_config['image_size']}")
        print(f"  - Generator features: {model_config['features_g']}")
        print(f"  - Discriminator features: {model_config['features_d']}")
        print(f"  - Training epochs: {training_config['epochs']}")
        print(f"  - Latent dimension: {model_config['latent_dim']}")
        print(f"  - Learning rate: {training_config['learning_rate']}")
        print("")
        print("📊 Metrics & Checkpoints Configuration:")
        print(f"  - CSV write frequency: every {metrics_config['csv_write_frequency']} iterations")
        print(f"  - Detailed logging: every {metrics_config['detailed_logging_frequency']} iterations")
        print(f"  - Emergency checkpoints: every {metrics_config['emergency_checkpoint_frequency']} iterations")
        print(f"  - Milestone checkpoints: every {metrics_config['milestone_checkpoint_frequency']} iterations")
        print(f"  - Milestone epochs: every {metrics_config['milestone_epoch_frequency']} epochs")
        print(f"  - Emergency buffer size: {metrics_config['emergency_buffer_size']} files")
        print(f"  - Image samples per checkpoint: {metrics_config['image_samples_per_checkpoint']}")
        print(f"  - Best model improvement threshold: {metrics_config['best_model_improvement_threshold']*100:.1f}%")
        print("=" * 60)
        print("")
        print("📁 Output Directories:")
        print(f"  - Base: {self.base_output_dir}")
        print(f"  - Models: {self.models_dir}")
        print(f"  - Samples: {self.samples_dir}")
        print(f"  - Metrics: {self.metrics_dir}")
        print("=" * 60)


def load_config(config_path: Optional[str] = None) -> DCGANConfig:
    """
    Ładuje konfigurację DCGAN z pliku JSON
    
    Args:
        config_path: Ścieżka do pliku konfiguracyjnego JSON
        
    Returns:
        DCGANConfig: Obiekt konfiguracji
    """
    return DCGANConfig(config_path)


def get_available_configs() -> Dict[str, str]:
    """
    Zwraca dostępne pliki konfiguracyjne w katalogu dcgan_src
    
    Returns:
        Dict mapujący nazwy konfiguracji na ścieżki plików
    """
    config_dir = os.path.dirname(__file__)
    configs = {}
    
    for filename in ['config.json', 'config_test.json', 'config_template.json']:
        path = os.path.join(config_dir, filename)
        if os.path.exists(path):
            config_name = filename.replace('.json', '')
            configs[config_name] = path
    
    return configs


def load_config_by_name(config_name: str) -> DCGANConfig:
    """
    Ładuje konfigurację po nazwie
    
    Args:
        config_name: Nazwa konfiguracji ('config', 'config_test', 'config_template')
        
    Returns:
        DCGANConfig: Obiekt konfiguracji
    """
    available_configs = get_available_configs()
    
    if config_name not in available_configs:
        available_names = list(available_configs.keys())
        raise ValueError(f"Configuration '{config_name}' not found. Available: {available_names}")
    
    return load_config(available_configs[config_name])
