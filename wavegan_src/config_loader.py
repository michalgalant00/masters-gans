"""
Configuration Loader for WaveGAN
================================

Ładuje konfigurację z plików JSON i ustawia wszystkie parametry treningu.
Zastępuje poprzedni config.py zgodnie z wytycznymi projektu.
"""

import json
import os
import torch
import logging
from datetime import datetime
from typing import Dict, Any, Optional


class WaveGANConfig:
    """Klasa do ładowania i zarządzania konfiguracją WaveGAN"""
    
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
        
        print(f"✅ Konfiguracja załadowana z: {self.config_path}")
        return config
    
    def _setup_computed_values(self):
        """Ustawia wartości obliczane na podstawie konfiguracji"""
        # Device configuration
        device_config = self.config['hardware']['device']
        if device_config == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device_config)
        
        # Audio length in samples (jeśli nie podano explicite)
        if 'audio_length_samples' not in self.config['audio'] or self.config['audio']['audio_length_samples'] is None:
            self.config['audio']['audio_length_samples'] = int(
                self.config['audio']['sample_rate'] * self.config['audio']['audio_length_seconds']
            )
        
        # Dataset paths
        base_path = self.config['dataset']['base_path']
        self.dataset_files = os.path.join(base_path, self.config['dataset']['files_path'])
        self.dataset_metadata = os.path.join(base_path, self.config['dataset']['metadata_file'])
        self.dataset_spectrograms = os.path.join(base_path, self.config['dataset']['spectrograms_path'])
        self.dataset_spectrograms_metadata = os.path.join(base_path, self.config['dataset']['spectrograms_metadata'])
        
        # Output directories - FIXED: use output_wavegan as per project-basics.txt
        base_output = "output_wavegan"  # FIXED: hardcoded to match project requirements
        self.output_dir = base_output
        self.models_dir = os.path.join(base_output, self.config['output']['models_dir'])
        self.samples_dir = os.path.join(base_output, self.config['output']['samples_dir'])
        self.metrics_dir = os.path.join(base_output, self.config['output']['metrics_dir'])
        
        # Create output directories
        for dir_path in [self.output_dir, self.models_dir, self.samples_dir, self.metrics_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def _setup_logging(self):
        """Konfiguruje system logowania"""
        log_config = self.config['logging']
        
        # Setup log filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_name = os.path.splitext(os.path.basename(self.config_path))[0]
        self.log_filename = os.path.join(self.metrics_dir, f'wavegan_training_{config_name}_{timestamp}.log')
        
        # Setup handlers
        handlers = []
        if log_config['log_to_file']:
            handlers.append(logging.FileHandler(self.log_filename, mode='w'))
        if log_config['log_to_console']:
            handlers.append(logging.StreamHandler())
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_config['log_level'].upper()),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=handlers,
            force=True  # Wymusza rekonfigurację
        )
        
        self.logger = logging.getLogger(__name__)
        
        # Log initial configuration
        self.logger.info("=== WAVEGAN TRAINING SESSION STARTED ===")
        self.logger.info(f"Configuration file: {self.config_path}")
        self.logger.info(f"Log file: {self.log_filename}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Output directory: {self.output_dir}")
    
    def _validate_config(self):
        """Waliduje konfigurację"""
        required_sections = ['dataset', 'model', 'training', 'audio', 'metrics_and_checkpoints', 'logging', 'hardware', 'output']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Brakująca sekcja w konfiguracji: {section}")
        
        # Sprawdź czy dataset istnieje
        if not os.path.exists(self.dataset_files):
            self.logger.warning(f"Dataset directory not found: {self.dataset_files}")
        
        print(f"✅ Konfiguracja zwalidowana pomyślnie")
    
    def get(self, section: str, key: Optional[str] = None, default=None):
        """
        Pobiera wartość z konfiguracji
        
        Args:
            section: Nazwa sekcji (np. 'training')
            key: Klucz w sekcji (opcjonalny, jeśli None zwraca całą sekcję)
            default: Wartość domyślna jeśli klucz nie istnieje
        """
        if section not in self.config:
            return default
        
        if key is None:
            return self.config[section]
        
        return self.config[section].get(key, default)
    
    def update_from_cli_args(self, args):
        """
        Aktualizuje konfigurację wartościami z argumentów CLI
        
        Args:
            args: Argumenty z argparse
        """
        # Mapowanie argumentów CLI na konfigurację
        cli_mappings = {
            'epochs': ('training', 'epochs'),
            'batch_size': ('training', 'batch_size'),
            'learning_rate': ('training', 'learning_rate'),
            'latent_dim': ('model', 'latent_dim'),
            'model_dim': ('model', 'model_dim'),
            'device': ('hardware', 'device'),
            'output_dir': ('output', 'base_output_dir'),
        }
        
        updated = []
        for cli_arg, (section, key) in cli_mappings.items():
            if hasattr(args, cli_arg) and getattr(args, cli_arg) is not None:
                old_value = self.config[section][key]
                new_value = getattr(args, cli_arg)
                self.config[section][key] = new_value
                updated.append(f"{section}.{key}: {old_value} -> {new_value}")
        
        if updated:
            print("🔧 Konfiguracja zaktualizowana z argumentów CLI:")
            for update in updated:
                print(f"   {update}")
            
            # Przelicz wartości zależne
            self._setup_computed_values()
    
    def print_summary(self):
        """Wyświetla podsumowanie konfiguracji"""
        print("=" * 60)
        print("📋 PODSUMOWANIE KONFIGURACJI WAVEGAN")
        print("=" * 60)
        print(f"📁 Źródło: {self.config_path}")
        print(f"🖥️  Urządzenie: {self.device}")
        print(f"📊 Dane: {self.dataset_files}")
        print(f"📈 Wyjście: {self.output_dir}")
        print()
        
        print("🧠 MODEL:")
        model = self.config['model']
        print(f"   Latent dim: {model['latent_dim']}")
        print(f"   Model dim: {model['model_dim']}")
        print(f"   Kernel length: {model['kernel_len']}")
        
        print("🎯 TRENING:")
        training = self.config['training']
        print(f"   Epochs: {training['epochs']}")
        print(f"   Batch size: {training['batch_size']}")
        print(f"   Learning rate: {training['learning_rate']}")
        print(f"   N critic: {training['n_critic']}")
        
        print("🎵 AUDIO:")
        audio = self.config['audio']
        print(f"   Sample rate: {audio['sample_rate']} Hz")
        print(f"   Length: {audio['audio_length_seconds']:.3f}s ({audio['audio_length_samples']} samples)")
        
        print("📊 METRYKI & CHECKPOINTY:")
        metrics = self.config['metrics_and_checkpoints']
        print(f"   CSV frequency: every {metrics['csv_write_frequency']} iterations")
        print(f"   Emergency checkpoints: every {metrics['emergency_checkpoint_frequency']} iterations")
        print(f"   Milestone checkpoints: every {metrics['milestone_checkpoint_frequency']} iterations")
        print(f"   Milestone epochs: every {metrics['milestone_epoch_frequency']} epochs")
        
        print("=" * 60)
    
    def check_and_rotate_log(self):
        """Sprawdza rozmiar loga i rotuje jeśli przekracza limit"""
        if not self.config['logging']['log_to_file']:
            return
        
        try:
            max_size = self.config['logging']['max_log_size_mb'] * 1024 * 1024
            if os.path.exists(self.log_filename):
                file_size = os.path.getsize(self.log_filename)
                if file_size > max_size:
                    with open(self.log_filename, 'w') as f:
                        f.write('')
                    self.logger.info("=== LOG ROTATED - Previous log exceeded size limit ===")
                    self.logger.info("=== CONTINUING TRAINING SESSION ===")
        except Exception as e:
            pass  # Ignore errors in log rotation
    
    def log_tensor_stats(self, tensor_name: str, tensor: torch.Tensor, step: Optional[int] = None):
        """Loguje szczegółowe statystyki tensora"""
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


# Funkcje pomocnicze dla kompatybilności wstecznej
def load_config(config_path: Optional[str] = None) -> WaveGANConfig:
    """Ładuje konfigurację WaveGAN"""
    return WaveGANConfig(config_path)


def load_config_by_name(config_name: str) -> WaveGANConfig:
    """Ładuje konfigurację po nazwie"""
    config_dir = os.path.dirname(__file__)
    config_path = os.path.join(config_dir, f'{config_name}.json')
    return WaveGANConfig(config_path)


def get_available_configs() -> Dict[str, str]:
    """Zwraca dostępne pliki konfiguracyjne"""
    config_dir = os.path.dirname(__file__)
    configs = {}
    
    # Znajdź wszystkie pliki config*.json
    for filename in os.listdir(config_dir):
        if filename.startswith('config') and filename.endswith('.json'):
            path = os.path.join(config_dir, filename)
            if os.path.exists(path):
                configs[filename.replace('.json', '')] = path
    
    return configs
