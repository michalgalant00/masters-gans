"""
WaveGAN - Main Entry Point
==========================

Main script for training WaveGAN for engine sound generation.

Usage:
    python WaveGAN.py [options]
    
Examples:
    # Użyj domyślnej konfigura        # Load configuration
        if args.config is None:
            # Use default config.json
            print("📁 Loading default configuration...")
            config = load_config()
        elif args.config.endswith('.json'):
            # Full path to JSON file
            print(f"📁 Loading configuration from: {args.config}")
            config = load_config(args.config)
        else:
            # Configuration name
            print(f"📁 Loading configuration: {args.config}")
            config = load_config_by_name(args.config)n WaveGAN.py
    
    # Użyj konfiguracji testowej
    python WaveGAN.py --config config_test
    
    # Nadpisz niektóre parametry
    python WaveGAN.py --epochs 1000 --batch-size 64 --learning-rate 0.0002
    
    # Użyj własnej konfiguracji
    python WaveGAN.py --config path/to/custom_config.json

Configuration:
    Konfiguracja ładowana z plików JSON w katalogu wavegan_src/:
    - config.json: Główna konfiguracja produkcyjna
    - config_test.json: Szybka konfiguracja testowa z pełną funkcjonalnością
    - config_template.json: Szablon z komentarzami
    
Structure:
    - wavegan_src/config_loader.py: System konfiguracji JSON
    - wavegan_src/dataset.py: Audio dataset loading
    - wavegan_src/generator.py: Generator network
    - wavegan_src/discriminator.py: Discriminator network  
    - wavegan_src/training.py: Training pipeline
"""

import os
import sys
import argparse
from typing import Optional

# Add the wavegan_src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'wavegan_src'))

from wavegan_src.config_loader import load_config, get_available_configs, load_config_by_name
from wavegan_src.training import WaveGANTrainer


def parse_arguments() -> argparse.Namespace:
    """Parsuje argumenty z linii poleceń"""
    parser = argparse.ArgumentParser(
        description='WaveGAN Implementation for Engine Sound Generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Podstawowe uruchomienie z domyślną konfiguracją
  python WaveGAN.py
  
  # Użyj konfiguracji testowej (szybko)
  python WaveGAN.py --config config_test
  
  # Nadpisz parametry treningu
  python WaveGAN.py --epochs 1000 --batch-size 64
  
  # Użyj własnej konfiguracji
  python WaveGAN.py --config my_config.json
  
  # Wymuś użycie CPU
  python WaveGAN.py --device cpu

Available configurations:
"""
    )
    
    # Configuration selection
    parser.add_argument('--config', type=str, 
                        help='Configuration to use (config, config_test, config_template) or path to JSON file')
    
    # Training parameters (override config file)
    parser.add_argument('--epochs', type=int,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, dest='batch_size',
                        help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, dest='learning_rate',
                        help='Learning rate for optimizers')
    parser.add_argument('--latent-dim', type=int, dest='latent_dim',
                        help='Latent dimension for generator input')
    
    # Model parameters
    parser.add_argument('--model-dim', type=int, dest='model_dim',
                        help='Basic model dimension (WaveGAN-specific)')
    parser.add_argument('--audio-length', type=int, dest='audio_length',
                        help='Audio length in samples')
    
    # Hardware
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'auto'],
                        help='Device to use for training')
    parser.add_argument('--num-workers', type=int, dest='num_workers',
                        help='Number of data loading workers')
    
    # Output
    parser.add_argument('--output-dir', type=str, dest='output_dir',
                        help='Base output directory')
    
    # WaveGAN-specific options
    parser.add_argument('--no-training', action='store_true',
                        help='Only load model without training')
    parser.add_argument('--generate-only', type=int, dest='generate_only',
                        help='Only generate N samples (requires trained model)')
    
    return parser.parse_args()


def show_available_configs():
    """Wyświetla dostępne konfiguracje"""
    configs = get_available_configs()
    print("\nAvailable configurations:")
    for name, path in configs.items():
        print(f"  {name:<15} - {path}")
    print()


def apply_cli_overrides(config, args):
    """Aplikuje nadpisania z CLI do konfiguracji"""
    overrides_applied = []
    
    # Training overrides
    if args.epochs is not None:
        config.config['training']['epochs'] = args.epochs
        overrides_applied.append(f"epochs={args.epochs}")
    
    if args.batch_size is not None:
        config.config['training']['batch_size'] = args.batch_size
        overrides_applied.append(f"batch_size={args.batch_size}")
    
    if args.learning_rate is not None:
        config.config['training']['learning_rate'] = args.learning_rate
        overrides_applied.append(f"learning_rate={args.learning_rate}")
    
    # Model overrides
    if args.latent_dim is not None:
        config.config['model']['latent_dim'] = args.latent_dim
        overrides_applied.append(f"latent_dim={args.latent_dim}")
    
    if args.model_dim is not None:
        config.config['model']['model_dim'] = args.model_dim
        overrides_applied.append(f"model_dim={args.model_dim}")
    
    if args.audio_length is not None:
        config.config['audio']['audio_length_samples'] = args.audio_length
        overrides_applied.append(f"audio_length={args.audio_length}")
    
    # Hardware overrides
    if args.device is not None:
        config.config['hardware']['device'] = args.device
        overrides_applied.append(f"device={args.device}")
    
    if args.num_workers is not None:
        config.config['hardware']['num_workers'] = args.num_workers
        overrides_applied.append(f"num_workers={args.num_workers}")
    
    # Output override
    if args.output_dir is not None:
        config.config['output']['base_output_dir'] = args.output_dir
        overrides_applied.append(f"output_dir={args.output_dir}")
    
    # Re-setup computed values after overrides
    if overrides_applied:
        print(f"📝 CLI Overrides applied: {', '.join(overrides_applied)}")
        config._setup_computed_values()
    
    return overrides_applied


def resolve_config_path(config_name: str) -> str:
    """
    Rozwiązuje ścieżkę do pliku konfiguracyjnego
    
    Args:
        config_name: Nazwa konfiguracji lub pełna ścieżka
    
    Returns:
        Pełna ścieżka do pliku konfiguracyjnego
    """
    # Jeśli to pełna ścieżka, zwróć ją
    if os.path.isabs(config_name) or config_name.endswith('.json'):
        return config_name
    
    # Sprawdź dostępne konfiguracje
    available_configs = get_available_configs()
    
    if config_name in available_configs:
        return available_configs[config_name]
    
    # Jeśli nie znaleziono, spróbuj dodać .json
    config_path = os.path.join(os.path.dirname(__file__), 'wavegan_src', f'{config_name}.json')
    if os.path.exists(config_path):
        return config_path
    
    raise FileNotFoundError(f"Nie znaleziono konfiguracji: {config_name}")


def main() -> int:
    """Główny punkt wejścia dla treningu WaveGAN"""
    print("WaveGAN Implementation for Engine Sound Generation")
    print("=" * 50)
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Show available configurations if requested
    if args.config and args.config in ['list', 'show', 'help']:
        show_available_configs()
        return 0
    
    try:
        # Load configuration
        if args.config is None:
            # Use default config.json
            print("� Loading default configuration...")
            config = load_config()
        elif args.config.endswith('.json'):
            # Full path to JSON file
            print(f"📁 Loading configuration from: {args.config}")
            config = load_config(args.config)
        else:
            # Configuration name
            print(f"📁 Loading configuration: {args.config}")
            config_path = resolve_config_path(args.config)
            config = load_config(config_path)
        
        # Apply CLI overrides
        overrides = apply_cli_overrides(config, args)
        
        # Show final configuration
        if overrides:
            print("\n" + "=" * 50)
            print("📊 Final Configuration (after CLI overrides):")
            config.print_summary()
        
        # Check if dataset exists
        if not os.path.exists(config.dataset_files):
            print(f"❌ Error: Dataset directory not found at {config.dataset_files}")
            print("Please ensure the dataset is prepared and paths are correct.")
            return 1
        
        # Only generate samples if requested
        if args.generate_only:
            print(f"🎵 Generating {args.generate_only} samples...")
            trainer = WaveGANTrainer(config)
            trainer.generate_samples(num_samples=args.generate_only)
            print("✅ Generation completed!")
            return 0
        
        # Initialize trainer
        print("\n🚀 Initializing WaveGAN trainer...")
        trainer = WaveGANTrainer(config)
        
        # Start training (if not disabled)
        if not args.no_training:
            print("🎯 Starting WaveGAN training...")
            trainer.train()
        
        print("\n✅ Training completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        return 1
    except FileNotFoundError as e:
        print(f"\n❌ Configuration file not found: {e}")
        show_available_configs()
        return 1
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
