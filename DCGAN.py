"""
DCGAN - Main Entry Point
========================

Main script for training DCGAN for spectrogram generation.

Usage:
    python DCGAN.py [options]
    
Examples:
    # Użyj domyślnej konfiguracji
    python DCGAN.py
    
    # Użyj konfiguracji testowej
    python DCGAN.py --config config_test
    
    # Nadpisz niektóre parametry
    python DCGAN.py --epochs 100 --batch-size 32 --learning-rate 0.0001
    
    # Użyj własnej konfiguracji
    python DCGAN.py --config path/to/custom_config.json

Configuration:
    Konfiguracja ładowana z plików JSON w katalogu dcgan_src/:
    - config.json: Główna konfiguracja produkcyjna
    - config_test.json: Lekka konfiguracja do testów
    - config_template.json: Szablon z komentarzami
    
Structure:
    - dcgan_src/config_loader.py: System konfiguracji JSON
    - dcgan_src/dataset.py: Spectrogram dataset loading
    - dcgan_src/generator.py: Generator network
    - dcgan_src/discriminator.py: Discriminator network  
    - dcgan_src/training.py: Training pipeline
"""

import os
import sys
import argparse
from typing import Optional

# Add the dcgan_src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'dcgan_src'))

from dcgan_src.config_loader import load_config, get_available_configs, load_config_by_name
from dcgan_src.training import DCGANTrainer


def parse_arguments() -> argparse.Namespace:
    """Parsuje argumenty z linii poleceń"""
    parser = argparse.ArgumentParser(
        description='DCGAN Implementation for Spectrogram Generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Podstawowe uruchomienie z domyślną konfiguracją
  python DCGAN.py
  
  # Użyj konfiguracji testowej (szybko)
  python DCGAN.py --config config_test
  
  # Nadpisz parametry treningu
  python DCGAN.py --epochs 100 --batch-size 32
  
  # Użyj własnej konfiguracji
  python DCGAN.py --config my_config.json
  
  # Wymuś użycie CPU
  python DCGAN.py --device cpu

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
    parser.add_argument('--image-size', type=int, dest='image_size',
                        help='Image size (must be power of 2)')
    parser.add_argument('--features-g', type=int, dest='features_g',
                        help='Number of generator features')
    parser.add_argument('--features-d', type=int, dest='features_d',
                        help='Number of discriminator features')
    
    # Hardware
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'auto'],
                        help='Device to use for training')
    parser.add_argument('--num-workers', type=int, dest='num_workers',
                        help='Number of data loading workers')
    
    # Output
    parser.add_argument('--output-dir', type=str, dest='output_dir',
                        help='Base output directory')
    
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
    
    if args.image_size is not None:
        config.config['model']['image_size'] = args.image_size
        overrides_applied.append(f"image_size={args.image_size}")
    
    if args.features_g is not None:
        config.config['model']['features_g'] = args.features_g
        overrides_applied.append(f"features_g={args.features_g}")
    
    if args.features_d is not None:
        config.config['model']['features_d'] = args.features_d
        overrides_applied.append(f"features_d={args.features_d}")
    
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


def main() -> int:
    """Główny punkt wejścia dla treningu DCGAN"""
    print("DCGAN Implementation for Spectrogram Generation")
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
            print("📁 Loading default configuration...")
            config = load_config()
        elif args.config.endswith('.json'):
            # Full path to JSON file
            print(f"📁 Loading configuration from: {args.config}")
            config = load_config(args.config)
        else:
            # Configuration name
            print(f"📁 Loading configuration: {args.config}")
            config = load_config_by_name(args.config)
        
        # Apply CLI overrides
        overrides = apply_cli_overrides(config, args)
        
        # Show final configuration
        if overrides:
            print("\n" + "=" * 50)
            print("📊 Final Configuration (after CLI overrides):")
            config.print_config_summary()
        
        # Initialize trainer with paths from config
        print("\n🚀 Initializing DCGAN trainer...")
        trainer = DCGANTrainer(config.dataset_spectrograms, config.dataset_spectrograms_metadata, config)
        
        # Start training
        print("🎯 Starting DCGAN training...")
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
