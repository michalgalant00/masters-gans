# GANs Implementation - DCGAN & WaveGAN

DCGAN implementation for spectrogram generation and WaveGAN implementation for engine sound generation.

## Scripts usage

### getting started

Create and activate a virtual environment, then install the required dependencies.

```bash
./setup_environment.ps1
```

After that test runs are recommended.

### run test dcgan pipeline

```bash
python DCGAN.py --config config_test
```

Quick DCGAN test with simplified configuration (10 epochs, batch 8, 64x64 images)

### run test wavegan pipeline

```bash
python WaveGAN.py --config config_test
```

Quick WaveGAN test with simplified configuration (15 epochs, 0.5s audio, batch 8)

### run main dcgan pipeline

```bash
python DCGAN.py
# or explicitly:
python DCGAN.py --config config
```

Production DCGAN training (500 epochs, batch 128, 128x128 images, cloud storage)

### run main wavegan pipeline

```bash
python WaveGAN.py
# or explicitly:
python WaveGAN.py --config config
```

Production WaveGAN training (5000 epochs, batch 128, 2s audio, cloud storage)

## cmd args for dcgan

```bash
# Basic usage
python DCGAN.py

# Configuration selection
python DCGAN.py --config config_test
python DCGAN.py --config path/to/custom.json

# Override training parameters
python DCGAN.py --epochs 100 --batch-size 32 --learning-rate 0.0001

# Model parameters
python DCGAN.py --image-size 256 --features-g 64 --features-d 64 --latent-dim 128

# Hardware settings
python DCGAN.py --device cpu --num-workers 4

# Output directory
python DCGAN.py --output-dir ./my_output
```

## cmd args for wavegan

```bash
# Basic usage
python WaveGAN.py

# Configuration selection
python WaveGAN.py --config config_test
python WaveGAN.py --config path/to/custom.json

# Override training parameters
python WaveGAN.py --epochs 1000 --batch-size 64 --learning-rate 0.0002

# Model parameters
python WaveGAN.py --model-dim 64 --audio-length 22050 --latent-dim 128

# Hardware settings
python WaveGAN.py --device cuda --num-workers 8

# Special options
python WaveGAN.py --generate-only 50  # generate samples only
python WaveGAN.py --no-training       # load model without training
```

## usage of configs json files

### Configuration structure:

- **`config.json`** - Main configuration
- **`config_gcloud.json`** - GC configuration with cloud storage (GCS)
- **`config_test.json`** - Lightweight test configuration with local storage

### configs:

All editable configurations located in `_src` directories.

- **Main**
- **GCloud**
- **Test**

### Custom configuration:

```bash
# Copy template and customize
cp dcgan_src/config_template.json my_config.json
python DCGAN.py --config my_config.json
```
