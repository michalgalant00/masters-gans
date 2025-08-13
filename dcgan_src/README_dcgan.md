# DCGAN - Spectrogram Generation Module

DCGAN to implementacja Deep Convolutional Generative Adversarial Network przeznaczona do generowania spektrogramów dźwięków silników spalinowych. Moduł wykorzystuje standardową architekturę DCGAN pracującą na obrazach spektrogramów 2D.

## 🎯 Cel Projektu

Generowanie realistycznych spektrogramów dźwięków silników poprzez:

- Trening na spektrogramach wygenerowanych z rzeczywistych próbek audio
- Wykorzystanie klasycznej architektury DCGAN 2D
- Generowanie obrazów spektrogramów o rozdzielczości 128x128 pikseli
- **Zapis sampli w formacie NumPy (.npy)** dla lepszej jakości i elastyczności

## 🏗️ Architektura

### Generator (`generator.py`)

- **Input**: Wektor szumu z rozkładu normalnego [batch_size, latent_dim, 1, 1]
- **Output**: Spektrogram [batch_size, 1, 128, 128]
- **Architektura**: Sieć transposed convolution 2D z upsampling
- **Warstwy**: Linear → 5x ConvTranspose2D → Tanh
- **Pattern**: 4×4 → 8×8 → 16×16 → 32×32 → 64×64 → 128×128

### Discriminator (`discriminator.py`)

- **Input**: Spektrogram [batch_size, 1, 128, 128]
- **Output**: Prawdopodobieństwo autentyczności [batch_size, 1]
- **Architektura**: Sieć convolution 2D z downsampling
- **Warstwy**: 5x Conv2D → Sigmoid
- **Pattern**: 128×128 → 64×64 → 32×32 → 16×16 → 8×8 → 4×4 → 1

### Training Pipeline (`training.py`)

- **Loss**: Binary Cross-Entropy (standard DCGAN)
- **Optimizer**: Adam (lr=0.0002, β1=0.5)
- **Strategy**: Równoczesny trening generatora i dyskryminatora
- **Samples**: Zapisywane w formacie .npy (bez strat kompresji)

## 📁 Struktura Kodu

```
dcgan_src/
├── config.json              # Główna konfiguracja produkcyjna
├── config_test.json         # Konfiguracja testowa (szybka)
├── config_template.json     # Szablon z komentarzami
├── config_loader.py         # System ładowania konfiguracji
├── config.py                # Legacy config (backward compatibility)
├── generator.py             # Architektura generatora DCGAN
├── discriminator.py         # Architektura dyskryminatora DCGAN
├── training.py              # Pipeline treningu DCGAN
├── dataset.py               # Ładowanie datasetu spektrogramów
├── metrics.py               # Zbieranie metryk dla obrazów
├── npy_to_images.py         # Konwerter NumPy → obrazy
├── NPY_FORMAT_README.md     # Dokumentacja formatu .npy
├── checkpoints.py           # Zarządzanie checkpoint'ami
├── analytics.py             # Analiza post-training
├── convergence_detector.py  # Detekcja konwergencji
├── advanced_monitoring.py   # Monitoring zasobów
└── advanced_metrics.py      # Zaawansowane metryki obrazów
```

## ⚙️ Konfiguracja

### Podstawowe Parametry

```json
{
  "model": {
    "latent_dim": 100, // Wymiar przestrzeni latentnej
    "features_g": 64, // Liczba feature maps w generatorze
    "features_d": 64, // Liczba feature maps w dyskryminatorze
    "image_size": 128, // Rozmiar spektrogramu (128x128)
    "channels": 1 // Kanały (1 = grayscale)
  },
  "training": {
    "batch_size": 64, // Rozmiar batcha
    "epochs": 200, // Liczba epok
    "learning_rate": 0.0002, // Współczynnik uczenia
    "beta1": 0.5 // Beta1 dla Adam optimizer
  }
}
```

### Rodzaje Konfiguracji

1. **`config.json`** - Produkcyjna (200 epok, batch=64, image_size=128)
2. **`config_test.json`** - Testowa (10 epok, batch=8, image_size=64)
3. **`config_template.json`** - Szablon z szczegółowymi komentarzami

## 🚀 Uruchomienie

### Podstawowe Użycie

```bash
# Domyślna konfiguracja
python DCGAN.py

# Konfiguracja testowa (szybka)
python DCGAN.py --config config_test

# Własna konfiguracja
python DCGAN.py --config custom_config.json
```

### CLI Overrides

```bash
# Nadpisanie parametrów treningu
python DCGAN.py --epochs 100 --batch-size 32 --learning-rate 0.0001

# Parametry modelu
python DCGAN.py --image-size 64 --features-g 32 --features-d 32

# Użycie CPU
python DCGAN.py --device cpu

# Własny katalog wyjściowy
python DCGAN.py --output-dir my_experiment
```

## 📊 Metryki i Monitoring

### Automatyczne Logowanie

- **Epoch Statistics**: Straty generatora/dyskryminatora, gradient norms
- **Iteration Details**: Szczegółowe logi co 100 iteracji
- **Checkpoint Metrics**: FID, IS scores per checkpoint

### Struktura Wyjść

```
output_dcgan/
├── epochs_statistics/
│   ├── epoch_statistics.csv      # Statystyki per epoka
│   ├── checkpoint_metrics.csv    # Metryki jakości modelu
│   └── experiment_config.json    # Snapshot konfiguracji
├── samples/                       # 🆕 NOWY FORMAT!
│   ├── epoch_001.npy             # Sample NumPy (surowe dane)
│   ├── epoch_005.npy             # Format: (batch, channels, H, W)
│   ├── epoch_001_grid.png        # Opcjonalne: grid z konwertera
│   └── epoch_001_images/         # Opcjonalne: indywidualne obrazy
│       ├── epoch_001_sample_01.png
│       └── ...
├── checkpoints/
│   ├── checkpoint_epoch_10.tar   # Rolling buffer (max 2)
│   ├── best_model.tar           # Najlepszy model
│   └── final_model.tar          # Końcowy stan
├── single_epochs_statistics/
│   ├── epoch_1/
│   │   └── epoch_1_iterations.csv
│   └── epoch_N/...
└── plots/
    ├── epoch_loss_curves.png     # Auto-generated
    ├── gradient_norms.png
    ├── generated_samples_grid.png
    └── convergence_analysis.png
```

## 💾 Nowy Format Sampli (.npy)

DCGAN teraz zapisuje sample w formacie NumPy zamiast PNG dla lepszej jakości:

```bash
# Konwertuj sample do obrazów
python dcgan_src/npy_to_images.py output_dcgan/samples/epoch_001.npy

# Wszystkie pliki w katalogu
python dcgan_src/npy_to_images.py --batch-convert output_dcgan/samples/

# Szczegóły: dcgan_src/NPY_FORMAT_README.md
```

## 🔧 Zaawansowane Funkcje

### Image Quality Metrics

- **FID** (Fréchet Inception Distance)
- **IS** (Inception Score)
- **SSIM** (Structural Similarity Index)
- **LPIPS** (Learned Perceptual Image Patch Similarity)

### Convergence Detection

- Monitoring stabilności loss functions
- Early stopping przy mode collapse
- Adaptive learning rate scheduling

### Resource Monitoring

- Real-time monitoring GPU/CPU usage
- Memory tracking per batch
- Performance profiling per epoch

## 🖼️ Spectrogram Processing

### Format Spektrogramów

- **Resolution**: 128×128 pikseli (production) / 64×64 (test)
- **Channels**: 1 (grayscale)
- **Format**: PNG, normalized [0, 1]
- **Frequency Range**: 0-11kHz (dla 22kHz audio)

### Preprocessing Pipeline

1. Audio → STFT (Short-Time Fourier Transform)
2. Magnitude spectrum calculation
3. Log-scale transformation
4. Normalization to [0, 1]
5. Resize to target resolution

## 🛠️ Wymagania Systemowe

### Minimalne (config_test)

- GPU: 4GB VRAM
- RAM: 8GB
- Storage: 2GB

### Rekomendowane (config produkcyjne)

- GPU: RTX 4090 (24GB VRAM) lub RTX 3080+ (10GB+)
- RAM: 16GB+
- Storage: 20GB+

## 📈 Przykładowe Rezultaty

Po 200 epokach treningu:

- **Generator Loss**: ~1.0-3.0 (BCE)
- **Discriminator Loss**: ~0.5-1.5 (BCE)
- **FID Score**: <50 (dobra jakość)
- **Training Time**: ~12h na RTX 4090

## 🔍 Debugowanie

### Częste Problemy

1. **Mode Collapse**

   ```bash
   # Zmniejsz learning rate
   python DCGAN.py --learning-rate 0.0001

   # Zwiększ regularyzację
   # Dodaj noise do dyskryminatora w config
   ```

2. **Discriminator Too Strong**

   ```bash
   # Balansuj trening
   python DCGAN.py --learning-rate 0.0002  # dla obu sieci
   ```

3. **CUDA Out of Memory**

   ```bash
   python DCGAN.py --batch-size 16 --image-size 64
   ```

4. **Brak spektrogramów**
   ```bash
   # Sprawdź ścieżki w config.json
   "dataset": {
     "spectrograms_path": "spectrograms/"
   }
   ```

### Monitoring Jakości

```bash
# Sprawdź postęp treningu
tail -f output_dcgan/epochs_statistics/epoch_statistics.csv

# Obejrzyj wygenerowane próbki
ls output_dcgan/checkpoints/*/samples/
```

## 🎨 Generowanie Próbek

### 🆕 Nowy Format NumPy (.npy)

DCGAN zapisuje teraz sample w formacie NumPy dla lepszej jakości:

```bash
# Basic conversion
python dcgan_src/npy_to_images.py output_dcgan/samples/epoch_001.npy

# Grid format
python dcgan_src/npy_to_images.py output_dcgan/samples/epoch_001.npy --mode grid

# Batch convert all files
python dcgan_src/npy_to_images.py --batch-convert output_dcgan/samples/

# Custom format and quality
python dcgan_src/npy_to_images.py epoch_001.npy --format jpeg --quality 95
```

### Programmatic Access

```python
import numpy as np

# Load samples
samples = np.load('output_dcgan/samples/epoch_001.npy')
print(f"Shape: {samples.shape}")  # (batch, channels, height, width)
print(f"Range: [{samples.min():.3f}, {samples.max():.3f}]")

# Access individual sample
first_sample = samples[0]  # Shape: (1, 64, 64) for grayscale
```

### Sample Grid Generation

- Automatyczne generowanie próbek co epokę
- Format: (batch_size, 1, 64x64) dla testów, (batch_size, 1, 128x128) produkcyjnie
- Zapis w formacie .npy bez strat kompresji

### Custom Generation

```python
# Wygeneruj własne spektrogramy
from dcgan_src.generator import DCGANGenerator
import torch
import numpy as np

generator = DCGANGenerator()
generator.load_state_dict(torch.load('best_model.tar')['generator'])
noise = torch.randn(9, 100, 1, 1)
samples = generator(noise)

# Save as .npy
samples_np = samples.detach().cpu().numpy()
samples_np = (samples_np + 1.0) / 2.0  # Denormalize
np.save('my_samples.npy', samples_np)
```

## 🔄 Pipeline Audio → Spektrogram → Audio

1. **Audio → Spektrogram**: STFT preprocessing
2. **Spektrogram → GAN**: DCGAN training/generation
3. **Spektrogram → Audio**: Griffin-Lim reconstruction

### Reconstruction Quality

- **Original**: High fidelity (źródłowy STFT)
- **Generated**: Medium fidelity (artifacts z Griffin-Lim)
- **Improvement**: Używaj WaveGAN dla lepszej jakości audio

## 🧪 Experimenty i Tuning

### Hyperparameter Ranges

- **Learning Rate**: 0.0001-0.0005
- **Batch Size**: 16-128 (zależy od GPU)
- **Features**: 32-128 (trade-off speed/quality)
- **Image Size**: 64, 128, 256

### Ablation Studies

- Wpływ batch size na stabilność
- Features_g vs features_d balance
- Image resolution vs training time

---

**Autor**: Projekt magisterski  
**Ostatnia aktualizacja**: Sierpień 2025  
**Licencja**: MIT
