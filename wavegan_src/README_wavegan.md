# WaveGAN - Audio Generation Module

WaveGAN to implementacja Generative Adversarial Network przeznaczona do generowania 2-sekundowych próbek audio dźwięków silników spalinowych. Moduł wykorzystuje architekturę WGAN-GP (Wasserstein GAN with Gradient Penalty) pracującą bezpośrednio na surowych próbkach audio.

## 🎯 Cel Projektu

Generowanie realistycznych dźwięków silników spalinowych poprzez:

- Trening na rzeczywistych próbkach audio (44.1kHz, 2 sekundy)
- Wykorzystanie architektury 1D konwolucyjnej
- Implementację WGAN-GP dla stabilnego treningu

## 🏗️ Architektura

### Generator (`generator.py`)

- **Input**: Wektor szumu z rozkładu normalnego [batch_size, latent_dim]
- **Output**: Próbka audio [batch_size, 1, 44100]
- **Architektura**: Sieć transposed convolution 1D z upsampling
- **Warstwy**: FC → 6x ConvTranspose1D → Tanh

### Discriminator (`discriminator.py`)

- **Input**: Próbka audio [batch_size, 1, 44100]
- **Output**: Prawdopodobieństwo autentyczności [batch_size, 1]
- **Architektura**: Sieć convolution 1D z downsampling
- **Warstwy**: 6x Conv1D → FC → Linear output

### Training Pipeline (`training.py`)

- **Loss**: WGAN-GP z gradient penalty
- **Optimizer**: Adam (lr=0.0001, β1=0.5, β2=0.9)
- **Strategy**: n_critic=5 (5 kroków dyskryminatora na 1 krok generatora)

## 📁 Struktura Kodu

```
wavegan_src/
├── config.json              # Główna konfiguracja produkcyjna
├── config_test.json         # Konfiguracja testowa (szybka)
├── config_template.json     # Szablon z komentarzami
├── config_loader.py         # System ładowania konfiguracji
├── generator.py             # Architektura generatora
├── discriminator.py         # Architektura dyskryminatora
├── training.py              # Pipeline treningu
├── dataset.py               # Ładowanie datasetu audio
├── metrics.py               # Zbieranie metryk
├── checkpoints.py           # Zarządzanie checkpoint'ami
├── analytics.py             # Analiza post-training
├── convergence_detector.py  # Detekcja konwergencji
└── advanced_monitoring.py   # Monitoring zasobów
```

## ⚙️ Konfiguracja

### Podstawowe Parametry

```json
{
  "model": {
    "latent_dim": 100, // Wymiar przestrzeni latentnej
    "model_dim": 128, // Podstawowy wymiar modelu
    "kernel_len": 25 // Długość kernela konwolucyjnego
  },
  "training": {
    "batch_size": 128, // Rozmiar batcha
    "epochs": 5000, // Liczba epok
    "learning_rate": 0.0001, // Współczynnik uczenia
    "n_critic": 5, // Kroki dyskryminatora/generator
    "gp_weight": 10.0 // Waga gradient penalty
  },
  "audio": {
    "sample_rate": 22050, // Częstotliwość próbkowania
    "audio_length_seconds": 2.0,
    "audio_length_samples": 44100
  }
}
```

### Rodzaje Konfiguracji

1. **`config.json`** - Produkcyjna (5000 epok, batch=128)
2. **`config_test.json`** - Testowa (15 epok, batch=8, mniejszy model)
3. **`config_template.json`** - Szablon z szczegółowymi komentarzami

## 🚀 Uruchomienie

### Podstawowe Użycie

```bash
# Domyślna konfiguracja
python WaveGAN.py

# Konfiguracja testowa (szybka)
python WaveGAN.py --config config_test

# Własna konfiguracja
python WaveGAN.py --config custom_config.json
```

### CLI Overrides

```bash
# Nadpisanie parametrów treningu
python WaveGAN.py --epochs 1000 --batch-size 64 --learning-rate 0.0002

# Użycie CPU
python WaveGAN.py --device cpu

# Własny katalog wyjściowy
python WaveGAN.py --output-dir my_experiment
```

### Tylko Generowanie

```bash
# Wygeneruj 10 próbek z wytrenowanego modelu
python WaveGAN.py --generate-only 10
```

## 📊 Metryki i Monitoring

### Automatyczne Logowanie

- **Epoch Statistics**: Straty, gradient norms, użycie zasobów
- **Iteration Details**: Szczegółowe logi co 100 iteracji
- **Checkpoint Metrics**: STOI, PESQ, MCD scores per checkpoint

### Struktura Wyjść

```
output_wavegan/
├── epochs_statistics/
│   ├── epoch_statistics.csv      # Statystyki per epoka
│   ├── checkpoint_metrics.csv    # Metryki jakości modelu
│   └── experiment_config.json    # Snapshot konfiguracji
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
    └── convergence_analysis.png
```

## 🔧 Zaawansowane Funkcje

### Convergence Detection

- Automatyczna detekcja konwergencji na podstawie stabilności loss
- Early stopping przy wykryciu overfitting
- Adaptive learning rate scheduling

### Resource Monitoring

- Real-time monitoring GPU/CPU usage
- Memory tracking and optimization
- Performance profiling per epoch

### Quality Metrics

- **STOI** (Short-Time Objective Intelligibility)
- **PESQ** (Perceptual Evaluation of Speech Quality)
- **MCD** (Mel-Cepstral Distortion)
- **FAD** (Fréchet Audio Distance)

## 🛠️ Wymagania Systemowe

### Minimalne (config_test)

- GPU: 4GB VRAM
- RAM: 8GB
- Storage: 5GB

### Rekomendowane (config produkcyjne)

- GPU: RTX 4090 (24GB VRAM)
- RAM: 32GB+
- Storage: 50GB+

## 📈 Przykładowe Rezultaty

Po 5000 epokach treningu:

- **Generator Loss**: ~-50 do -100 (WGAN-GP)
- **Discriminator Loss**: ~-20 do -50 (WGAN-GP)
- **STOI Score**: >0.7 (dobra jakość)
- **Training Time**: ~48h na RTX 4090

## 🔍 Debugowanie

### Częste Problemy

1. **CUDA Out of Memory**

   ```bash
   python WaveGAN.py --batch-size 32 --num-workers 2
   ```

2. **Brak datasetu**

   ```bash
   # Sprawdź ścieżki w config.json
   "dataset": {
     "base_path": "../01_dataset_prep/dataset-processed/final"
   }
   ```

3. **Mode Collapse**
   - Zwiększ gradient penalty weight: `"gp_weight": 15.0`
   - Zmniejsz learning rate: `"learning_rate": 0.00005`

### Monitoring Treningu

```bash
# Obserwuj logi w czasie rzeczywistym
tail -f output_wavegan/epochs_statistics/epoch_statistics.csv

# Sprawdź ostatnie checkpointy
ls -la output_wavegan/checkpoints/
```

## 🎵 Audio Processing

### Format Wejściowy

- **Sample Rate**: 22050 Hz
- **Duration**: 2.0 sekundy
- **Channels**: Mono (1 kanał)
- **Format**: WAV, 16-bit

### Preprocessing

- Normalizacja do zakresu [-1, 1]
- Windowing i overlapping dla stabilności
- Spectral normalization (opcjonalne)

---

**Autor**: Projekt magisterski  
**Ostatnia aktualizacja**: Sierpień 2025  
**Licencja**: MIT
