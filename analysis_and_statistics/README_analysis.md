# Analysis and Statistics Module

Moduł analizy i statystyk zapewnia kompleksowe zbieranie, przetwarzanie i wizualizację metryk treningu dla obu architektur GAN (WaveGAN i DCGAN). Implementuje system zgodny z wytycznymi z `metrics-checkpoints-how-to.txt`.

## 🎯 Cel Modułu

Zapewnienie kompleksowego systemu:

- **Metrics Collection**: Zbieranie metryk treningu w czasie rzeczywistym
- **Checkpoint Management**: Zarządzanie modelami i rolling buffer
- **Post-training Analysis**: Automatyczna analiza i wizualizacja wyników
- **Resource Monitoring**: Monitoring zasobów systemowych

## 🏗️ Architektura Modułu

### Struktura Katalogów

```
analysis_and_statistics/
├── base/                     # Funkcjonalność bazowa
│   ├── analytics_base.py     # Klasa bazowa dla analiz
│   ├── checkpoint_manager.py # Zarządzanie checkpoint'ami
│   ├── metrics_collector.py  # Zbieranie metryk
│   ├── post_training_analyzer.py # Analiza post-training
│   └── system_monitor.py     # Monitoring zasobów
├── audio/                    # Specificzne dla WaveGAN
│   ├── audio_analytics.py    # Analiza audio
│   ├── audio_checkpoint.py   # Checkpoint'y audio
│   └── audio_metrics.py      # Metryki audio (STOI, PESQ, etc.)
├── image/                    # Specificzne dla DCGAN
│   ├── image_analytics.py    # Analiza obrazów
│   ├── image_checkpoint.py   # Checkpoint'y obrazów
│   └── image_metrics.py      # Metryki obrazów (FID, IS, etc.)
└── utils/                    # Narzędzia pomocnicze
    ├── file_utils.py         # Operacje na plikach
    └── plot_utils.py         # Narzędzia do wykresów
```

## 📊 System Metryk

### 1. Statystyki Per Epoka (`epoch_statistics.csv`)

```csv
epoch,timestamp,avg_generator_loss,avg_discriminator_loss,min_generator_loss,max_generator_loss,min_discriminator_loss,max_discriminator_loss,avg_grad_norm_g,avg_grad_norm_d,max_grad_norm_g,max_grad_norm_d,learning_rate_g,learning_rate_d,gpu_memory_used,gpu_utilization,cpu_usage,ram_usage,epoch_duration_minutes,convergence_indicator,training_stability_score,batch_size,total_iterations_in_epoch
```

### 2. Metryki Checkpoint'ów (`checkpoint_metrics.csv`)

```csv
checkpoint_epoch,timestamp,generator_params_count,discriminator_params_count,model_performance_score,checkpoint_file_path,sample_1_path,sample_2_path,sample_3_path
```

### 3. Szczegóły Iteracji (`epoch_N_iterations.csv`)

```csv
iteration,generator_loss,discriminator_loss,generator_grad_norm,discriminator_grad_norm,gpu_memory_used,iteration_time_seconds,timestamp
```

## 🗂️ Struktura Zapisu (Zoptymalizowana)

```
output_analysis/
├── epochs_statistics/
│   ├── epoch_statistics.csv        # Główne statystyki per epoka
│   ├── checkpoint_metrics.csv      # Jakość modelu per checkpoint
│   └── experiment_config.json      # Konfiguracja + model info
├── checkpoints/
│   ├── checkpoint_epoch_10.tar     # Rolling buffer (max 2)
│   ├── checkpoint_epoch_20.tar
│   ├── best_model.tar             # Najlepszy model (osobno)
│   └── final_model.tar            # Końcowy stan
├── single_epochs_statistics/
│   ├── epoch_1/
│   │   └── epoch_1_iterations.csv  # Snapshots co 100 iteracji
│   ├── epoch_2/
│   │   └── epoch_2_iterations.csv
│   └── epoch_N/...
└── plots/                          # Auto-generated po treningu
    ├── epoch_loss_curves.png
    ├── gradient_norms.png
    ├── resource_usage.png
    ├── convergence_analysis.png
    └── training_summary.json
```

## 🔧 Kluczowe Komponenty

### MetricsCollector (`base/metrics_collector.py`)

```python
class MetricsCollector:
    """Zbieranie i agregacja metryk treningu"""

    def log_epoch_stats(self, epoch, losses, grads, lr, gpu_stats, timer):
        """Log głównych statystyk per epoka"""

    def log_iteration_snapshot(self, epoch, iter, losses, grads, gpu_stats):
        """Log snapshot co 100 iteracji"""

    def aggregate_epoch_metrics(self, iteration_buffer):
        """Agregacja metryk z iteracji w epoce"""
```

### CheckpointManager (`base/checkpoint_manager.py`)

```python
class CheckpointManager:
    """Zarządzanie checkpoint'ami z rolling buffer"""

    def save_checkpoint(self, epoch, models, optimizers, context):
        """Zapisz checkpoint co 10 epok z rolling buffer (max 2)"""

    def load_checkpoint(self, checkpoint_path):
        """Pełne odtworzenie stanu dla wznowienia treningu"""

    def cleanup_old_checkpoints(self):
        """Auto-cleanup: usuwanie najstarszego przy zapisie nowego"""
```

### PostTrainingAnalyzer (`base/post_training_analyzer.py`)

```python
class PostTrainingAnalyzer:
    """Automatyczna analiza po treningu"""

    def generate_training_report(self, output_dir):
        """Auto-generate plots i summary"""

    def analyze_convergence_patterns(self):
        """Analiza wzorców konwergencji"""

    def identify_best_checkpoint(self):
        """Identyfikacja najlepszego modelu"""
```

## 📈 Automatyczne Wykresy

### 1. `epoch_loss_curves.png`

- Generator vs Discriminator losses over epochs
- Trend lines i moving averages
- Identyfikacja punktów konwergencji

### 2. `gradient_norms.png`

- Training stability per epoch
- Gradient explosion/vanishing detection
- Stability score visualization

### 3. `resource_usage.png`

- GPU memory/utilization over epochs
- CPU usage trends
- RAM consumption patterns

### 4. `convergence_analysis.png`

- Loss variance trends per epoch
- Convergence indicators
- Training stability metrics

## 🎵 Audio-Specific Metrics (WaveGAN)

### Implemented in `audio/audio_metrics.py`

```python
class AudioMetrics:
    """Metryki jakości audio"""

    def calculate_stoi(self, original, generated):
        """Short-Time Objective Intelligibility"""

    def calculate_pesq(self, original, generated):
        """Perceptual Evaluation of Speech Quality"""

    def calculate_mcd(self, original, generated):
        """Mel-Cepstral Distortion"""

    def calculate_fad(self, original_features, generated_features):
        """Fréchet Audio Distance"""
```

### Częstotliwość Metryk Audio

- **STOI/PESQ/MCD**: Per checkpoint (co 10 epok)
- **FAD**: Per milestone (co 1000 iteracji)
- **SNR**: Per epoka (w epoch_statistics.csv)

## 🖼️ Image-Specific Metrics (DCGAN)

### Implemented in `image/image_metrics.py`

```python
class ImageMetrics:
    """Metryki jakości obrazów"""

    def calculate_fid(self, real_images, generated_images):
        """Fréchet Inception Distance"""

    def calculate_inception_score(self, generated_images):
        """Inception Score"""

    def calculate_ssim(self, real_images, generated_images):
        """Structural Similarity Index"""

    def calculate_lpips(self, real_images, generated_images):
        """Learned Perceptual Image Patch Similarity"""
```

### Częstotliwość Metryk Image

- **FID/IS**: Per checkpoint (co 10 epok)
- **SSIM/LPIPS**: Per milestone (co 1000 iteracji)
- **Pixel-level metrics**: Per epoka

## 🖥️ System Monitoring

### ResourceMonitor (`base/system_monitor.py`)

```python
class ResourceMonitor:
    """Monitoring zasobów systemowych"""

    def get_gpu_stats(self):
        """GPU memory, utilization, temperature"""

    def get_cpu_stats(self):
        """CPU usage, load average"""

    def get_memory_stats(self):
        """RAM usage, swap, available memory"""

    def get_disk_io(self):
        """Disk read/write statistics"""
```

### Zbierane Metryki

- **GPU**: Memory usage, utilization %, temperature
- **CPU**: Usage %, load average, process count
- **RAM**: Used/total, swap usage, available
- **Disk**: I/O read/write rates per process

## ⚙️ Konfiguracja Metryk

### W config.json każdej architektury:

```json
{
  "metrics_and_checkpoints": {
    "csv_write_frequency": 100, // Co ile iteracji CSV
    "detailed_logging_frequency": 100, // Co ile iteracji detale
    "emergency_checkpoint_frequency": 100, // Emergency backup
    "milestone_checkpoint_frequency": 1000, // Milestone checkpoints
    "milestone_epoch_frequency": 10, // Co ile epok checkpoint
    "emergency_buffer_size": 5, // Rolling buffer size
    "audio_samples_per_checkpoint": 5, // Próbek na checkpoint
    "best_model_improvement_threshold": 0.05, // Threshold dla best model
    "key_epochs_first": 5, // Pierwsze epoki (zawsze save)
    "key_epochs_last": 5 // Ostatnie epoki (zawsze save)
  }
}
```

## 🔄 Integration z Training Pipeline

### WaveGAN Integration

```python
# W wavegan_src/training.py
from analysis_and_statistics.audio import AudioMetrics, AudioCheckpoint
from analysis_and_statistics.base import MetricsCollector, PostTrainingAnalyzer

class WaveGANTrainer:
    def __init__(self, config):
        self.metrics = MetricsCollector(output_dir=config.output_dir)
        self.audio_metrics = AudioMetrics()
        self.checkpoint_manager = AudioCheckpoint(output_dir=config.output_dir)

    def train_epoch(self, epoch):
        # Training loop...
        self.metrics.log_epoch_stats(epoch, losses, grads, lr, gpu_stats, timer)

    def save_checkpoint(self, epoch):
        if epoch % 10 == 0:
            audio_samples = self.generate_samples(num_samples=5)
            quality_scores = self.audio_metrics.evaluate_checkpoint(audio_samples)
            self.checkpoint_manager.save_with_audio_metrics(epoch, models, quality_scores)
```

### DCGAN Integration

```python
# W dcgan_src/training.py
from analysis_and_statistics.image import ImageMetrics, ImageCheckpoint
from analysis_and_statistics.base import MetricsCollector, PostTrainingAnalyzer

class DCGANTrainer:
    def __init__(self, config):
        self.metrics = MetricsCollector(output_dir=config.output_dir)
        self.image_metrics = ImageMetrics()
        self.checkpoint_manager = ImageCheckpoint(output_dir=config.output_dir)

    def train_epoch(self, epoch):
        # Training loop...
        self.metrics.log_epoch_stats(epoch, losses, grads, lr, gpu_stats, timer)

    def save_checkpoint(self, epoch):
        if epoch % 10 == 0:
            generated_images = self.generate_samples(num_samples=9)
            quality_scores = self.image_metrics.evaluate_checkpoint(generated_images)
            self.checkpoint_manager.save_with_image_metrics(epoch, models, quality_scores)
```

## 📊 Post-Training Report

### Automatyczny Summary Report (`training_summary.json`)

```json
{
  "training_summary": {
    "total_time": "48.5 hours",
    "epochs": 5000,
    "total_iterations": 125000,
    "best_losses": {
      "generator": -85.2,
      "discriminator": -45.1,
      "epoch": 4250
    },
    "final_losses": {
      "generator": -78.9,
      "discriminator": -42.3
    },
    "convergence_epoch": 3800,
    "resource_efficiency_score": 0.87,
    "model_parameter_counts": {
      "generator": 15234567,
      "discriminator": 12456789
    },
    "recommended_checkpoint": "checkpoint_epoch_4250.tar",
    "epoch_performance_trends": {
      "improvement_rate": 0.023,
      "stability_score": 0.91
    }
  }
}
```

## 🚀 Używanie Modułu

### Standalone Analysis

```python
from analysis_and_statistics.base import PostTrainingAnalyzer

# Analiza po zakończonym treningu
analyzer = PostTrainingAnalyzer("output_wavegan")
analyzer.generate_training_report()
analyzer.create_final_visualizations()
```

### Real-time Monitoring

```python
from analysis_and_statistics.base import MetricsCollector

# W trakcie treningu
metrics = MetricsCollector("output_wavegan")
metrics.start_real_time_monitoring()
```

## 🔍 Optymalizacje Wydajności

### Storage Optimization

- CSV z kompresją gzip dla epoch_statistics
- Checkpoint'y jako .tar.gz (50-70% redukcja)
- Audio samples w 16-bit mono
- Rolling buffer (max 2 pliki)
- Auto-cleanup starszych niż 50 epok

### Memory Optimization

- Lazy loading metrics data
- Batch compression dla iteration snapshots
- Metrics aggregation w epoch_statistics
- Streaming CSV writes

### Expected Storage

- **Epoch statistics**: ~5-20MB (zależnie od liczby epok)
- **Single epochs details**: ~2-10MB per epoch
- **Checkpoints**: ~100-500MB per checkpoint (max 2 + best + final)
- **Total footprint**: 3-10GB dla pełnego 200-epoch training run

---

**Autor**: Projekt magisterski  
**Ostatnia aktualizacja**: Sierpień 2025  
**Licencja**: MIT
