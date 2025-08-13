"""
Audio Analytics
===============

WaveGAN-specific analytics and visualization tools.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any
import pandas as pd

# Import base class
from ..base.analytics_base import AnalyticsBase


class AudioAnalytics(AnalyticsBase):
    """Analytics for audio-based GANs (WaveGAN)"""
    
    def __init__(self, output_dir: str = "output_wavegan"):
        super().__init__(output_dir, model_type="audio")
    
    def _generate_domain_specific_plots(self):
        """Generate WaveGAN-specific plots"""
        try:
            self.plot_discriminator_scores()
            self.plot_audio_quality_metrics()
            self.plot_spectral_analysis()
            
        except Exception as e:
            print(f"❌ Error generating audio-specific plots: {e}")
    
    def plot_discriminator_scores(self):
        """Plot discriminator performance metrics for audio"""
        try:
            df = pd.read_csv(self.training_log_path)
            if df.empty or 'avg_d_real_score' not in df.columns:
                print("⚠️  No discriminator score data available")
                return
            
            plt.figure(figsize=(15, 10))
            
            # Real vs Fake scores over time
            plt.subplot(2, 3, 1)
            plt.plot(df['epoch'], df['avg_d_real_score'], label='Real Score', alpha=0.7, marker='o')
            plt.plot(df['epoch'], df['avg_d_fake_score'], label='Fake Score', alpha=0.7, marker='s')
            plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
            plt.xlabel('Epoch')
            plt.ylabel('Discriminator Score')
            plt.title('Audio Discriminator Scores Over Time')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Score distributions
            plt.subplot(2, 3, 2)
            plt.hist(df['avg_d_real_score'], bins=50, alpha=0.7, label='Real Audio', density=True)
            plt.hist(df['avg_d_fake_score'], bins=50, alpha=0.7, label='Fake Audio', density=True)
            plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
            plt.xlabel('Score')
            plt.ylabel('Density')
            plt.title('Audio Score Distributions')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Score difference (real - fake)
            plt.subplot(2, 3, 3)
            score_diff = df['avg_d_real_score'] - df['avg_d_fake_score']
            plt.plot(df['epoch'], score_diff, color='purple', alpha=0.7, marker='o')
            plt.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Balanced')
            plt.xlabel('Epoch')
            plt.ylabel('Real Score - Fake Score')
            plt.title('Audio Discriminator Balance')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Moving averages (skip for epoch data - too few points)
            plt.subplot(2, 3, 4)
            plt.plot(df['epoch'], df['avg_d_real_score'], label='Real Score', linewidth=2, marker='o')
            plt.plot(df['epoch'], df['avg_d_fake_score'], label='Fake Score', linewidth=2, marker='s')
            plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
            plt.xlabel('Epoch')
            plt.ylabel('Discriminator Score')
            plt.title('Audio Discriminator Scores by Epoch')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Score variance (simplified for epoch data)
            plt.subplot(2, 3, 5)
            if len(df) > 1:
                # Use standard deviation of scores across epochs
                plt.plot(df['epoch'], df['avg_d_real_score'], label='Real Score', alpha=0.7, marker='o')
                plt.plot(df['epoch'], df['avg_d_fake_score'], label='Fake Score', alpha=0.7, marker='s')
            plt.xlabel('Epoch')
            plt.ylabel('Score Value')
            plt.title('Audio Discriminator Score Trends')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Discriminator accuracy if available
            plt.subplot(2, 3, 6)
            if 'd_real_accuracy' in df.columns and 'd_fake_accuracy' in df.columns:
                plt.plot(df['epoch'], df['d_real_accuracy'], label='Real Accuracy', alpha=0.7, marker='o')
                plt.plot(df['epoch'], df['d_fake_accuracy'], label='Fake Accuracy', alpha=0.7, marker='s')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.title('Discriminator Accuracy')
                plt.legend()
                plt.grid(True, alpha=0.3)
            else:
                plt.text(0.5, 0.5, 'No accuracy\ndata available', ha='center', va='center',
                        transform=plt.gca().transAxes)
                plt.title('Discriminator Accuracy')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'audio_discriminator_analysis.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print("📊 Audio discriminator analysis plot saved")
            
        except Exception as e:
            print(f"❌ Error plotting audio discriminator scores: {e}")
    
    def plot_audio_quality_metrics(self):
        """Plot audio quality metrics over training"""
        try:
            if not os.path.exists(self.checkpoint_metrics_path):
                print("⚠️  No checkpoint metrics data available")
                return
                
            df = pd.read_csv(self.checkpoint_metrics_path)
            if df.empty:
                print("ℹ️  Checkpoint metrics file exists but contains no data")
                print("ℹ️  This is normal for test configurations without advanced audio quality metrics")
                return
            
            # Check if any audio quality metrics columns exist
            audio_metrics_columns = ['fad_score', 'snr_score', 'audio_quality_score']
            available_metrics = [col for col in audio_metrics_columns if col in df.columns and not df[col].isna().all()]
            
            if not available_metrics:
                print("ℹ️  No audio quality metrics data found in checkpoint file")
                print("ℹ️  Available columns:", list(df.columns))
                print("ℹ️  This is normal for test configurations without advanced audio quality metrics")
                return
                
            print(f"📊 Found audio quality metrics: {available_metrics}")
            
            plt.figure(figsize=(15, 10))
            
            # FAD score over time
            if 'fad_score' in df.columns:
                plt.subplot(2, 3, 1)
                plt.plot(df['epoch'], df['fad_score'], marker='o', alpha=0.7)
                plt.xlabel('Epoch')
                plt.ylabel('FAD Score')
                plt.title('Fréchet Audio Distance Over Training (Lower is Better)')
                plt.grid(True, alpha=0.3)
            
            # SNR score over time
            if 'snr_score' in df.columns:
                plt.subplot(2, 3, 2)
                plt.plot(df['epoch'], df['snr_score'], marker='s', alpha=0.7, color='green')
                plt.xlabel('Epoch')
                plt.ylabel('SNR (dB)')
                plt.title('Signal-to-Noise Ratio Over Training (Higher is Better)')
                plt.grid(True, alpha=0.3)
            
            # Audio quality score over time
            if 'audio_quality_score' in df.columns:
                plt.subplot(2, 3, 3)
                plt.plot(df['epoch'], df['audio_quality_score'], marker='^', alpha=0.7, color='purple')
                plt.xlabel('Epoch')
                plt.ylabel('Quality Score')
                plt.title('Composite Audio Quality Score')
                plt.grid(True, alpha=0.3)
            
            # FAD vs SNR correlation
            if 'fad_score' in df.columns and 'snr_score' in df.columns:
                plt.subplot(2, 3, 4)
                plt.scatter(df['fad_score'], df['snr_score'], alpha=0.7)
                plt.xlabel('FAD Score')
                plt.ylabel('SNR (dB)')
                plt.title('FAD vs SNR')
                plt.grid(True, alpha=0.3)
                
                # Add correlation coefficient
                try:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        correlation = df['fad_score'].corr(df['snr_score'])
                        if pd.isna(correlation):
                            correlation = 0.0
                except:
                    correlation = 0.0
                plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                        transform=plt.gca().transAxes, bbox=dict(boxstyle='round', facecolor='wheat'))
            
            # Quality metrics distribution
            plt.subplot(2, 3, 5)
            metrics_to_plot = []
            if 'fad_score' in df.columns:
                metrics_to_plot.append(('FAD', df['fad_score']))
            if 'snr_score' in df.columns:
                metrics_to_plot.append(('SNR', df['snr_score']))
            if 'audio_quality_score' in df.columns:
                metrics_to_plot.append(('Quality', df['audio_quality_score']))
            
            if metrics_to_plot:
                positions = range(len(metrics_to_plot))
                values = [metric[1] for metric in metrics_to_plot]
                labels = [metric[0] for metric in metrics_to_plot]
                
                box_plot = plt.boxplot(values, positions=positions)
                plt.xticks(positions, labels)
                plt.ylabel('Score')
                plt.title('Audio Quality Metrics Distribution')
                plt.grid(True, alpha=0.3)
            
            # Quality improvement trend
            plt.subplot(2, 3, 6)
            if 'audio_quality_score' in df.columns and len(df) > 5:
                # Calculate trend
                x = np.arange(len(df))
                z = np.polyfit(x, df['audio_quality_score'], 1)
                p = np.poly1d(z)
                
                plt.plot(df['epoch'], df['audio_quality_score'], 'o', alpha=0.7, label='Quality Score')
                plt.plot(df['epoch'], p(x), "r--", alpha=0.8, label='Trend')
                plt.xlabel('Epoch')
                plt.ylabel('Quality Score')
                plt.title('Audio Quality Improvement Trend')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Add trend info
                slope = z[0]
                trend_text = f'Trend: {"Improving" if slope > 0 else "Declining"}\nSlope: {slope:.3f}'
                plt.text(0.05, 0.95, trend_text, transform=plt.gca().transAxes,
                        bbox=dict(boxstyle='round', facecolor='wheat'))
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'audio_quality_metrics.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print("📊 Audio quality metrics plot saved")
            
        except Exception as e:
            print(f"❌ Error plotting audio quality metrics: {e}")
    
    def plot_spectral_analysis(self):
        """Plot spectral analysis of generated audio"""
        try:
            # Look for audio sample directories
            samples_base_dir = os.path.join(self.output_dir, "samples")
            if not os.path.exists(samples_base_dir):
                print("⚠️  No audio samples directory found")
                return
            
            # Find audio files - check both direct files and subdirectories
            audio_files_with_paths = []
            
            # First, check for direct audio files in samples directory
            for ext in ['.wav', '.flac', '.mp3']:
                direct_files = [f for f in os.listdir(samples_base_dir) 
                               if f.endswith(ext) and os.path.isfile(os.path.join(samples_base_dir, f))]
                for f in direct_files:
                    audio_files_with_paths.append(os.path.join(samples_base_dir, f))
            
            # Then check subdirectories for audio files
            for subdir in os.listdir(samples_base_dir):
                subdir_path = os.path.join(samples_base_dir, subdir)
                if os.path.isdir(subdir_path):
                    for ext in ['.wav', '.flac', '.mp3']:
                        subdir_files = [f for f in os.listdir(subdir_path) if f.endswith(ext)]
                        for f in subdir_files:
                            audio_files_with_paths.append(os.path.join(subdir_path, f))
            
            if not audio_files_with_paths:
                print("⚠️  No audio files found for spectral analysis")
                return
            
            print(f"📊 Found {len(audio_files_with_paths)} audio files for spectral analysis")
            
            # Analyze up to 4 audio files
            plt.figure(figsize=(15, 10))
            
            for i, audio_path in enumerate(audio_files_with_paths[:4]):
                audio_file = os.path.basename(audio_path)
                
                try:
                    # Load audio file
                    import soundfile as sf
                    audio_data, sample_rate = sf.read(audio_path)
                    
                    # Ensure audio is 1D
                    if len(audio_data.shape) > 1:
                        audio_data = audio_data.mean(axis=1)
                    
                    # Time domain plot
                    plt.subplot(4, 2, i*2 + 1)
                    time_axis = np.arange(len(audio_data)) / sample_rate
                    plt.plot(time_axis, audio_data, alpha=0.7)
                    plt.xlabel('Time (s)')
                    plt.ylabel('Amplitude')
                    plt.title(f'Waveform: {audio_file}')
                    plt.grid(True, alpha=0.3)
                    
                    # Frequency domain plot
                    plt.subplot(4, 2, i*2 + 2)
                    fft = np.fft.fft(audio_data)
                    magnitude = np.abs(fft)
                    freq_axis = np.fft.fftfreq(len(audio_data), 1/sample_rate)
                    
                    # Plot positive frequencies only
                    positive_freqs = freq_axis[:len(freq_axis)//2]
                    positive_magnitude = magnitude[:len(magnitude)//2]
                    
                    plt.semilogy(positive_freqs, positive_magnitude, alpha=0.7)
                    plt.xlabel('Frequency (Hz)')
                    plt.ylabel('Magnitude')
                    plt.title(f'Spectrum: {audio_file}')
                    plt.grid(True, alpha=0.3)
                    
                except Exception as e:
                    print(f"⚠️  Error analyzing {audio_file}: {e}")
                    
                    # Create placeholder plots
                    plt.subplot(4, 2, i*2 + 1)
                    plt.text(0.5, 0.5, f'Error loading\n{audio_file}', 
                            ha='center', va='center', transform=plt.gca().transAxes)
                    plt.title(f'Waveform: {audio_file}')
                    
                    plt.subplot(4, 2, i*2 + 2)
                    plt.text(0.5, 0.5, f'Error loading\n{audio_file}', 
                            ha='center', va='center', transform=plt.gca().transAxes)
                    plt.title(f'Spectrum: {audio_file}')
            
            # Hide unused subplots
            for i in range(len(audio_files_with_paths), 4):
                plt.subplot(4, 2, i*2 + 1)
                plt.axis('off')
                plt.subplot(4, 2, i*2 + 2)
                plt.axis('off')
            
            plt.suptitle('Spectral Analysis of Generated Audio')
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'audio_spectral_analysis.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print("📊 Audio spectral analysis plot saved")
            
        except Exception as e:
            print(f"❌ Error plotting spectral analysis: {e}")
    
    def analyze_generated_audio(self, samples_dir: str) -> Dict[str, Any]:
        """Analyze characteristics of generated audio samples
        
        Args:
            samples_dir: Directory containing generated audio samples
            
        Returns:
            Dictionary with analysis results
        """
        try:
            if not os.path.exists(samples_dir):
                return {}
            
            # Find all audio files
            audio_files = []
            for ext in ['.wav', '.flac', '.mp3']:
                audio_files.extend([f for f in os.listdir(samples_dir) if f.endswith(ext)])
            
            if not audio_files:
                return {'error': 'No audio files found'}
            
            # Analyze sample characteristics
            sample_stats = {
                'total_samples': len(audio_files),
                'amplitude_statistics': {},
                'duration_statistics': {},
                'spectral_statistics': {}
            }
            
            amplitudes = []
            durations = []
            spectral_centroids = []
            
            for audio_file in audio_files[:20]:  # Analyze up to 20 files
                audio_path = os.path.join(samples_dir, audio_file)
                try:
                    import soundfile as sf
                    audio_data, sample_rate = sf.read(audio_path)
                    
                    # Ensure audio is 1D
                    if len(audio_data.shape) > 1:
                        audio_data = audio_data.mean(axis=1)
                    
                    # Collect statistics
                    amplitudes.extend(audio_data.flatten())
                    durations.append(len(audio_data) / sample_rate)
                    
                    # Calculate spectral centroid
                    fft = np.fft.fft(audio_data)
                    magnitude = np.abs(fft)
                    freqs = np.fft.fftfreq(len(audio_data), 1/sample_rate)
                    
                    # Spectral centroid (weighted average frequency)
                    positive_freqs = freqs[:len(freqs)//2]
                    positive_magnitude = magnitude[:len(magnitude)//2]
                    
                    if np.sum(positive_magnitude) > 0:
                        centroid = np.sum(positive_freqs * positive_magnitude) / np.sum(positive_magnitude)
                        spectral_centroids.append(centroid)
                        
                except Exception as e:
                    print(f"⚠️  Error analyzing {audio_file}: {e}")
            
            # Calculate statistics
            if amplitudes:
                sample_stats['amplitude_statistics'] = {
                    'mean': float(np.mean(amplitudes)),
                    'std': float(np.std(amplitudes)),
                    'min': float(np.min(amplitudes)),
                    'max': float(np.max(amplitudes))
                }
            
            if durations:
                sample_stats['duration_statistics'] = {
                    'mean_seconds': float(np.mean(durations)),
                    'std_seconds': float(np.std(durations)),
                    'min_seconds': float(np.min(durations)),
                    'max_seconds': float(np.max(durations))
                }
            
            if spectral_centroids:
                sample_stats['spectral_statistics'] = {
                    'mean_centroid_hz': float(np.mean(spectral_centroids)),
                    'std_centroid_hz': float(np.std(spectral_centroids))
                }
            
            return sample_stats
            
        except Exception as e:
            print(f"❌ Error analyzing generated audio: {e}")
            return {'error': str(e)}
    
    def _get_domain_specific_report_sections(self) -> List[str]:
        """Get WaveGAN-specific sections for training report"""
        sections = []
        
        try:
            # Add discriminator performance section
            sections.append("## Audio Discriminator Performance")
            
            if os.path.exists(self.training_log_path):
                df = pd.read_csv(self.training_log_path)
                if not df.empty and 'd_real_score' in df.columns:
                    avg_real_score = df['d_real_score'].mean()
                    avg_fake_score = df['d_fake_score'].mean()
                    
                    sections.append(f"- **Average Real Score**: {avg_real_score:.3f}")
                    sections.append(f"- **Average Fake Score**: {avg_fake_score:.3f}")
                    sections.append(f"- **Score Difference**: {avg_real_score - avg_fake_score:.3f}")
                    
                    # Calculate discriminator balance
                    balance_score = abs(avg_real_score - 0.5) + abs(avg_fake_score - 0.5)
                    sections.append(f"- **Balance Score**: {balance_score:.3f} (lower is better)")
            
            sections.append("")
            
            # Add audio quality section
            sections.append("## Audio Quality Metrics")
            
            if os.path.exists(self.checkpoint_metrics_path):
                df = pd.read_csv(self.checkpoint_metrics_path)
                if not df.empty:
                    if 'fad_score' in df.columns:
                        best_fad = df['fad_score'].min()
                        sections.append(f"- **Best FAD Score**: {best_fad:.2f}")
                    
                    if 'snr_score' in df.columns:
                        best_snr = df['snr_score'].max()
                        sections.append(f"- **Best SNR**: {best_snr:.1f} dB")
                    
                    if 'audio_quality_score' in df.columns:
                        best_quality = df['audio_quality_score'].max()
                        sections.append(f"- **Best Quality Score**: {best_quality:.1f}/100")
            
            sections.append("")
            
        except Exception as e:
            print(f"⚠️  Error generating WaveGAN report sections: {e}")
        
        return sections
