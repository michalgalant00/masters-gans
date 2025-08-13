"""
Image Analytics
===============

DCGAN-specific analytics and visualization tools.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Dict, List, Optional, Any
import pandas as pd

# Import base class
from ..base.analytics_base import AnalyticsBase


class ImageAnalytics(AnalyticsBase):
    """Analytics for image-based GANs (DCGAN)"""
    
    def __init__(self, output_dir: str = "output"):
        super().__init__(output_dir, model_type="image")
    
    def _generate_domain_specific_plots(self):
        """Generate DCGAN-specific plots"""
        try:
            self.plot_discriminator_scores()
            self.plot_image_quality_metrics()
            self.plot_sample_evolution()
            
        except Exception as e:
            print(f"❌ Error generating image-specific plots: {e}")
    
    def plot_discriminator_scores(self):
        """Plot discriminator performance metrics"""
        try:
            df = pd.read_csv(self.training_log_path)
            if df.empty or 'd_real_score' not in df.columns:
                print("⚠️  No discriminator score data available")
                return
            
            plt.figure(figsize=(15, 10))
            
            # Real vs Fake scores over time
            plt.subplot(2, 3, 1)
            plt.plot(df['iteration'], df['d_real_score'], label='Real Score', alpha=0.7)
            plt.plot(df['iteration'], df['d_fake_score'], label='Fake Score', alpha=0.7)
            plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
            plt.xlabel('Iteration')
            plt.ylabel('Discriminator Score')
            plt.title('Discriminator Scores Over Time')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Score distributions
            plt.subplot(2, 3, 2)
            plt.hist(df['d_real_score'], bins=50, alpha=0.7, label='Real', density=True)
            plt.hist(df['d_fake_score'], bins=50, alpha=0.7, label='Fake', density=True)
            plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
            plt.xlabel('Score')
            plt.ylabel('Density')
            plt.title('Score Distributions')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Score difference (real - fake)
            plt.subplot(2, 3, 3)
            score_diff = df['d_real_score'] - df['d_fake_score']
            plt.plot(df['iteration'], score_diff, color='purple', alpha=0.7)
            plt.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Balanced')
            plt.xlabel('Iteration')
            plt.ylabel('Real Score - Fake Score')
            plt.title('Discriminator Balance')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Moving averages
            plt.subplot(2, 3, 4)
            window_size = min(100, len(df) // 10)
            if window_size > 1:
                real_smooth = df['d_real_score'].rolling(window=window_size).mean()
                fake_smooth = df['d_fake_score'].rolling(window=window_size).mean()
                plt.plot(df['iteration'], real_smooth, label='Real (smooth)', linewidth=2)
                plt.plot(df['iteration'], fake_smooth, label='Fake (smooth)', linewidth=2)
                plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
            plt.xlabel('Iteration')
            plt.ylabel('Discriminator Score')
            plt.title('Smoothed Discriminator Scores')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Score variance
            plt.subplot(2, 3, 5)
            if window_size > 1:
                real_var = df['d_real_score'].rolling(window=window_size).std()
                fake_var = df['d_fake_score'].rolling(window=window_size).std()
                plt.plot(df['iteration'], real_var, label='Real Variance', alpha=0.7)
                plt.plot(df['iteration'], fake_var, label='Fake Variance', alpha=0.7)
            plt.xlabel('Iteration')
            plt.ylabel('Score Variance')
            plt.title('Discriminator Score Stability')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Epoch-level discriminator accuracy
            plt.subplot(2, 3, 6)
            if os.path.exists(self.epoch_summary_path):
                epoch_df = pd.read_csv(self.epoch_summary_path)
                if not epoch_df.empty and 'd_real_accuracy' in epoch_df.columns:
                    plt.plot(epoch_df['epoch'], epoch_df['d_real_accuracy'], 
                            label='Real Accuracy', marker='o')
                    plt.plot(epoch_df['epoch'], epoch_df['d_fake_accuracy'], 
                            label='Fake Accuracy', marker='s')
                    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
                    plt.xlabel('Epoch')
                    plt.ylabel('Accuracy')
                    plt.title('Discriminator Accuracy per Epoch')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'discriminator_analysis.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print("📊 Discriminator analysis plot saved")
            
        except Exception as e:
            print(f"❌ Error plotting discriminator scores: {e}")
    
    def plot_image_quality_metrics(self):
        """Plot image quality metrics over training"""
        try:
            if not os.path.exists(self.checkpoint_metrics_path):
                print("⚠️  No checkpoint metrics data available")
                return
                
            df = pd.read_csv(self.checkpoint_metrics_path)
            if df.empty:
                print("⚠️  No checkpoint data for image quality metrics")
                return
            
            plt.figure(figsize=(15, 10))
            
            # FID score over time
            if 'fid_score' in df.columns:
                plt.subplot(2, 3, 1)
                plt.plot(df['epoch'], df['fid_score'], marker='o', alpha=0.7)
                plt.xlabel('Epoch')
                plt.ylabel('FID Score')
                plt.title('FID Score Over Training (Lower is Better)')
                plt.grid(True, alpha=0.3)
            
            # Inception Score over time
            if 'inception_score' in df.columns:
                plt.subplot(2, 3, 2)
                plt.plot(df['epoch'], df['inception_score'], marker='s', alpha=0.7, color='green')
                plt.xlabel('Epoch')
                plt.ylabel('Inception Score')
                plt.title('Inception Score Over Training (Higher is Better)')
                plt.grid(True, alpha=0.3)
            
            # Image quality score over time
            if 'image_quality_score' in df.columns:
                plt.subplot(2, 3, 3)
                plt.plot(df['epoch'], df['image_quality_score'], marker='^', alpha=0.7, color='purple')
                plt.xlabel('Epoch')
                plt.ylabel('Quality Score')
                plt.title('Composite Image Quality Score')
                plt.grid(True, alpha=0.3)
            
            # FID vs Inception Score correlation
            if 'fid_score' in df.columns and 'inception_score' in df.columns:
                plt.subplot(2, 3, 4)
                plt.scatter(df['fid_score'], df['inception_score'], alpha=0.7)
                plt.xlabel('FID Score')
                plt.ylabel('Inception Score')
                plt.title('FID vs Inception Score')
                plt.grid(True, alpha=0.3)
                
                # Add correlation coefficient
                correlation = df['fid_score'].corr(df['inception_score'])
                plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                        transform=plt.gca().transAxes, bbox=dict(boxstyle='round', facecolor='wheat'))
            
            # Quality metrics distribution
            plt.subplot(2, 3, 5)
            metrics_to_plot = []
            if 'fid_score' in df.columns:
                metrics_to_plot.append(('FID', df['fid_score']))
            if 'inception_score' in df.columns:
                metrics_to_plot.append(('IS', df['inception_score']))
            if 'image_quality_score' in df.columns:
                metrics_to_plot.append(('Quality', df['image_quality_score']))
            
            if metrics_to_plot:
                positions = range(len(metrics_to_plot))
                values = [metric[1] for metric in metrics_to_plot]
                labels = [metric[0] for metric in metrics_to_plot]
                
                box_plot = plt.boxplot(values, positions=positions)
                plt.xticks(positions, labels)
                plt.ylabel('Score')
                plt.title('Quality Metrics Distribution')
                plt.grid(True, alpha=0.3)
            
            # Quality improvement trend
            plt.subplot(2, 3, 6)
            if 'image_quality_score' in df.columns and len(df) > 5:
                # Calculate trend
                x = np.arange(len(df))
                z = np.polyfit(x, df['image_quality_score'], 1)
                p = np.poly1d(z)
                
                plt.plot(df['epoch'], df['image_quality_score'], 'o', alpha=0.7, label='Quality Score')
                plt.plot(df['epoch'], p(x), "r--", alpha=0.8, label='Trend')
                plt.xlabel('Epoch')
                plt.ylabel('Quality Score')
                plt.title('Quality Improvement Trend')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Add trend info
                slope = z[0]
                trend_text = f'Trend: {"Improving" if slope > 0 else "Declining"}\nSlope: {slope:.3f}'
                plt.text(0.05, 0.95, trend_text, transform=plt.gca().transAxes,
                        bbox=dict(boxstyle='round', facecolor='wheat'))
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'image_quality_metrics.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print("📊 Image quality metrics plot saved")
            
        except Exception as e:
            print(f"❌ Error plotting image quality metrics: {e}")
    
    def plot_sample_evolution(self):
        """Plot evolution of generated samples over training"""
        try:
            # Look for sample directories
            samples_base_dir = os.path.join(self.output_dir, "samples")
            if not os.path.exists(samples_base_dir):
                print("⚠️  No samples directory found")
                return
            
            # Find checkpoint sample directories
            checkpoint_dirs = []
            for subdir in os.listdir(samples_base_dir):
                subdir_path = os.path.join(samples_base_dir, subdir)
                if os.path.isdir(subdir_path):
                    checkpoint_dirs.append(subdir_path)
            
            if not checkpoint_dirs:
                print("⚠️  No checkpoint sample directories found")
                return
            
            # Create evolution plot
            fig, axes = plt.subplots(2, min(5, len(checkpoint_dirs)), 
                                   figsize=(15, 6))
            
            if len(checkpoint_dirs) == 1:
                axes = np.array([[axes], [axes]])
            elif len(checkpoint_dirs) <= 5:
                axes = axes.reshape(2, -1)
            
            for i, checkpoint_dir in enumerate(sorted(checkpoint_dirs)[:5]):
                # Find grid images in this checkpoint directory
                grid_files = [f for f in os.listdir(checkpoint_dir) 
                             if f.startswith('grid_') and f.endswith('.png')]
                
                if grid_files:
                    # Use the first grid file found
                    grid_path = os.path.join(checkpoint_dir, grid_files[0])
                    
                    try:
                        img = Image.open(grid_path)
                        img_array = np.array(img)
                        
                        # Plot in first row
                        axes[0, i].imshow(img_array)
                        axes[0, i].set_title(f'{os.path.basename(checkpoint_dir)}')
                        axes[0, i].axis('off')
                        
                        # Plot histogram of pixel intensities in second row
                        if len(img_array.shape) == 3:
                            pixel_intensities = img_array.flatten()
                        else:
                            pixel_intensities = img_array.flatten()
                        
                        axes[1, i].hist(pixel_intensities, bins=50, alpha=0.7)
                        axes[1, i].set_title('Pixel Distribution')
                        axes[1, i].set_xlabel('Intensity')
                        axes[1, i].set_ylabel('Frequency')
                        
                    except Exception as e:
                        print(f"⚠️  Error loading image {grid_path}: {e}")
                        axes[0, i].text(0.5, 0.5, 'Image\nLoad Error', 
                                       ha='center', va='center')
                        axes[0, i].set_title(f'{os.path.basename(checkpoint_dir)}')
                        axes[1, i].text(0.5, 0.5, 'No Data', ha='center', va='center')
                else:
                    axes[0, i].text(0.5, 0.5, 'No Grid\nFound', ha='center', va='center')
                    axes[0, i].set_title(f'{os.path.basename(checkpoint_dir)}')
                    axes[1, i].text(0.5, 0.5, 'No Data', ha='center', va='center')
            
            # Hide unused subplots
            for i in range(len(checkpoint_dirs), min(5, axes.shape[1])):
                axes[0, i].set_visible(False)
                axes[1, i].set_visible(False)
            
            plt.suptitle('Sample Evolution Over Training')
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'sample_evolution.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print("📊 Sample evolution plot saved")
            
        except Exception as e:
            print(f"❌ Error plotting sample evolution: {e}")
    
    def analyze_generated_samples(self, samples_dir: str) -> Dict[str, Any]:
        """Analyze characteristics of generated samples
        
        Args:
            samples_dir: Directory containing generated samples
            
        Returns:
            Dictionary with analysis results
        """
        try:
            if not os.path.exists(samples_dir):
                return {}
            
            # Find all image files
            image_files = []
            for ext in ['.png', '.jpg', '.jpeg']:
                pattern = f"*{ext}"
                image_files.extend([f for f in os.listdir(samples_dir) if f.endswith(ext)])
            
            if not image_files:
                return {'error': 'No image files found'}
            
            # Analyze sample characteristics
            sample_stats = {
                'total_samples': len(image_files),
                'pixel_statistics': {},
                'size_statistics': {},
                'color_statistics': {}
            }
            
            pixel_values = []
            image_sizes = []
            color_channels = []
            
            for img_file in image_files[:50]:  # Analyze up to 50 images
                img_path = os.path.join(samples_dir, img_file)
                try:
                    img = Image.open(img_path)
                    img_array = np.array(img)
                    
                    # Collect statistics
                    pixel_values.extend(img_array.flatten())
                    image_sizes.append(img_array.shape[:2])
                    
                    if len(img_array.shape) == 3:
                        color_channels.append(img_array.shape[2])
                    else:
                        color_channels.append(1)
                        
                except Exception as e:
                    print(f"⚠️  Error analyzing {img_file}: {e}")
            
            # Calculate statistics
            if pixel_values:
                sample_stats['pixel_statistics'] = {
                    'mean': float(np.mean(pixel_values)),
                    'std': float(np.std(pixel_values)),
                    'min': float(np.min(pixel_values)),
                    'max': float(np.max(pixel_values))
                }
            
            if image_sizes:
                heights = [size[0] for size in image_sizes]
                widths = [size[1] for size in image_sizes]
                sample_stats['size_statistics'] = {
                    'height_mean': float(np.mean(heights)),
                    'width_mean': float(np.mean(widths)),
                    'height_std': float(np.std(heights)),
                    'width_std': float(np.std(widths))
                }
            
            if color_channels:
                sample_stats['color_statistics'] = {
                    'channels_mode': int(max(set(color_channels), key=color_channels.count)),
                    'channels_unique': list(set(color_channels))
                }
            
            return sample_stats
            
        except Exception as e:
            print(f"❌ Error analyzing generated samples: {e}")
            return {'error': str(e)}
    
    def _get_domain_specific_report_sections(self) -> List[str]:
        """Get DCGAN-specific sections for training report"""
        sections = []
        
        try:
            # Add discriminator performance section
            sections.append("## Discriminator Performance")
            
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
            
            # Add image quality section
            sections.append("## Image Quality Metrics")
            
            if os.path.exists(self.checkpoint_metrics_path):
                df = pd.read_csv(self.checkpoint_metrics_path)
                if not df.empty:
                    if 'fid_score' in df.columns:
                        best_fid = df['fid_score'].min()
                        sections.append(f"- **Best FID Score**: {best_fid:.2f}")
                    
                    if 'inception_score' in df.columns:
                        best_is = df['inception_score'].max()
                        sections.append(f"- **Best Inception Score**: {best_is:.2f}")
                    
                    if 'image_quality_score' in df.columns:
                        best_quality = df['image_quality_score'].max()
                        sections.append(f"- **Best Quality Score**: {best_quality:.1f}/100")
            
            sections.append("")
            
        except Exception as e:
            print(f"⚠️  Error generating DCGAN report sections: {e}")
        
        return sections
