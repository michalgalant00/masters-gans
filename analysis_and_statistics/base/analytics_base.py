"""
Base Analytics
==============

Common analytics and visualization functionality for GAN training.
Domain-agnostic plots and analysis tools.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from abc import ABC, abstractmethod

# Set matplotlib backend for headless servers
import matplotlib
matplotlib.use('Agg')


class AnalyticsBase(ABC):
    """Base class for training analytics and visualization"""
    
    def __init__(self, output_dir: str = "output", model_type: str = "base"):
        self.output_dir = output_dir
        self.model_type = model_type
        self.plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # File paths - updated to match optimized MetricsCollector structure
        self.epochs_stats_dir = os.path.join(output_dir, "epochs_statistics")
        self.single_epochs_dir = os.path.join(output_dir, "single_epochs_statistics")
        self.training_log_path = os.path.join(self.epochs_stats_dir, "epoch_statistics.csv")  # For epoch-level data
        self.epoch_summary_path = os.path.join(self.epochs_stats_dir, "epoch_statistics.csv")  # Same as above
        self.checkpoint_metrics_path = os.path.join(self.epochs_stats_dir, "checkpoint_metrics.csv")
        self.experiment_config_path = os.path.join(self.epochs_stats_dir, "experiment_config.json")
        
        # Aggregated iteration data (created on demand)
        self.aggregated_iterations_path = os.path.join(self.plots_dir, "aggregated_iterations.csv")
        
        print(f"📈 {model_type.upper()} Training analyzer initialized - output: {output_dir}")
    
    def _aggregate_iteration_data(self):
        """Aggregate all iteration data from single_epochs_statistics into one CSV for plotting"""
        aggregated_data = []
        
        if not os.path.exists(self.single_epochs_dir):
            print("⚠️  No single epochs statistics directory found")
            return
        
        # Get all epoch directories
        epoch_dirs = [d for d in os.listdir(self.single_epochs_dir) 
                     if d.startswith('epoch_') and os.path.isdir(os.path.join(self.single_epochs_dir, d))]
        
        if not epoch_dirs:
            print("⚠️  No epoch data directories found")
            return
        
        # Sort by epoch number
        epoch_dirs.sort(key=lambda x: int(x.split('_')[1]))
        
        global_iteration = 0
        for epoch_dir in epoch_dirs:
            epoch_csv = os.path.join(self.single_epochs_dir, epoch_dir, f"{epoch_dir}_iterations.csv")
            if os.path.exists(epoch_csv):
                try:
                    df = pd.read_csv(epoch_csv)
                    # Add global iteration counter
                    df['global_iteration'] = range(global_iteration, global_iteration + len(df))
                    global_iteration += len(df)
                    aggregated_data.append(df)
                except Exception as e:
                    print(f"⚠️  Error reading {epoch_csv}: {e}")
        
        if aggregated_data:
            # Combine all data
            combined_df = pd.concat(aggregated_data, ignore_index=True)
            combined_df.to_csv(self.aggregated_iterations_path, index=False)
            print(f"✅ Aggregated {len(combined_df)} iterations from {len(epoch_dirs)} epochs")
            return True
        else:
            print("⚠️  No iteration data found to aggregate")
            return False
    
    def generate_all_plots(self):
        """Generate all training analysis plots"""
        print("🎨 Generating training analysis plots...")
        
        try:
            self.plot_loss_curves()
            self.plot_gradient_norms()
            self.plot_resource_usage()
            self.plot_convergence_analysis()
            self.plot_training_stability()
            
            # Generate domain-specific plots
            self._generate_domain_specific_plots()
            
            print(f"✅ All plots generated successfully in {self.plots_dir}")
            
        except Exception as e:
            print(f"❌ Error generating plots: {e}")
    
    def plot_loss_curves(self):
        """Plot generator and discriminator loss curves"""
        try:
            # First try to aggregate iteration data
            if not os.path.exists(self.aggregated_iterations_path):
                if not self._aggregate_iteration_data():
                    print("⚠️  No iteration data available for loss curves")
                    return
            
            df = pd.read_csv(self.aggregated_iterations_path)
            if df.empty:
                print("⚠️  No training data available for loss curves")
                return
            
            plt.figure(figsize=(15, 5))
            
            # Loss curves - use global_iteration for x-axis
            plt.subplot(1, 3, 1)
            plt.plot(df['global_iteration'], df['generator_loss'], label='Generator', alpha=0.7)
            plt.plot(df['global_iteration'], df['discriminator_loss'], label='Discriminator', alpha=0.7)
            plt.xlabel('Global Iteration')
            plt.ylabel('Loss')
            plt.title('Training Losses')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Loss ratio
            plt.subplot(1, 3, 2)
            loss_ratio = df['generator_loss'] / (df['discriminator_loss'] + 1e-8)
            plt.plot(df['global_iteration'], loss_ratio, color='purple', alpha=0.7)
            plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Balanced')
            plt.xlabel('Global Iteration')
            plt.ylabel('Generator/Discriminator Loss Ratio')
            plt.title('Loss Balance')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Smoothed losses (moving average)
            plt.subplot(1, 3, 3)
            window_size = min(100, len(df) // 10)
            if window_size > 1:
                gen_smooth = df['generator_loss'].rolling(window=window_size).mean()
                disc_smooth = df['discriminator_loss'].rolling(window=window_size).mean()
                plt.plot(df['global_iteration'], gen_smooth, label='Generator (smooth)', linewidth=2)
                plt.plot(df['global_iteration'], disc_smooth, label='Discriminator (smooth)', linewidth=2)
            plt.xlabel('Global Iteration')
            plt.ylabel('Loss')
            plt.title('Smoothed Losses')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'loss_curves.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            print("📊 Loss curves plot saved")
            
        except Exception as e:
            print(f"❌ Error plotting loss curves: {e}")
    
    def plot_gradient_norms(self):
        """Plot gradient norms over training"""
        try:
            # Use aggregated iteration data
            if not os.path.exists(self.aggregated_iterations_path):
                if not self._aggregate_iteration_data():
                    print("⚠️  No iteration data available for gradient plots")
                    return
                    
            df = pd.read_csv(self.aggregated_iterations_path)
            if df.empty or 'generator_grad_norm' not in df.columns:
                print("⚠️  No gradient data available")
                return
            
            plt.figure(figsize=(12, 4))
            
            # Gradient norms
            plt.subplot(1, 2, 1)
            plt.plot(df['global_iteration'], df['generator_grad_norm'], label='Generator', alpha=0.7)
            plt.plot(df['global_iteration'], df['discriminator_grad_norm'], label='Discriminator', alpha=0.7)
            plt.xlabel('Global Iteration')
            plt.ylabel('Gradient Norm')
            plt.title('Gradient Norms')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.yscale('log')
            
            # Gradient variance
            plt.subplot(1, 2, 2)
            if 'gradient_variance_last_10' in df.columns:
                plt.plot(df['global_iteration'], df['gradient_variance_last_10'], color='orange', alpha=0.7)
                plt.xlabel('Global Iteration')
                plt.ylabel('Gradient Variance (Last 10)')
                plt.title('Gradient Stability')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'gradient_norms.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            print("📊 Gradient norms plot saved")
            
        except Exception as e:
            print(f"❌ Error plotting gradient norms: {e}")
    
    def plot_resource_usage(self):
        """Plot system resource usage over training"""
        try:
            # Use aggregated iteration data
            if not os.path.exists(self.aggregated_iterations_path):
                if not self._aggregate_iteration_data():
                    print("⚠️  No iteration data available for resource plots")
                    return
                    
            df = pd.read_csv(self.aggregated_iterations_path)
            if df.empty:
                print("⚠️  No training data available for resource usage")
                return
            
            plt.figure(figsize=(15, 8))
            
            # GPU memory
            if 'gpu_memory_used' in df.columns:
                plt.subplot(2, 3, 1)
                plt.plot(df['global_iteration'], df['gpu_memory_used'], color='green', alpha=0.7)
                plt.xlabel('Global Iteration')
                plt.ylabel('GPU Memory (GB)')
                plt.title('GPU Memory Usage')
                plt.grid(True, alpha=0.3)
            
            # GPU utilization
            if 'gpu_utilization' in df.columns:
                plt.subplot(2, 3, 2)
                plt.plot(df['global_iteration'], df['gpu_utilization'], color='blue', alpha=0.7)
                plt.xlabel('Global Iteration')
                plt.ylabel('GPU Utilization (%)')
                plt.title('GPU Utilization')
                plt.grid(True, alpha=0.3)
            
            # CPU usage
            if 'cpu_usage' in df.columns:
                plt.subplot(2, 3, 3)
                plt.plot(df['global_iteration'], df['cpu_usage'], color='red', alpha=0.7)
                plt.xlabel('Global Iteration')
                plt.ylabel('CPU Usage (%)')
                plt.title('CPU Usage')
                plt.grid(True, alpha=0.3)
            
            # RAM usage
            if 'ram_usage' in df.columns:
                plt.subplot(2, 3, 4)
                plt.plot(df['global_iteration'], df['ram_usage'], color='purple', alpha=0.7)
                plt.xlabel('Global Iteration')
                plt.ylabel('RAM Usage (%)')
                plt.title('RAM Usage')
                plt.grid(True, alpha=0.3)
            
            # Iteration time
            if 'iteration_time_seconds' in df.columns:
                plt.subplot(2, 3, 5)
                plt.plot(df['global_iteration'], df['iteration_time_seconds'], color='orange', alpha=0.7)
                plt.xlabel('Global Iteration')
                plt.ylabel('Time (seconds)')
                plt.title('Iteration Time')
                plt.grid(True, alpha=0.3)
            
            # Resource efficiency (samples per second)
            if 'iteration_time_seconds' in df.columns and 'batch_size' in df.columns:
                plt.subplot(2, 3, 6)
                samples_per_sec = df['batch_size'] / (df['iteration_time_seconds'] + 1e-8)
                plt.plot(df['global_iteration'], samples_per_sec, color='brown', alpha=0.7)
                plt.xlabel('Global Iteration')
                plt.ylabel('Samples/Second')
                plt.title('Training Throughput')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'resource_usage.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            print("📊 Resource usage plot saved")
            
        except Exception as e:
            print(f"❌ Error plotting resource usage: {e}")
    
    def plot_convergence_analysis(self):
        """Plot convergence analysis from epoch summaries"""
        try:
            if not os.path.exists(self.epoch_summary_path):
                print("⚠️  No epoch summary data available")
                return
                
            df = pd.read_csv(self.epoch_summary_path)
            if df.empty:
                print("⚠️  No epoch data available for convergence analysis")
                return
            
            plt.figure(figsize=(12, 8))
            
            # Average losses per epoch
            plt.subplot(2, 2, 1)
            plt.plot(df['epoch'], df['avg_generator_loss'], label='Generator', marker='o')
            plt.plot(df['epoch'], df['avg_discriminator_loss'], label='Discriminator', marker='s')
            plt.xlabel('Epoch')
            plt.ylabel('Average Loss')
            plt.title('Average Losses per Epoch')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Loss variance (min-max range)
            plt.subplot(2, 2, 2)
            gen_range = df['max_generator_loss'] - df['min_generator_loss']
            disc_range = df['max_discriminator_loss'] - df['min_discriminator_loss']
            plt.plot(df['epoch'], gen_range, label='Generator Range', marker='o')
            plt.plot(df['epoch'], disc_range, label='Discriminator Range', marker='s')
            plt.xlabel('Epoch')
            plt.ylabel('Loss Range (Max - Min)')
            plt.title('Loss Variance per Epoch')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Convergence indicator
            if 'convergence_indicator' in df.columns:
                plt.subplot(2, 2, 3)
                plt.plot(df['epoch'], df['convergence_indicator'], color='green', marker='o')
                plt.xlabel('Epoch')
                plt.ylabel('Convergence Indicator')
                plt.title('Convergence Progress')
                plt.grid(True, alpha=0.3)
            
            # Training stability
            if 'training_stability_score' in df.columns:
                plt.subplot(2, 2, 4)
                plt.plot(df['epoch'], df['training_stability_score'], color='purple', marker='o')
                plt.xlabel('Epoch')
                plt.ylabel('Stability Score')
                plt.title('Training Stability')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'convergence_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            print("📊 Convergence analysis plot saved")
            
        except Exception as e:
            print(f"❌ Error plotting convergence analysis: {e}")
    
    def plot_training_stability(self):
        """Plot training stability metrics"""
        try:
            # Use aggregated iteration data
            if not os.path.exists(self.aggregated_iterations_path):
                if not self._aggregate_iteration_data():
                    print("⚠️  No iteration data available for stability plots")
                    return
                    
            df = pd.read_csv(self.aggregated_iterations_path)
            if df.empty:
                print("⚠️  No training data available for stability analysis")
                return
            
            plt.figure(figsize=(12, 6))
            
            # Loss variance over time
            plt.subplot(1, 2, 1)
            if 'loss_variance_last_10' in df.columns:
                plt.plot(df['global_iteration'], df['loss_variance_last_10'], alpha=0.7)
                plt.xlabel('Global Iteration')
                plt.ylabel('Loss Variance (Last 10)')
                plt.title('Loss Stability')
                plt.grid(True, alpha=0.3)
            
            # Learning rate changes
            plt.subplot(1, 2, 2)
            if 'learning_rate_g' in df.columns and 'learning_rate_d' in df.columns:
                plt.plot(df['global_iteration'], df['learning_rate_g'], label='Generator LR')
                plt.plot(df['global_iteration'], df['learning_rate_d'], label='Discriminator LR')
                plt.xlabel('Global Iteration')
                plt.ylabel('Learning Rate')
                plt.title('Learning Rate Schedule')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.yscale('log')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'training_stability.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            print("📊 Training stability plot saved")
            
        except Exception as e:
            print(f"❌ Error plotting training stability: {e}")
    
    @abstractmethod
    def _generate_domain_specific_plots(self):
        """Generate domain-specific plots - implemented by subclasses"""
        pass
    
    def generate_training_report(self) -> str:
        """Generate comprehensive training report"""
        try:
            report_lines = []
            report_lines.append(f"# {self.model_type.upper()} Training Report")
            report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append("")
            
            # Load experiment config
            if os.path.exists(self.experiment_config_path):
                with open(self.experiment_config_path, 'r') as f:
                    config = json.load(f)
                    
                report_lines.append("## Experiment Configuration")
                for key, value in config.items():
                    if isinstance(value, (int, float, str, bool)):
                        report_lines.append(f"- **{key}**: {value}")
                report_lines.append("")
            
            # Training summary
            if os.path.exists(self.training_log_path):
                df = pd.read_csv(self.training_log_path)
                if not df.empty:
                    report_lines.append("## Training Summary")
                    report_lines.append(f"- **Total Epochs**: {len(df)}")
                    report_lines.append(f"- **Final Epoch**: {df['epoch'].max()}")
                    
                    # Use correct column names for epoch statistics
                    if 'avg_generator_loss' in df.columns:
                        report_lines.append(f"- **Best Generator Loss**: {df['avg_generator_loss'].min():.6f}")
                    if 'avg_discriminator_loss' in df.columns:
                        report_lines.append(f"- **Best Discriminator Loss**: {df['avg_discriminator_loss'].min():.6f}")
                    if 'epoch_duration_minutes' in df.columns:
                        report_lines.append(f"- **Average Epoch Duration**: {df['epoch_duration_minutes'].mean():.3f} minutes")
                    
                    if 'gpu_memory_used' in df.columns:
                        report_lines.append(f"- **Peak GPU Memory**: {df['gpu_memory_used'].max():.2f}GB")
                    
                    report_lines.append("")
            
            # Add domain-specific report sections
            domain_sections = self._get_domain_specific_report_sections()
            report_lines.extend(domain_sections)
            
            # Save report
            report_path = os.path.join(self.output_dir, f"{self.model_type}_training_report.md")
            with open(report_path, 'w') as f:
                f.write('\n'.join(report_lines))
            
            print(f"📄 Training report saved: {report_path}")
            return report_path
            
        except Exception as e:
            print(f"❌ Error generating training report: {e}")
            return ""
    
    @abstractmethod
    def _get_domain_specific_report_sections(self) -> List[str]:
        """Get domain-specific sections for training report"""
        pass
    
    def cleanup_old_plots(self, keep_n: int = 10):
        """Clean up old plot files"""
        try:
            if not os.path.exists(self.plots_dir):
                return
            
            # Get all PNG files with timestamps
            png_files = []
            for filename in os.listdir(self.plots_dir):
                if filename.endswith('.png'):
                    file_path = os.path.join(self.plots_dir, filename)
                    mtime = os.path.getmtime(file_path)
                    png_files.append((file_path, mtime))
            
            # Sort by modification time (newest first)
            png_files.sort(key=lambda x: x[1], reverse=True)
            
            # Remove old files beyond keep_n
            for file_path, _ in png_files[keep_n:]:
                try:
                    os.remove(file_path)
                    print(f"🗑️  Removed old plot: {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"⚠️  Error removing {file_path}: {e}")
                    
        except Exception as e:
            print(f"❌ Error cleaning up plots: {e}")
