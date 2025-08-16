"""
Post-Training Analyzer
=====================

Auto-generated plots and analysis according to metrics-checkpoints-how-to.txt

POST-TRAINING ANALYSIS (AUTOMATYCZNE):
- Podstawowe wykresy (auto-generated)
- Summary report (JSON)
- Training efficiency analysis
- Convergence detection
- Resource usage analysis
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple


class PostTrainingAnalyzer:
    """
    Auto-generate plots and analysis from optimized metrics structure
    
    PODSTAWOWE WYKRESY (AUTO-GENERATED):
    - epoch_loss_curves.png (G vs D losses over epochs)
    - gradient_norms.png (training stability per epoch)
    - resource_usage.png (GPU/memory over epochs)
    - convergence_analysis.png (loss variance trends per epoch)
    - detailed_iteration_analysis.png (analiza z single_epochs_statistics)
    """
    
    def __init__(self, output_analysis_dir: str = "output_analysis"):
        self.output_analysis_dir = output_analysis_dir
        
        # Look for data in the metrics subdirectory (where training actually saves data)
        metrics_dir = os.path.join(output_analysis_dir, "metrics")
        if os.path.exists(os.path.join(metrics_dir, "epochs_statistics")):
            self.epochs_stats_dir = os.path.join(metrics_dir, "epochs_statistics")
            self.single_epochs_dir = os.path.join(metrics_dir, "single_epochs_statistics")
        elif os.path.exists(os.path.join("output", "epochs_statistics")):
            # Legacy fallback for old structure
            self.epochs_stats_dir = os.path.join("output", "epochs_statistics")
            self.single_epochs_dir = os.path.join("output", "single_epochs_statistics")
        else:
            # Direct fallback - create expected structure
            self.epochs_stats_dir = os.path.join(output_analysis_dir, "epochs_statistics")
            self.single_epochs_dir = os.path.join(output_analysis_dir, "single_epochs_statistics")
            
        # Always save plots to analysis directory
        self.plots_dir = os.path.join(output_analysis_dir, "plots")
        
        # Create plots directory
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Load data
        self.epoch_stats_df = None
        self.checkpoint_metrics_df = None
        self.experiment_config = None
        
        self._load_data()
        
        print(f"📈 Post-training analyzer initialized - output: {output_analysis_dir}")
    
    def _load_data(self):
        """Load data from CSV files"""
        try:
            # Load epoch statistics
            epoch_stats_path = os.path.join(self.epochs_stats_dir, "epoch_statistics.csv")
            if os.path.exists(epoch_stats_path):
                self.epoch_stats_df = pd.read_csv(epoch_stats_path)
                print(f"📊 Loaded epoch statistics: {len(self.epoch_stats_df)} epochs")
            
            # Load checkpoint metrics
            checkpoint_metrics_path = os.path.join(self.epochs_stats_dir, "checkpoint_metrics.csv")
            if os.path.exists(checkpoint_metrics_path):
                self.checkpoint_metrics_df = pd.read_csv(checkpoint_metrics_path)
                print(f"💾 Loaded checkpoint metrics: {len(self.checkpoint_metrics_df)} checkpoints")
            
            # Load experiment config
            config_path = os.path.join(self.epochs_stats_dir, "experiment_config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.experiment_config = json.load(f)
                print(f"⚙️  Loaded experiment configuration")
                
        except Exception as e:
            print(f"⚠️  Error loading data: {e}")
    
    def generate_all_plots(self):
        """Generate all basic plots according to metrics-checkpoints-how-to.txt"""
        if self.epoch_stats_df is None or len(self.epoch_stats_df) == 0:
            print("⚠️  No epoch statistics data available for plotting")
            return
        
        print("🎨 Generating training analysis plots...")
        
        # Basic plots
        self.plot_epoch_loss_curves()
        self.plot_gradient_norms()
        self.plot_resource_usage() 
        self.plot_convergence_analysis()
        
        # Detailed iteration analysis if available
        self.plot_detailed_iteration_analysis()
        
        print(f"✅ All plots generated in: {self.plots_dir}")
    
    def plot_epoch_loss_curves(self):
        """epoch_loss_curves.png (G vs D losses over epochs)"""
        try:
            plt.figure(figsize=(12, 8))
            
            # Main loss curves
            plt.subplot(2, 2, 1)
            plt.plot(self.epoch_stats_df['epoch'], self.epoch_stats_df['avg_generator_loss'], 
                    'b-', label='Generator Loss', linewidth=2)
            plt.plot(self.epoch_stats_df['epoch'], self.epoch_stats_df['avg_discriminator_loss'], 
                    'r-', label='Discriminator Loss', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Average Loss Curves')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Loss variance (convergence indicator)
            plt.subplot(2, 2, 2)
            plt.plot(self.epoch_stats_df['epoch'], self.epoch_stats_df['convergence_indicator'], 
                    'g-', label='Convergence Indicator', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Convergence Score')
            plt.title('Training Convergence')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Min/Max loss bounds
            plt.subplot(2, 2, 3)
            epochs = self.epoch_stats_df['epoch']
            plt.fill_between(epochs, 
                           self.epoch_stats_df['min_generator_loss'],
                           self.epoch_stats_df['max_generator_loss'],
                           alpha=0.3, label='Generator Loss Range')
            plt.fill_between(epochs,
                           self.epoch_stats_df['min_discriminator_loss'], 
                           self.epoch_stats_df['max_discriminator_loss'],
                           alpha=0.3, label='Discriminator Loss Range')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Loss Variance per Epoch')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Training stability
            plt.subplot(2, 2, 4)
            plt.plot(self.epoch_stats_df['epoch'], self.epoch_stats_df['training_stability_score'],
                    'purple', label='Training Stability', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Stability Score')
            plt.title('Training Stability')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = os.path.join(self.plots_dir, "epoch_loss_curves.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Loss curves plot saved: epoch_loss_curves.png")
            
        except Exception as e:
            print(f"❌ Error generating loss curves plot: {e}")
    
    def plot_gradient_norms(self):
        """gradient_norms.png (training stability per epoch)"""
        try:
            plt.figure(figsize=(12, 6))
            
            # Average gradient norms
            plt.subplot(1, 2, 1)
            plt.plot(self.epoch_stats_df['epoch'], self.epoch_stats_df['avg_grad_norm_g'],
                    'b-', label='Generator Gradients', linewidth=2)
            plt.plot(self.epoch_stats_df['epoch'], self.epoch_stats_df['avg_grad_norm_d'],
                    'r-', label='Discriminator Gradients', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Average Gradient Norm')
            plt.title('Gradient Norms (Average)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.yscale('log')
            
            # Maximum gradient norms (for explosion detection)
            plt.subplot(1, 2, 2)
            plt.plot(self.epoch_stats_df['epoch'], self.epoch_stats_df['max_grad_norm_g'],
                    'b--', label='Max Generator Gradients', linewidth=2)
            plt.plot(self.epoch_stats_df['epoch'], self.epoch_stats_df['max_grad_norm_d'],
                    'r--', label='Max Discriminator Gradients', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Maximum Gradient Norm')
            plt.title('Gradient Norms (Maximum)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.yscale('log')
            
            plt.tight_layout()
            plot_path = os.path.join(self.plots_dir, "gradient_norms.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Gradient norms plot saved: gradient_norms.png")
            
        except Exception as e:
            print(f"❌ Error generating gradient norms plot: {e}")
    
    def plot_resource_usage(self):
        """resource_usage.png (GPU/memory over epochs)"""
        try:
            plt.figure(figsize=(12, 8))
            
            # GPU Memory Usage
            plt.subplot(2, 2, 1)
            plt.plot(self.epoch_stats_df['epoch'], self.epoch_stats_df['gpu_memory_used'],
                    'g-', label='GPU Memory Used', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('GPU Memory (MB)')
            plt.title('GPU Memory Usage')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # GPU Utilization
            if 'gpu_utilization' in self.epoch_stats_df.columns:
                plt.subplot(2, 2, 2)
                plt.plot(self.epoch_stats_df['epoch'], self.epoch_stats_df['gpu_utilization'],
                        'orange', label='GPU Utilization (%)', linewidth=2)
                plt.xlabel('Epoch')
                plt.ylabel('GPU Utilization (%)')
                plt.title('GPU Utilization')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            # CPU Usage
            if 'cpu_usage' in self.epoch_stats_df.columns:
                plt.subplot(2, 2, 3)
                plt.plot(self.epoch_stats_df['epoch'], self.epoch_stats_df['cpu_usage'],
                        'purple', label='CPU Usage (%)', linewidth=2)
                plt.xlabel('Epoch')
                plt.ylabel('CPU Usage (%)')
                plt.title('CPU Usage')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            # RAM Usage
            if 'ram_usage' in self.epoch_stats_df.columns:
                plt.subplot(2, 2, 4)
                plt.plot(self.epoch_stats_df['epoch'], self.epoch_stats_df['ram_usage'],
                        'brown', label='RAM Usage (%)', linewidth=2)
                plt.xlabel('Epoch')
                plt.ylabel('RAM Usage (%)')
                plt.title('RAM Usage')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = os.path.join(self.plots_dir, "resource_usage.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Resource usage plot saved: resource_usage.png")
            
        except Exception as e:
            print(f"❌ Error generating resource usage plot: {e}")
    
    def plot_convergence_analysis(self):
        """convergence_analysis.png (loss variance trends per epoch)"""
        try:
            plt.figure(figsize=(12, 8))
            
            # Loss variance over time
            plt.subplot(2, 2, 1)
            loss_variance = []
            window_size = 10
            
            for i in range(len(self.epoch_stats_df)):
                start_idx = max(0, i - window_size + 1)
                end_idx = i + 1
                recent_losses = self.epoch_stats_df['avg_generator_loss'].iloc[start_idx:end_idx]
                loss_variance.append(np.var(recent_losses))
            
            plt.plot(self.epoch_stats_df['epoch'], loss_variance, 
                    'red', label=f'Loss Variance (window={window_size})', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Loss Variance')
            plt.title('Loss Variance Trends')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.yscale('log')
            
            # Epoch duration trends
            plt.subplot(2, 2, 2)
            plt.plot(self.epoch_stats_df['epoch'], self.epoch_stats_df['epoch_duration_minutes'],
                    'blue', label='Epoch Duration', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Duration (minutes)')
            plt.title('Training Speed')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Convergence detection
            plt.subplot(2, 2, 3)
            plt.plot(self.epoch_stats_df['epoch'], self.epoch_stats_df['convergence_indicator'],
                    'green', label='Convergence Indicator', linewidth=2)
            
            # Mark potential convergence points
            convergence_threshold = 0.8
            converged_epochs = self.epoch_stats_df[
                self.epoch_stats_df['convergence_indicator'] > convergence_threshold
            ]
            
            if len(converged_epochs) > 0:
                plt.scatter(converged_epochs['epoch'], converged_epochs['convergence_indicator'],
                          color='red', s=50, label=f'Potential Convergence (>{convergence_threshold})')
            
            plt.axhline(y=convergence_threshold, color='red', linestyle='--', alpha=0.5)
            plt.xlabel('Epoch')
            plt.ylabel('Convergence Score')
            plt.title('Convergence Detection')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Training efficiency (iterations per epoch)
            plt.subplot(2, 2, 4)
            if 'total_iterations_in_epoch' in self.epoch_stats_df.columns:
                plt.plot(self.epoch_stats_df['epoch'], self.epoch_stats_df['total_iterations_in_epoch'],
                        'orange', label='Iterations per Epoch', linewidth=2)
                plt.xlabel('Epoch')
                plt.ylabel('Iterations')
                plt.title('Training Efficiency')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = os.path.join(self.plots_dir, "convergence_analysis.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Convergence analysis plot saved: convergence_analysis.png")
            
        except Exception as e:
            print(f"❌ Error generating convergence analysis plot: {e}")
    
    def plot_detailed_iteration_analysis(self):
        """detailed_iteration_analysis.png (analiza z single_epochs_statistics)"""
        try:
            # Find available epoch directories
            if not os.path.exists(self.single_epochs_dir):
                print("⚠️  No single epoch statistics available for detailed analysis")
                return
            
            epoch_dirs = [d for d in os.listdir(self.single_epochs_dir) 
                         if d.startswith('epoch_') and os.path.isdir(os.path.join(self.single_epochs_dir, d))]
            
            if not epoch_dirs:
                print("⚠️  No epoch directories found for detailed analysis")
                return
            
            # Analyze a few representative epochs
            sample_epochs = sorted(epoch_dirs)[-3:]  # Last 3 epochs
            
            plt.figure(figsize=(15, 10))
            
            for i, epoch_dir in enumerate(sample_epochs):
                epoch_csv = os.path.join(self.single_epochs_dir, epoch_dir, f"{epoch_dir}_iterations.csv")
                
                if os.path.exists(epoch_csv):
                    epoch_df = pd.read_csv(epoch_csv)
                    
                    # Plot iteration losses
                    plt.subplot(2, 3, i + 1)
                    plt.plot(epoch_df['iteration'], epoch_df['generator_loss'], 
                            'b-', label='Generator', alpha=0.7)
                    plt.plot(epoch_df['iteration'], epoch_df['discriminator_loss'], 
                            'r-', label='Discriminator', alpha=0.7)
                    plt.xlabel('Iteration')
                    plt.ylabel('Loss')
                    plt.title(f'{epoch_dir.replace("_", " ").title()} - Losses')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    # Plot gradient norms
                    plt.subplot(2, 3, i + 4)
                    plt.plot(epoch_df['iteration'], epoch_df['generator_grad_norm'],
                            'b--', label='Generator Grad', alpha=0.7)
                    plt.plot(epoch_df['iteration'], epoch_df['discriminator_grad_norm'],
                            'r--', label='Discriminator Grad', alpha=0.7)
                    plt.xlabel('Iteration')
                    plt.ylabel('Gradient Norm')
                    plt.title(f'{epoch_dir.replace("_", " ").title()} - Gradients')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.yscale('log')
            
            plt.tight_layout()
            plot_path = os.path.join(self.plots_dir, "detailed_iteration_analysis.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Detailed iteration analysis plot saved: detailed_iteration_analysis.png")
            
        except Exception as e:
            print(f"❌ Error generating detailed iteration analysis: {e}")
    
    def generate_training_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive training summary report
        
        SUMMARY REPORT (JSON):
        - training_summary.json
        """
        
        if self.epoch_stats_df is None or len(self.epoch_stats_df) == 0:
            return {'error': 'No training data available'}
        
        try:
            # Basic training statistics
            total_epochs = len(self.epoch_stats_df)
            start_time = self.experiment_config.get('experiment_start_time', '') if self.experiment_config else ''
            
            # Loss analysis
            best_gen_loss = self.epoch_stats_df['avg_generator_loss'].min()
            best_gen_epoch = self.epoch_stats_df.loc[
                self.epoch_stats_df['avg_generator_loss'].idxmin(), 'epoch'
            ]
            final_gen_loss = self.epoch_stats_df['avg_generator_loss'].iloc[-1]
            final_disc_loss = self.epoch_stats_df['avg_discriminator_loss'].iloc[-1]
            
            # Training efficiency
            total_time_hours = self.epoch_stats_df['epoch_duration_minutes'].sum() / 60.0
            avg_epoch_duration = self.epoch_stats_df['epoch_duration_minutes'].mean()
            
            # Convergence analysis
            convergence_epochs = self.epoch_stats_df[
                self.epoch_stats_df['convergence_indicator'] > 0.8
            ]
            convergence_epoch = convergence_epochs['epoch'].min() if len(convergence_epochs) > 0 else None
            
            # Resource efficiency
            avg_gpu_memory = self.epoch_stats_df['gpu_memory_used'].mean()
            max_gpu_memory = self.epoch_stats_df['gpu_memory_used'].max()
            
            # Training stability
            avg_stability = self.epoch_stats_df['training_stability_score'].mean()
            
            # Model performance score (composite metric)
            # Lower loss + higher stability + convergence
            performance_score = self._calculate_model_performance_score()
            
            # Checkpoint recommendations
            checkpoint_recommendations = self._get_checkpoint_recommendations()
            
            report = {
                'training_summary': {
                    'total_time_hours': round(total_time_hours, 2),
                    'total_epochs': total_epochs,
                    'total_iterations': self.epoch_stats_df['total_iterations_in_epoch'].sum() if 'total_iterations_in_epoch' in self.epoch_stats_df.columns else 0,
                    'start_time': start_time,
                    'completion_time': datetime.now().isoformat()
                },
                'loss_analysis': {
                    'best_generator_loss': round(best_gen_loss, 6),
                    'best_generator_epoch': int(best_gen_epoch),
                    'final_generator_loss': round(final_gen_loss, 6),
                    'final_discriminator_loss': round(final_disc_loss, 6),
                    'loss_improvement': round((self.epoch_stats_df['avg_generator_loss'].iloc[0] - final_gen_loss), 6)
                },
                'convergence_analysis': {
                    'convergence_detected': convergence_epoch is not None,
                    'convergence_epoch': int(convergence_epoch) if convergence_epoch is not None else None,
                    'epochs_to_convergence': int(convergence_epoch) if convergence_epoch is not None else None
                },
                'training_efficiency': {
                    'average_epoch_duration_minutes': round(avg_epoch_duration, 2),
                    'training_speed_epochs_per_hour': round(60.0 / avg_epoch_duration, 2),
                    'resource_efficiency_score': self._calculate_resource_efficiency_score()
                },
                'resource_usage': {
                    'average_gpu_memory_mb': round(avg_gpu_memory, 1),
                    'peak_gpu_memory_mb': round(max_gpu_memory, 1),
                    'average_training_stability': round(avg_stability, 3)
                },
                'model_performance': {
                    'performance_score': performance_score,
                    'recommended_checkpoint': checkpoint_recommendations['best_checkpoint'],
                    'model_quality_assessment': self._assess_model_quality()
                },
                'checkpoint_recommendations': checkpoint_recommendations,
                'experiment_config': self.experiment_config or {},
                'generation_time': datetime.now().isoformat()
            }
            
            # Save report to JSON
            report_path = os.path.join(self.epochs_stats_dir, "training_summary.json")
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            print(f"📋 Training report generated: training_summary.json")
            
            return report
            
        except Exception as e:
            print(f"❌ Error generating training report: {e}")
            return {'error': str(e)}
    
    def _calculate_model_performance_score(self) -> float:
        """Calculate composite model performance score"""
        try:
            # Normalize metrics to 0-1 scale
            loss_score = 1.0 / (1.0 + self.epoch_stats_df['avg_generator_loss'].iloc[-1])
            stability_score = self.epoch_stats_df['training_stability_score'].mean()
            convergence_score = self.epoch_stats_df['convergence_indicator'].max()
            
            # Weighted composite score
            performance_score = 0.5 * loss_score + 0.3 * stability_score + 0.2 * convergence_score
            
            return round(float(performance_score), 4)
        except:
            return 0.0
    
    def _calculate_resource_efficiency_score(self) -> float:
        """Calculate resource efficiency score"""
        try:
            # Based on training speed and resource usage
            avg_duration = self.epoch_stats_df['epoch_duration_minutes'].mean()
            avg_gpu_usage = self.epoch_stats_df.get('gpu_utilization', pd.Series([50.0])).mean()
            
            # Lower duration and higher GPU utilization = better efficiency
            speed_score = 1.0 / (1.0 + avg_duration / 10.0)  # Normalize around 10 min baseline
            utilization_score = avg_gpu_usage / 100.0
            
            efficiency_score = 0.6 * speed_score + 0.4 * utilization_score
            
            return round(float(efficiency_score), 4)
        except:
            return 0.5
    
    def _get_checkpoint_recommendations(self) -> Dict[str, Any]:
        """Get checkpoint recommendations"""
        try:
            # Find best performing epoch
            best_epoch = self.epoch_stats_df.loc[
                self.epoch_stats_df['avg_generator_loss'].idxmin(), 'epoch'
            ]
            
            # Find most stable epoch
            most_stable_epoch = self.epoch_stats_df.loc[
                self.epoch_stats_df['training_stability_score'].idxmax(), 'epoch'
            ]
            
            return {
                'best_checkpoint': f"checkpoint_epoch_{int(best_epoch)}.tar",
                'most_stable_checkpoint': f"checkpoint_epoch_{int(most_stable_epoch)}.tar",
                'final_checkpoint': "final_model.tar",
                'recommended_for_inference': f"checkpoint_epoch_{int(best_epoch)}.tar"
            }
        except:
            return {
                'best_checkpoint': "best_model.tar",
                'most_stable_checkpoint': "final_model.tar", 
                'final_checkpoint': "final_model.tar",
                'recommended_for_inference': "best_model.tar"
            }
    
    def _assess_model_quality(self) -> str:
        """Assess overall model quality"""
        try:
            final_loss = self.epoch_stats_df['avg_generator_loss'].iloc[-1]
            avg_stability = self.epoch_stats_df['training_stability_score'].mean()
            convergence_achieved = self.epoch_stats_df['convergence_indicator'].max() > 0.8
            
            if final_loss < 0.1 and avg_stability > 0.8 and convergence_achieved:
                return "Excellent - Low loss, high stability, converged"
            elif final_loss < 0.5 and avg_stability > 0.6:
                return "Good - Reasonable loss and stability"
            elif final_loss < 1.0 and avg_stability > 0.4:
                return "Moderate - Acceptable but could be improved"
            else:
                return "Poor - High loss or unstable training"
        except:
            return "Unknown - Insufficient data for assessment"


def generate_training_report(output_analysis_dir: str = "output_analysis"):
    """
    Convenience function to generate complete post-training analysis
    
    Args:
        output_analysis_dir: Directory containing training metrics
    """
    print("🚀 Starting post-training analysis...")
    
    analyzer = PostTrainingAnalyzer(output_analysis_dir)
    
    # Generate all plots
    analyzer.generate_all_plots()
    
    # Generate summary report
    report = analyzer.generate_training_report()
    
    if 'error' not in report:
        print("✅ Post-training analysis completed successfully!")
        print(f"📊 Plots saved to: {analyzer.plots_dir}")
        print(f"📋 Report saved to: {os.path.join(analyzer.epochs_stats_dir, 'training_summary.json')}")
        
        # Print key insights
        if 'model_performance' in report:
            score = report['model_performance']['performance_score']
            quality = report['model_performance']['model_quality_assessment']
            print(f"🎯 Model Performance Score: {score:.3f}")
            print(f"🔍 Quality Assessment: {quality}")
        
    else:
        print(f"❌ Analysis failed: {report['error']}")
    
    return report
