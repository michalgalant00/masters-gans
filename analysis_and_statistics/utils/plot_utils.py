"""
Plot Utilities
==============

Common plotting utilities for analysis and statistics module.
"""

import matplotlib.pyplot as plt
import matplotlib.figure as mfig
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any, Union

# Try to import seaborn, use matplotlib styling if not available
try:
    import seaborn as sns
    _seaborn_available = True
    sns.set_palette("husl")
except ImportError:
    _seaborn_available = False

# Set style
plt.style.use('default')


class PlotUtils:
    """Utility functions for creating plots"""
    
    @staticmethod
    def setup_plot_style():
        """Setup consistent plot styling"""
        plt.rcParams.update({
            'figure.figsize': (10, 6),
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.dpi': 100,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight'
        })
    
    @staticmethod
    def create_subplot_grid(n_plots: int, max_cols: int = 3) -> Tuple[int, int]:
        """Calculate optimal subplot grid dimensions
        
        Args:
            n_plots: Number of plots
            max_cols: Maximum number of columns
            
        Returns:
            Tuple of (rows, cols)
        """
        if n_plots <= max_cols:
            return 1, n_plots
        
        cols = min(max_cols, n_plots)
        rows = (n_plots + cols - 1) // cols  # Ceiling division
        return rows, cols
    
    @staticmethod
    def save_plot(fig, file_path: str, close_fig: bool = True):
        """Save plot with consistent settings
        
        Args:
            fig: Matplotlib figure
            file_path: Path to save plot
            close_fig: Whether to close figure after saving
        """
        try:
            fig.savefig(file_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            
            if close_fig:
                plt.close(fig)
                
        except Exception as e:
            print(f"⚠️  Error saving plot to {file_path}: {e}")
    
    @staticmethod
    def plot_time_series(data: pd.DataFrame, x_col: str, y_cols: List[str],
                        title: str = "", xlabel: str = "", ylabel: str = "",
                        colors: Optional[List[str]] = None,
                        alpha: float = 0.7) -> mfig.Figure:
        """Create time series plot
        
        Args:
            data: DataFrame with data
            x_col: Column name for x-axis
            y_cols: List of column names for y-axis
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            colors: List of colors for each series
            alpha: Line transparency
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if colors is None:
            # Use basic color cycle
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        for i, col in enumerate(y_cols):
            if col in data.columns:
                color = colors[i % len(colors)]  # Use modulo to cycle through colors
                ax.plot(data[x_col], data[col], label=col, alpha=alpha, color=color)
        
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_distribution(data: np.ndarray, title: str = "", 
                         bins: int = 50, alpha: float = 0.7,
                         show_stats: bool = True) -> mfig.Figure:
        """Create distribution plot with statistics
        
        Args:
            data: Data array
            title: Plot title
            bins: Number of histogram bins
            alpha: Histogram transparency
            show_stats: Whether to show statistics on plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Histogram
        ax.hist(data, bins=bins, alpha=alpha, edgecolor='black')
        
        # Statistics
        if show_stats:
            mean_val = np.mean(data)
            std_val = np.std(data)
            median_val = np.median(data)
            
            ax.axvline(float(mean_val), color='red', linestyle='--', 
                      label=f'Mean: {mean_val:.3f}')
            ax.axvline(float(median_val), color='green', linestyle='--', 
                      label=f'Median: {median_val:.3f}')
            
            # Add text box with statistics
            stats_text = f'μ = {mean_val:.3f}\nσ = {std_val:.3f}\nMedian = {median_val:.3f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', 
                   facecolor='wheat', alpha=0.8))
        
        ax.set_title(title)
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_correlation_matrix(data: pd.DataFrame, title: str = "",
                               figsize: Tuple[int, int] = (10, 8)) -> mfig.Figure:
        """Create correlation matrix heatmap
        
        Args:
            data: DataFrame with numeric data
            title: Plot title
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Select only numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            print("⚠️  No numeric data for correlation matrix")
            return mfig.Figure()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate correlation matrix
        corr_matrix = numeric_data.corr()
        
        # Create heatmap
        im = ax.imshow(corr_matrix.values, cmap='coolwarm', aspect='auto',
                      vmin=-1, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(range(len(corr_matrix.columns)))
        ax.set_yticks(range(len(corr_matrix.columns)))
        ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
        ax.set_yticklabels(corr_matrix.columns)
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Add correlation values as text
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title(title)
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_loss_comparison(losses_dict: Dict[str, List[float]], 
                           title: str = "Loss Comparison",
                           smooth_window: int = 10) -> mfig.Figure:
        """Create loss comparison plot with smoothing
        
        Args:
            losses_dict: Dictionary of {loss_name: loss_values}
            title: Plot title
            smooth_window: Window size for smoothing
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Raw losses
        for name, losses in losses_dict.items():
            iterations = range(len(losses))
            ax1.plot(iterations, losses, label=name, alpha=0.7)
        
        ax1.set_title('Raw Losses')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Smoothed losses
        for name, losses in losses_dict.items():
            if len(losses) >= smooth_window:
                losses_series = pd.Series(losses)
                smoothed = losses_series.rolling(window=smooth_window).mean()
                iterations = range(len(smoothed))
                ax2.plot(iterations, smoothed, label=f'{name} (smoothed)', linewidth=2)
        
        ax2.set_title(f'Smoothed Losses (window={smooth_window})')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_training_progress(metrics_df: pd.DataFrame,
                             metric_columns: List[str],
                             title: str = "Training Progress") -> mfig.Figure:
        """Create comprehensive training progress plot
        
        Args:
            metrics_df: DataFrame with training metrics
            metric_columns: List of metric column names to plot
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        n_metrics = len(metric_columns)
        rows, cols = PlotUtils.create_subplot_grid(n_metrics, max_cols=2)
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        
        if n_metrics == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for i, metric in enumerate(metric_columns):
            if metric in metrics_df.columns:
                ax = axes[i]
                
                # Determine x-axis (iteration or epoch)
                x_col = 'iteration' if 'iteration' in metrics_df.columns else 'epoch'
                
                ax.plot(metrics_df[x_col], metrics_df[metric], alpha=0.7)
                ax.set_title(f'{metric.replace("_", " ").title()}')
                ax.set_xlabel(x_col.title())
                ax.set_ylabel(metric.replace("_", " ").title())
                ax.grid(True, alpha=0.3)
                
                # Add trend line
                if len(metrics_df) > 10:
                    z = np.polyfit(metrics_df[x_col], metrics_df[metric], 1)
                    p = np.poly1d(z)
                    ax.plot(metrics_df[x_col], p(metrics_df[x_col]), 
                           "r--", alpha=0.8, label=f'Trend')
                    ax.legend()
        
        # Hide empty subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_resource_usage_summary(resource_df: pd.DataFrame) -> mfig.Figure:
        """Create resource usage summary plot
        
        Args:
            resource_df: DataFrame with resource usage data
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        resource_columns = {
            'GPU Memory (GB)': 'gpu_memory_used',
            'GPU Utilization (%)': 'gpu_utilization', 
            'CPU Usage (%)': 'cpu_usage',
            'RAM Usage (%)': 'ram_usage'
        }
        
        for i, (title, col) in enumerate(resource_columns.items()):
            row, col_idx = i // 2, i % 2
            ax = axes[row, col_idx]
            
            if col in resource_df.columns:
                # Time series
                x_col = 'iteration' if 'iteration' in resource_df.columns else range(len(resource_df))
                ax.plot(resource_df[x_col], resource_df[col], alpha=0.7)
                
                # Statistics
                mean_val = resource_df[col].mean()
                max_val = resource_df[col].max()
                
                ax.axhline(mean_val, color='red', linestyle='--', alpha=0.7,
                          label=f'Mean: {mean_val:.1f}')
                ax.axhline(max_val, color='orange', linestyle='--', alpha=0.7,
                          label=f'Max: {max_val:.1f}')
                
                ax.set_title(title)
                ax.set_xlabel('Iteration')
                ax.set_ylabel(title.split('(')[0].strip())
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'No data for\n{title}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title)
        
        plt.suptitle('Resource Usage Summary')
        plt.tight_layout()
        return fig
