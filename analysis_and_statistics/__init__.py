"""
Analysis and Statistics Shared Module
====================================

Comprehensive shared library for metrics collection, checkpoint management, 
and analytics across different GAN architectures.

This module eliminates code duplication between DCGAN and WaveGAN implementations
while providing robust, production-grade analysis capabilities.

Key Features:
- 80% code reuse between different GAN architectures
- Unified API for metrics, checkpointing, and analytics
- Domain-specific extensions for image and audio processing
- Production-grade error handling and monitoring

Quick Start:

For Image GANs (DCGAN):
    from analysis_and_statistics.image import ImageMetricsCollector, ImageAnalytics
    
For Audio GANs (WaveGAN):
    from analysis_and_statistics.audio import AudioMetricsCollector, AudioAnalytics
    
For Universal Components:
    from analysis_and_statistics.base import CheckpointManagerBase, SystemMonitor
"""

__version__ = "1.0.0"
__author__ = "Magisterka GANs Project"

# Base components (domain-agnostic)
from .base.metrics_collector import MetricsCollectorBase
from .base.checkpoint_manager import CheckpointManagerBase
from .base.system_monitor import SystemMonitor
from .base.analytics_base import AnalyticsBase

# Image-specific components
try:
    from .image.image_metrics import ImageMetricsCollector
    from .image.image_analytics import ImageAnalytics
    _image_available = True
except ImportError as e:
    print(f"⚠️  Image module not available: {e}")
    _image_available = False

# Audio-specific components  
try:
    from .audio.audio_metrics import AudioMetricsCollector
    from .audio.audio_analytics import AudioAnalytics
    _audio_available = True
except ImportError as e:
    print(f"⚠️  Audio module not available: {e}")
    _audio_available = False

# Utilities
from .utils.file_utils import FileUtils
from .utils.plot_utils import PlotUtils

# Core exports (always available)
__all__ = [
    'MetricsCollectorBase',
    'CheckpointManagerBase', 
    'SystemMonitor',
    'AnalyticsBase',
    'FileUtils',
    'PlotUtils'
]

# Add image exports if available
if _image_available:
    __all__.extend(['ImageMetricsCollector', 'ImageAnalytics'])

# Add audio exports if available  
if _audio_available:
    __all__.extend(['AudioMetricsCollector', 'AudioAnalytics'])

# Compatibility aliases for backward compatibility
MetricsCollector = MetricsCollectorBase  # Generic alias
CheckpointManager = CheckpointManagerBase  # Generic alias
Analytics = AnalyticsBase  # Generic alias

def get_module_info():
    """Get information about the module and its capabilities"""
    return {
        'version': __version__,
        'author': __author__,
        'image_support': _image_available,
        'audio_support': _audio_available,
        'available_classes': __all__
    }
