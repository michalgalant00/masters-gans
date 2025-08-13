"""
WaveGAN Training Analytics - Using Shared Components
====================================================

WaveGAN-specific training analytics using the shared analysis_and_statistics module.
"""

import sys
import os

# Add analysis_and_statistics to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from analysis_and_statistics.audio.audio_analytics import AudioAnalytics

# For backward compatibility, create an alias
TrainingAnalyzer = AudioAnalytics
