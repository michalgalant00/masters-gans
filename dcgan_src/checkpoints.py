"""
DCGAN Checkpoint Management - Using Shared Components
=====================================================

DCGAN-specific checkpoint management using the shared analysis_and_statistics module.
"""

import sys
import os

# Add analysis_and_statistics to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from analysis_and_statistics.image.image_checkpoint import ImageCheckpointManager

# For backward compatibility, create an alias
CheckpointManager = ImageCheckpointManager
