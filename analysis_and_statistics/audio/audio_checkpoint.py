"""
Audio Checkpoint Manager
========================

WaveGAN-specific checkpoint management including audio samples.
"""

import os
import torch
import numpy as np
import torchaudio
from typing import Dict, List, Optional, Any

# Import base class
from ..base.checkpoint_manager import CheckpointManagerBase


class AudioCheckpointManager(CheckpointManagerBase):
    """Checkpoint manager for audio-based GANs (WaveGAN)"""
    
    def __init__(self, output_dir: str = "output_wavegan"):
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        super().__init__(checkpoint_dir, model_type="audio")
        
        # Audio-specific paths
        self.samples_dir = os.path.join(output_dir, "samples")
        os.makedirs(self.samples_dir, exist_ok=True)
    
    def _generate_checkpoint_samples(self, generator, epoch: int, iteration: int,
                                   checkpoint_type: str, output_dir: Optional[str] = None) -> List[str]:
        """Generate 5 audio sample files for checkpoint"""
        if output_dir is None:
            output_dir = self.samples_dir
        
        samples_dir = os.path.join(output_dir, f"checkpoint_epoch_{epoch}")
        os.makedirs(samples_dir, exist_ok=True)
        
        sample_paths = []
        generator.eval()
        
        with torch.no_grad():
            for i in range(5):
                # Generate random latent vector - get latent_dim from generator
                # First try to get it from generator's latent_dim attribute
                if hasattr(generator, 'latent_dim'):
                    latent_dim = generator.latent_dim
                elif hasattr(generator, 'fc') and hasattr(generator.fc, 'in_features'):
                    latent_dim = generator.fc.in_features
                elif hasattr(generator, 'dense') and hasattr(generator.dense, 'in_features'):
                    latent_dim = generator.dense.in_features
                else:
                    # Fall back to 64 which is common for WaveGAN
                    latent_dim = 64
                    
                z = torch.randn(1, latent_dim).to(next(generator.parameters()).device)
                
                # Generate audio
                fake_audio = generator(z)
                
                # Save as WAV file
                sample_path = os.path.join(samples_dir, f"sample_{i+1:02d}.wav")
                
                # Convert to CPU and squeeze batch dimension
                audio_data = fake_audio.cpu().squeeze(0).squeeze(0)
                
                # Save with torchaudio
                torchaudio.save(sample_path, audio_data.unsqueeze(0), 22050)
                sample_paths.append(sample_path)
        
        generator.train()
        return sample_paths
