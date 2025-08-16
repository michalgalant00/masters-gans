"""
Audio Dataset Loading and Preprocessing
======================================

AudioDataset class for loading and preprocessing audio files from
directory-based class structure with robust error handling.
"""

import torch
import torchaudio
import pandas as pd
import os
import logging
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    """Dataset class for loading and preprocessing audio files"""
    
    def __init__(self, audio_dir, metadata_file, target_length=44100, sample_rate=22050, max_files_per_class=None):
        self.audio_dir = audio_dir
        self.target_length = target_length
        self.sample_rate = sample_rate
        self.max_files_per_class = max_files_per_class
        self.audio_files = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Check if metadata file exists and try to use it
        if os.path.exists(metadata_file):
            try:
                self.metadata = pd.read_csv(metadata_file)
                
                # Check if 'filename' column exists
                if 'filename' in self.metadata.columns:
                    self.audio_files = self.metadata['filename'].tolist()
                    print(f"Loaded {len(self.audio_files)} files from metadata")
                else:
                    print(f"Warning: 'filename' column not found in metadata. Available columns: {list(self.metadata.columns)}")
                    print("Scanning directories instead...")
                    self.audio_files = self._scan_audio_directories()
            except Exception as e:
                print(f"Error reading metadata file: {e}")
                print("Scanning directories instead...")
                self.audio_files = self._scan_audio_directories()
        else:
            print(f"Metadata file not found at {metadata_file}")
            self.audio_files = self._scan_audio_directories()
        
        print(f"Found {len(self.audio_files)} audio files total")
        self.logger.info(f"AudioDataset initialized with {len(self.audio_files)} files")
    
    def _scan_audio_directories(self):
        """Scan subdirectories for .wav files"""
        audio_files = []
        
        if not os.path.exists(self.audio_dir):
            print(f"Warning: Audio directory not found: {self.audio_dir}")
            return audio_files
        
        # Get all subdirectories (classes)
        subdirs = [d for d in os.listdir(self.audio_dir) 
                  if os.path.isdir(os.path.join(self.audio_dir, d))]
        
        # Scan each subdirectory for .wav files
        for subdir in subdirs:
            subdir_path = os.path.join(self.audio_dir, subdir)
            wav_files = [f for f in os.listdir(subdir_path) if f.endswith('.wav')]
            
            # Limit files per class if specified
            if self.max_files_per_class is not None:
                wav_files = wav_files[:self.max_files_per_class]
            
            # Add files with relative path from audio_dir
            for wav_file in wav_files:
                relative_path = os.path.join(subdir, wav_file)
                audio_files.append(relative_path)
        
        return audio_files
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.audio_files[idx])
        
        try:
            # Load audio file
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if necessary
            if sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
                waveform = resampler(waveform)
            
            # Pad or truncate to target length
            if waveform.shape[1] < self.target_length:
                padding = self.target_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            else:
                waveform = waveform[:, :self.target_length]
            
            # CHECK FOR NaN/Inf and replace with zeros
            if torch.isnan(waveform).any() or torch.isinf(waveform).any():
                self.logger.warning(f"NaN/Inf detected in {audio_path}, replacing with zeros")
                waveform = torch.zeros_like(waveform)
            
            # Normalize to [-1, 1] but avoid division by zero
            max_val = torch.max(torch.abs(waveform))
            if max_val > 1e-7:  # Only normalize if not silent
                waveform = waveform / max_val
            else:
                self.logger.warning(f"Silent audio detected in {audio_path}")
            
            # Final check - ensure values are in valid range
            waveform = torch.clamp(waveform, -1.0, 1.0)
            
            return waveform  # Keep channel dimension [1, length]
            
        except Exception as e:
            self.logger.error(f"Error loading {audio_path}: {e}")
            # Return zeros if file can't be loaded - keep channel dimension
            return torch.zeros(1, self.target_length)
