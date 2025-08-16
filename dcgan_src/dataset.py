"""
Spectrogram Dataset Loading and Preprocessing
============================================

SpectrogramDataset class for loading and preprocessing spectrogram images
from directory structure with robust error handling.
"""

import torch
import pandas as pd
import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset

# Remove config import - will use passed parameters instead

class SpectrogramDataset(Dataset):
    """Dataset class for loading and preprocessing spectrogram images"""
    
    def __init__(self, spectrograms_dir, metadata_file, image_size=64, max_files=None):
        self.spectrograms_dir = spectrograms_dir
        self.image_size = image_size
        self.spectrogram_files = []
        self.use_full_path = False  # Default to using relative paths
        self.max_files = max_files  # Limit number of files for testing
        
        # Check if metadata file exists and try to use it
        if os.path.exists(metadata_file):
            try:
                self.metadata = pd.read_csv(metadata_file)
                
                # Try different possible column names for spectrogram files
                # Priority: full paths first, then just filenames
                path_columns = ['spectrogram_path', 'full_path', 'path']
                file_columns = ['spectrogram_file', 'filename', 'file', 'spectrogram_filename', 'image_file']
                
                spectrogram_column = None
                use_full_path = False
                
                # First try columns with full paths
                for col in path_columns:
                    if col in self.metadata.columns:
                        spectrogram_column = col
                        use_full_path = True
                        break
                
                # If no path column found, try filename columns
                if not spectrogram_column:
                    for col in file_columns:
                        if col in self.metadata.columns:
                            spectrogram_column = col
                            use_full_path = False
                            break
                
                if spectrogram_column:
                    self.spectrogram_files = self.metadata[spectrogram_column].tolist()
                    self.use_full_path = use_full_path
                    print(f"Loaded {len(self.spectrogram_files)} files from metadata column '{spectrogram_column}'")
                    print(f"Using {'full paths' if use_full_path else 'relative filenames'}")
                else:
                    print(f"Warning: No spectrogram filename column found in metadata.")
                    print(f"Available columns: {list(self.metadata.columns)}")
                    print("Scanning directories instead...")
                    self.spectrogram_files = self._scan_spectrogram_directories()
                    self.use_full_path = False
                    
            except Exception as e:
                print(f"Error reading metadata file: {e}")
                print("Scanning directories instead...")
                self.spectrogram_files = self._scan_spectrogram_directories()
                self.use_full_path = False
        else:
            print(f"Metadata file not found at {metadata_file}")
            print("Scanning directories for spectrogram files...")
            self.spectrogram_files = self._scan_spectrogram_directories()
            self.use_full_path = False
        
        print(f"Found {len(self.spectrogram_files)} spectrogram files total")
        
        # Apply max_files limit if specified (for testing)
        if self.max_files is not None and len(self.spectrogram_files) > self.max_files:
            print(f"🔄 Limiting dataset to {self.max_files} files for testing (from {len(self.spectrogram_files)})")
            self.spectrogram_files = self.spectrogram_files[:self.max_files]
        
        if len(self.spectrogram_files) == 0:
            raise ValueError(f"No spectrogram files found in {spectrograms_dir}")
            
        print(f"SpectrogramDataset initialized with {len(self.spectrogram_files)} files")
    
    def _scan_spectrogram_directories(self):
        """Scan directories for spectrogram image files"""
        spectrogram_files = []
        
        if not os.path.exists(self.spectrograms_dir):
            print(f"Warning: Spectrograms directory not found: {self.spectrograms_dir}")
            return spectrogram_files
        
        # Supported spectrogram file extensions
        spectrogram_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.npy']
        
        # Check if it's a flat directory or has subdirectories
        items = os.listdir(self.spectrograms_dir)
        subdirs = [d for d in items if os.path.isdir(os.path.join(self.spectrograms_dir, d))]
        files = [f for f in items if os.path.isfile(os.path.join(self.spectrograms_dir, f))]
        
        if subdirs:
            # Has subdirectories - scan each one
            print(f"Found {len(subdirs)} subdirectories: {subdirs}")
            
            for subdir in subdirs:
                subdir_path = os.path.join(self.spectrograms_dir, subdir)
                subdir_files = [f for f in os.listdir(subdir_path) 
                              if any(f.lower().endswith(ext) for ext in spectrogram_extensions)]
                
                # Add files with relative path from spectrograms_dir
                for spectrogram_file in subdir_files:
                    relative_path = os.path.join(subdir, spectrogram_file)
                    spectrogram_files.append(relative_path)
                
                print(f"  - {subdir}: {len(subdir_files)} files")
        else:
            # Flat directory - get all spectrogram files
            spectrogram_files_flat = [f for f in files 
                                    if any(f.lower().endswith(ext) for ext in spectrogram_extensions)]
            spectrogram_files.extend(spectrogram_files_flat)
            print(f"Found {len(spectrogram_files_flat)} spectrogram files in flat directory")
        
        return spectrogram_files
    
    def __len__(self):
        return len(self.spectrogram_files)
    
    def __getitem__(self, idx):
        if hasattr(self, 'use_full_path') and self.use_full_path:
            # Use full path from metadata, but fix relative paths
            spectrogram_path = self.spectrogram_files[idx]
            
            # Fix relative paths that start with "./"
            if spectrogram_path.startswith('./'):
                # Replace "./" with the actual base path
                spectrogram_path = spectrogram_path[2:]  # Remove "./"
                # Prepend the correct base path
                spectrogram_path = os.path.join('../01_dataset_prep/', spectrogram_path)
                
            # Normalize path separators for current OS
            spectrogram_path = os.path.normpath(spectrogram_path)
        else:
            # Combine directory with filename
            spectrogram_path = os.path.join(self.spectrograms_dir, self.spectrogram_files[idx])
        
        try:
            # Check file extension and load accordingly
            if spectrogram_path.lower().endswith('.npy'):
                # Load numpy array
                spectrogram_array = np.load(spectrogram_path)
                
                # Ensure it's 2D
                if spectrogram_array.ndim == 3:
                    spectrogram_array = spectrogram_array.squeeze()
                
                # Normalize to [0, 1] range
                if spectrogram_array.max() > spectrogram_array.min():
                    spectrogram_array = (spectrogram_array - spectrogram_array.min()) / (
                        spectrogram_array.max() - spectrogram_array.min())
                
                # Resize if needed
                if spectrogram_array.shape != (self.image_size, self.image_size):
                    # Convert to PIL Image for resizing
                    image = Image.fromarray((spectrogram_array * 255).astype(np.uint8), mode='L')
                    image = self._resize_image(image, self.image_size)
                    # Convert back to normalized tensor
                    image_tensor = self._image_to_tensor(image)
                else:
                    # Convert directly to tensor
                    image_tensor = torch.from_numpy(spectrogram_array).float().unsqueeze(0)  # Add channel dimension
                
            else:
                # Load as regular image file
                image = Image.open(spectrogram_path).convert('L')  # Convert to grayscale
                
                # Resize image
                image = self._resize_image(image, self.image_size)
                
                # Convert to tensor and normalize
                image_tensor = self._image_to_tensor(image)
            
            return image_tensor
            
        except Exception as e:
            print(f"Error loading image {spectrogram_path}: {e}")
            # Return a dummy image in case of error
            dummy_image = self._create_dummy_image()
            return dummy_image
    
    def _resize_image(self, image, size):
        """Resize image using available PIL method"""
        try:
            # Try to use newer PIL API
            image = image.resize((size, size), Image.Resampling.LANCZOS)
        except AttributeError:
            # Fallback for older PIL versions - just use basic resize
            image = image.resize((size, size))
        
        return image
    
    def _image_to_tensor(self, image):
        """Convert PIL image to normalized tensor"""
        # Convert to numpy array
        img_array = np.array(image, dtype=np.float32)
        
        # Normalize to [0, 1]
        img_array = img_array / 255.0
        
        # Normalize to [-1, 1] (DCGAN standard)
        img_array = (img_array - 0.5) / 0.5
        
        # Add channel dimension if needed
        if len(img_array.shape) == 2:  # Grayscale
            img_array = img_array[np.newaxis, :, :]
        
        # Convert to tensor
        return torch.from_numpy(img_array)
    
    def _create_dummy_image(self):
        """Create a dummy image tensor for error cases"""
        # Create random noise image normalized to [-1, 1]
        dummy_array = np.random.randn(1, self.image_size, self.image_size).astype(np.float32)
        # Normalize to [-1, 1] range
        dummy_array = np.tanh(dummy_array)
        return torch.from_numpy(dummy_array)
    
    def get_sample_batch(self, batch_size=9):
        """Get a sample batch for visualization"""
        indices = np.random.choice(len(self), min(batch_size, len(self)), replace=False)
        samples = []
        
        for idx in indices:
            try:
                sample = self[idx]
                samples.append(sample)
            except:
                # Add dummy sample if loading fails
                samples.append(self._create_dummy_image())
        
        return torch.stack(samples)

class DummySpectrogramDataset(Dataset):
    """Dummy dataset for testing when real spectrograms are not available"""
    
    def __init__(self, size=100, image_size=64, channels=1):
        self.size = size
        self.image_size = image_size
        self.channels = channels
        print(f"Created dummy spectrogram dataset with {size} samples of size {channels}x{image_size}x{image_size}")
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Generate random spectrogram-like data
        # Create noise with some structure (like a spectrogram might have)
        data = torch.randn(self.channels, self.image_size, self.image_size)
        
        # Add some spectral structure (horizontal patterns like spectrograms)
        for i in range(0, self.image_size, 8):
            data[:, i:i+2, :] *= 1.5  # Enhance certain frequency bands
        
        # Normalize to [-1, 1]
        data = torch.tanh(data)
        
        return data

def create_dataloader(dataset, batch_size, shuffle=True, num_workers=0):
    """Create a DataLoader for the dataset"""
    from torch.utils.data import DataLoader
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,  # Set to 0 for Windows compatibility
        pin_memory=True if torch.cuda.is_available() else False
    )
