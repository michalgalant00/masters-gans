"""
DCGAN Generator Architecture
===========================

Generator network for creating spectrograms from noise vectors.
Implements the standard DCGAN generator with transposed convolutions.
"""

import torch
import torch.nn as nn
from .config import LATENT_DIM, FEATURES_G, IMAGE_SIZE, CHANNELS

class DCGANGenerator(nn.Module):
    """DCGAN Generator for spectrogram generation"""
    
    def __init__(self, latent_dim=LATENT_DIM, features_g=FEATURES_G, 
                 channels=CHANNELS, image_size=IMAGE_SIZE):
        super(DCGANGenerator, self).__init__()
        
        self.latent_dim = latent_dim
        self.features_g = features_g
        self.channels = channels
        self.image_size = image_size
        
        # Calculate the initial size for the first layer
        # We need to work backwards from final image_size
        # Standard DCGAN: 4 -> 8 -> 16 -> 32 -> 64 -> 128
        if image_size == 64:
            # 64x64: 4 -> 8 -> 16 -> 32 -> 64
            self.init_size = 4
            self.num_layers = 4
        elif image_size == 128:
            # 128x128: 4 -> 8 -> 16 -> 32 -> 64 -> 128
            self.init_size = 4
            self.num_layers = 5
        else:
            raise ValueError(f"Unsupported image size: {image_size}. Use 64 or 128.")
        
        # Build the generator network
        self.main = self._build_generator()
        
        print(f"Generator initialized: {latent_dim}D -> {channels}x{image_size}x{image_size}")
        print(f"  Features: {features_g}, Layers: {self.num_layers}")
    
    def _build_generator(self):
        """Build the generator network dynamically based on image size"""
        layers = []
        
        # Calculate feature maps for each layer (decreasing by factor of 2)
        feature_maps = [self.features_g * (2**(self.num_layers - i)) for i in range(self.num_layers + 1)]
        
        # Input layer: latent_dim x 1 x 1 -> features_g*16 x 4 x 4
        layers.append(nn.ConvTranspose2d(
            self.latent_dim, feature_maps[0], 
            kernel_size=4, stride=1, padding=0, bias=False
        ))
        layers.append(nn.BatchNorm2d(feature_maps[0]))
        layers.append(nn.ReLU(True))
        
        # Hidden layers: progressively upsample
        for i in range(self.num_layers - 1):
            layers.append(nn.ConvTranspose2d(
                feature_maps[i], feature_maps[i + 1],
                kernel_size=4, stride=2, padding=1, bias=False
            ))
            layers.append(nn.BatchNorm2d(feature_maps[i + 1]))
            layers.append(nn.ReLU(True))
        
        # Output layer: final features -> channels
        layers.append(nn.ConvTranspose2d(
            feature_maps[-2], self.channels,
            kernel_size=4, stride=2, padding=1, bias=False
        ))
        layers.append(nn.Tanh())  # Output in range [-1, 1]
        
        return nn.Sequential(*layers)
    
    def forward(self, z):
        """Forward pass through generator
        
        Args:
            z: Noise tensor of shape (batch_size, latent_dim, 1, 1)
            
        Returns:
            Generated images of shape (batch_size, channels, image_size, image_size)
        """
        return self.main(z)
    
    def generate_samples(self, num_samples=1, device=None):
        """Generate sample images
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate on
            
        Returns:
            Generated images tensor
        """
        if device is None:
            device = next(self.parameters()).device
            
        self.eval()
        with torch.no_grad():
            noise = torch.randn(num_samples, self.latent_dim, 1, 1, device=device)
            samples = self(noise)
        return samples
    
    def get_model_info(self):
        """Get information about the model"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = {
            'architecture': 'DCGAN Generator',
            'input_dim': self.latent_dim,
            'output_shape': (self.channels, self.image_size, self.image_size),
            'features_g': self.features_g,
            'num_layers': self.num_layers,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }
        
        return info

def weights_init_generator(m):
    """Initialize generator weights according to DCGAN paper"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # Initialize conv weights with normal distribution (mean=0, std=0.02)
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        # Initialize BatchNorm weights with normal distribution (mean=1, std=0.02)
        # and bias with constant 0
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def test_generator(latent_dim=LATENT_DIM, features_g=FEATURES_G, 
                  image_size=IMAGE_SIZE, channels=CHANNELS, device=None):
    """Test generator with random input"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Testing Generator...")
    
    generator = DCGANGenerator(latent_dim, features_g, channels, image_size).to(device)
    generator.apply(weights_init_generator)
    
    # Test forward pass
    batch_size = 4
    noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
    
    with torch.no_grad():
        output = generator(noise)
    
    print(f"Input shape: {noise.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    # Verify output shape
    expected_shape = (batch_size, channels, image_size, image_size)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    
    # Verify output range (should be in [-1, 1] due to Tanh)
    assert -1.1 <= output.min().item() <= output.max().item() <= 1.1, "Output should be in range [-1, 1]"
    
    # Get model info
    info = generator.get_model_info()
    print(f"Model info: {info}")
    
    print("✓ Generator test passed!")
    return generator

if __name__ == "__main__":
    # Test the generator
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_generator(device=device)
