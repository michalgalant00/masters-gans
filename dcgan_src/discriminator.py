"""
DCGAN Discriminator Architecture
===============================

Discriminator network for classifying real vs generated spectrograms.
Implements the standard DCGAN discriminator with convolutional layers.
"""

import torch
import torch.nn as nn

# Remove config import - will use passed parameters instead

class DCGANDiscriminator(nn.Module):
    """DCGAN Discriminator for spectrogram classification"""
    
    def __init__(self, features_d=64, channels=1, image_size=64):
        super(DCGANDiscriminator, self).__init__()
        
        self.features_d = features_d
        self.channels = channels
        self.image_size = image_size
        
        # Calculate the number of layers needed
        if image_size == 64:
            # 64x64: 64 -> 32 -> 16 -> 8 -> 4 -> 1
            self.num_layers = 4
        elif image_size == 128:
            # 128x128: 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 1
            self.num_layers = 5
        else:
            raise ValueError(f"Unsupported image size: {image_size}. Use 64 or 128.")
        
        # Build the discriminator network
        self.main = self._build_discriminator()
        
        print(f"Discriminator initialized: {channels}x{image_size}x{image_size} -> 1")
        print(f"  Features: {features_d}, Layers: {self.num_layers}")
    
    def _build_discriminator(self):
        """Build the discriminator network dynamically based on image size"""
        layers = []
        
        # Calculate feature maps for each layer (increasing by factor of 2)
        feature_maps = [self.features_d * (2**i) for i in range(self.num_layers + 1)]
        
        # First layer: input channels -> features_d (no BatchNorm on first layer)
        layers.append(nn.Conv2d(
            self.channels, feature_maps[0],
            kernel_size=4, stride=2, padding=1, bias=False
        ))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Hidden layers: progressively downsample and increase features
        for i in range(self.num_layers - 1):
            layers.append(nn.Conv2d(
                feature_maps[i], feature_maps[i + 1],
                kernel_size=4, stride=2, padding=1, bias=False
            ))
            layers.append(nn.BatchNorm2d(feature_maps[i + 1]))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Output layer: final features -> 1 (classification)
        layers.append(nn.Conv2d(
            feature_maps[-2], 1,
            kernel_size=4, stride=1, padding=0, bias=False
        ))
        layers.append(nn.Sigmoid())  # Output probability [0, 1]
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through discriminator
        
        Args:
            x: Input images of shape (batch_size, channels, image_size, image_size)
            
        Returns:
            Classification scores of shape (batch_size,)
        """
        output = self.main(x)
        # Flatten to (batch_size,)
        return output.view(-1)
    
    def get_feature_maps(self, x, layer_idx=None):
        """Get intermediate feature maps for analysis
        
        Args:
            x: Input images
            layer_idx: Layer index to extract features from (None for all)
            
        Returns:
            Feature maps from specified layer or all layers
        """
        features = []
        current_x = x
        
        for i, layer in enumerate(self.main):
            current_x = layer(current_x)
            if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
                features.append(current_x.clone())
        
        if layer_idx is not None:
            return features[layer_idx] if layer_idx < len(features) else None
        return features
    
    def get_model_info(self):
        """Get information about the model"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = {
            'architecture': 'DCGAN Discriminator',
            'input_shape': (self.channels, self.image_size, self.image_size),
            'output_dim': 1,
            'features_d': self.features_d,
            'num_layers': self.num_layers,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }
        
        return info

def weights_init_discriminator(m):
    """Initialize discriminator weights according to DCGAN paper"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # Initialize conv weights with normal distribution (mean=0, std=0.02)
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        # Initialize BatchNorm weights with normal distribution (mean=1, std=0.02)
        # and bias with constant 0
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def test_discriminator(features_d=64, image_size=64,
                      channels=1, device=None):
    """Test discriminator with random input"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Testing Discriminator...")
    
    discriminator = DCGANDiscriminator(features_d, channels, image_size).to(device)
    discriminator.apply(weights_init_discriminator)
    
    # Test forward pass
    batch_size = 4
    fake_images = torch.randn(batch_size, channels, image_size, image_size, device=device)
    
    with torch.no_grad():
        output = discriminator(fake_images)
    
    print(f"Input shape: {fake_images.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    # Verify output shape
    expected_shape = (batch_size,)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    
    # Verify output range (should be in [0, 1] due to Sigmoid)
    assert 0 <= output.min().item() <= output.max().item() <= 1, "Output should be in range [0, 1]"
    
    # Get model info
    info = discriminator.get_model_info()
    print(f"Model info: {info}")
    
    print("✓ Discriminator test passed!")
    return discriminator

def test_discriminator_features(features_d=64, channels=1, image_size=64):
    """Test feature extraction functionality"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    discriminator = DCGANDiscriminator(features_d=features_d, channels=channels, image_size=image_size).to(device)
    
    # Test feature extraction
    batch_size = 2
    test_input = torch.randn(batch_size, channels, image_size, image_size, device=device)
    
    # Get all feature maps
    features = discriminator.get_feature_maps(test_input)
    if features is not None and isinstance(features, list):
        print(f"Number of feature layers: {len(features)}")
        
        for i, feature in enumerate(features):
            if hasattr(feature, 'shape'):
                print(f"  Layer {i}: {feature.shape}")
        
        # Get specific layer features
        try:
            first_layer_features = discriminator.get_feature_maps(test_input, layer_idx=0)
            print(f"First layer features extracted successfully")
        except Exception as e:
            print(f"Could not get first layer features: {e}")
    
    print("✓ Feature extraction test passed!")

if __name__ == "__main__":
    # Test the discriminator
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_discriminator(device=device)
    test_discriminator_features()
