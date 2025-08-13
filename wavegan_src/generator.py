"""
WaveGAN Generator Network
========================

Generator network that transforms noise vectors into audio waveforms
using transposed convolutions for upsampling.
"""

import torch
import torch.nn as nn
import logging

class WaveGANGenerator(nn.Module):
    """Generator network for WaveGAN"""
    
    def __init__(self, latent_dim=100, output_length=44100, model_dim=128, kernel_len=25):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_length = output_length
        self.model_dim = model_dim
        self.kernel_len = kernel_len
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Calculate initial size for upsampling to target length
        self.initial_size = 1024  # Starting size for upsampling
        
        # Linear layer to expand latent vector
        self.fc = nn.Linear(latent_dim, self.model_dim * 4 * self.initial_size)
        
        # Transposed convolution layers for upsampling
        self.conv_blocks = nn.Sequential(
            # 1024 -> 2048
            nn.ConvTranspose1d(self.model_dim * 4, self.model_dim * 2, kernel_size=self.kernel_len, stride=2, padding=12, output_padding=1),
            nn.BatchNorm1d(self.model_dim * 2),
            nn.ReLU(inplace=True),
            
            # 2048 -> 4096
            nn.ConvTranspose1d(self.model_dim * 2, self.model_dim, kernel_size=self.kernel_len, stride=2, padding=12, output_padding=1),
            nn.BatchNorm1d(self.model_dim),
            nn.ReLU(inplace=True),
            
            # 4096 -> 8192
            nn.ConvTranspose1d(self.model_dim, self.model_dim // 2, kernel_size=self.kernel_len, stride=2, padding=12, output_padding=1),
            nn.BatchNorm1d(self.model_dim // 2),
            nn.ReLU(inplace=True),
            
            # 8192 -> 16384
            nn.ConvTranspose1d(self.model_dim // 2, self.model_dim // 4, kernel_size=self.kernel_len, stride=2, padding=12, output_padding=1),
            nn.BatchNorm1d(self.model_dim // 4),
            nn.ReLU(inplace=True),
            
            # 16384 -> 32768
            nn.ConvTranspose1d(self.model_dim // 4, self.model_dim // 8, kernel_size=self.kernel_len, stride=2, padding=12, output_padding=1),
            nn.BatchNorm1d(self.model_dim // 8),
            nn.ReLU(inplace=True),
            
            # 32768 -> 65536 (or target length)
            nn.ConvTranspose1d(self.model_dim // 8, self.model_dim // 16, kernel_size=self.kernel_len, stride=2, padding=12, output_padding=1),
            nn.BatchNorm1d(self.model_dim // 16),
            nn.ReLU(inplace=True)
        )
        
        # Final convolution to get single channel output
        self.final_conv = nn.ConvTranspose1d(self.model_dim // 16, 1, kernel_size=self.kernel_len, stride=1, padding=12)
        
        # Final activation
        self.tanh = nn.Tanh()
        
        # Initialize weights
        self._initialize_weights()
        
        self.logger.info(f"WaveGAN Generator initialized:")
        self.logger.info(f"  - Latent dim: {latent_dim}")
        self.logger.info(f"  - Output length: {output_length}")
        self.logger.info(f"  - Model dim: {model_dim}")
        self.logger.info(f"  - Kernel length: {kernel_len}")
        
    def _initialize_weights(self):
        """Initialize weights with normal distribution"""
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose1d, nn.Linear)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
    
    def forward(self, z):
        """
        Forward pass through generator
        
        Args:
            z: Latent noise vector [batch_size, latent_dim]
            
        Returns:
            Generated audio [batch_size, 1, output_length]
        """
        # Expand latent vector
        x = self.fc(z)  # [batch_size, model_dim * 4 * initial_size]
        
        # Reshape for convolution
        x = x.view(x.size(0), self.model_dim * 4, self.initial_size)  # [batch_size, channels, length]
        
        # Apply transposed convolutions
        x = self.conv_blocks(x)
        
        # Final convolution
        x = self.final_conv(x)
        
        # Apply tanh activation
        x = self.tanh(x)
        
        # Crop or pad to exact output length
        if x.size(2) > self.output_length:
            x = x[:, :, :self.output_length]
        elif x.size(2) < self.output_length:
            padding = self.output_length - x.size(2)
            x = torch.nn.functional.pad(x, (0, padding))
        
        return x
    
    def generate_batch(self, batch_size, device='cpu'):
        """
        Generate a batch of random audio samples
        
        Args:
            batch_size: Number of samples to generate
            device: Device to generate on
            
        Returns:
            Generated audio batch [batch_size, 1, output_length]
        """
        self.eval()
        with torch.no_grad():
            z = torch.randn(batch_size, self.latent_dim).to(device)
            return self.forward(z)
