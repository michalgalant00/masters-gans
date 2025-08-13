"""
WaveGAN Discriminator Network
============================

Discriminator network that classifies real vs. generated audio waveforms
using 1D convolutions for downsampling.
"""

import torch
import torch.nn as nn
import logging

class WaveGANDiscriminator(nn.Module):
    """Discriminator network for WaveGAN"""
    
    def __init__(self, input_length=44100, model_dim=128, kernel_len=25):
        super().__init__()
        self.input_length = input_length
        self.model_dim = model_dim
        self.kernel_len = kernel_len
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Discriminator layers with downsampling
        self.conv_blocks = nn.Sequential(
            # Input: [batch, 1, input_length] -> [batch, model_dim//8, input_length//2]
            nn.Conv1d(1, self.model_dim // 8, kernel_size=self.kernel_len, stride=2, padding=12),
            nn.LeakyReLU(0.2, inplace=True),
            
            # [batch, model_dim//8, length//2] -> [batch, model_dim//4, length//4]
            nn.Conv1d(self.model_dim // 8, self.model_dim // 4, kernel_size=self.kernel_len, stride=2, padding=12),
            nn.BatchNorm1d(self.model_dim // 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # [batch, model_dim//4, length//4] -> [batch, model_dim//2, length//8]
            nn.Conv1d(self.model_dim // 4, self.model_dim // 2, kernel_size=self.kernel_len, stride=2, padding=12),
            nn.BatchNorm1d(self.model_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # [batch, model_dim//2, length//8] -> [batch, model_dim, length//16]
            nn.Conv1d(self.model_dim // 2, self.model_dim, kernel_size=self.kernel_len, stride=2, padding=12),
            nn.BatchNorm1d(self.model_dim),
            nn.LeakyReLU(0.2, inplace=True),
            
            # [batch, model_dim, length//16] -> [batch, model_dim*2, length//32]
            nn.Conv1d(self.model_dim, self.model_dim * 2, kernel_size=self.kernel_len, stride=2, padding=12),
            nn.BatchNorm1d(self.model_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # [batch, model_dim*2, length//32] -> [batch, model_dim*4, length//64]
            nn.Conv1d(self.model_dim * 2, self.model_dim * 4, kernel_size=self.kernel_len, stride=2, padding=12),
            nn.BatchNorm1d(self.model_dim * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Calculate the actual size after convolutions by doing a test forward pass
        with torch.no_grad():
            test_input = torch.zeros(1, 1, self.input_length)
            test_output = self.conv_blocks(test_input)
            self.final_length = test_output.shape[2]
            self.final_features = test_output.shape[1]
        
        # Final linear layer
        self.final_fc = nn.Linear(self.final_features * self.final_length, 1)
        
        # Initialize weights
        self._initialize_weights()
        
        self.logger.info(f"WaveGAN Discriminator initialized:")
        self.logger.info(f"  - Input length: {input_length}")
        self.logger.info(f"  - Model dim: {model_dim}")
        self.logger.info(f"  - Kernel length: {kernel_len}")
        self.logger.info(f"  - Final length after conv: {self.final_length}")
        
    def _initialize_weights(self):
        """Initialize weights with normal distribution"""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
    
    def forward(self, x):
        """
        Forward pass through discriminator
        
        Args:
            x: Input audio [batch_size, 1, input_length]
            
        Returns:
            Discriminator output [batch_size, 1] (logits)
        """
        # Apply convolution blocks
        x = self.conv_blocks(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Final linear layer
        x = self.final_fc(x)
        
        return x
