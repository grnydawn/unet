import torch
import torch.nn as nn
from typing import List, Tuple, Optional

class AttentionGate(nn.Module):
    def __init__(self, F_g: int, F_l: int, F_int: int):
        """
        Attention Gate
        Args:
            F_g: Number of feature channels from decoder (gating signal)
            F_l: Number of feature channels from encoder (skip connection)
            F_int: Number of intermediate channels
        """
        super().__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Handle dimension mismatch
        if g1.shape[2:] != x1.shape[2:]:
            g1 = nn.functional.interpolate(g1, size=x1.shape[2:])
            
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi

class UNetBlock(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 is_encoder: bool = True):
        super().__init__()
        self.is_encoder = is_encoder
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        if is_encoder:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.conv(x)
        if self.is_encoder:
            return self.pool(x), x
        return x, None

class AttentionUNet(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 num_blocks: int = 4,
                 base_channels: int = 64):
        super().__init__()
        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        self.attention_gates = nn.ModuleList()
        
        # Encoder path
        current_channels = in_channels
        encoder_channels = []
        for i in range(num_blocks):
            out_channels_block = base_channels * (2 ** i)
            #print(f"Encoder block {i}: in_channels={current_channels}, out_channels={out_channels_block}")
            self.encoder_blocks.append(
                UNetBlock(current_channels, out_channels_block, is_encoder=True)
            )
            current_channels = out_channels_block
            encoder_channels.append(current_channels)
        
        # Decoder path
        for i in range(num_blocks-1, 0, -1):
            # Add attention gates
            self.attention_gates.append(
                AttentionGate(
                    F_g=current_channels,  # Upsampled features, channels from the decoder
                    F_l=encoder_channels[i-1],  # Skip connection features
                    F_int=encoder_channels[i-1]//2  # Intermediate channels
                )
            )
            
            in_channels_block = current_channels + encoder_channels[i-1]
            out_channels_block = encoder_channels[i-1]
            
            #print(f"Decoder block {num_blocks-i}: in_channels={in_channels_block}, out_channels={out_channels_block}")
            
            self.up_convs.append(
                nn.ConvTranspose2d(current_channels, current_channels, kernel_size=2, stride=2)
            )
            
            self.decoder_blocks.append(
                UNetBlock(in_channels_block, out_channels_block, is_encoder=False)
            )
            
            current_channels = out_channels_block
        
        self.final_conv = nn.Conv2d(current_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoder_features = []
        
        # Encoder path
        #print("\nEncoder path:")
        for i, block in enumerate(self.encoder_blocks):
            x, pre_pool = block(x)
            if i < len(self.encoder_blocks) - 1: 
                encoder_features.append(pre_pool)
            else:
                x = pre_pool
            #print(f"Encoder block {i} output shape: {x.shape}")
            #print(f"Encoder block {i} skip connection shape: {pre_pool.shape}")
        
        #print("\nDecoder path:")
        # Decoder path
        for i, (up_conv, dec_block, attn_gate) in enumerate(zip(self.up_convs, self.decoder_blocks, self.attention_gates)):
            #print(f"\nDecoder step {i}:")
            #print(f"Before up-conv shape: {x.shape}")
            
            # Upsample
            x = up_conv(x)
            #print(f"After up-conv shape: {x.shape}")
            
            # Get corresponding encoder features
            skip_connection = encoder_features.pop()
            #print(f"Skip connection shape: {skip_connection.shape}")
            
            # Apply attention gate
            attended_skip = attn_gate(x, skip_connection)
            #print(f"After attention gate shape: {attended_skip.shape}")

            # Handle dimension mismatch
            if x.shape[2:] != attended_skip.shape[2:]:
                diff_h = attended_skip.size()[2] - x.size()[2]
                diff_w = attended_skip.size()[3] - x.size()[3]
                x = nn.functional.pad(x, [diff_w//2, diff_w-diff_w//2,
                                       diff_h//2, diff_h-diff_h//2])
                #print(f"After padding: {x.shape}")            
            
            # Concatenate attended skip connection
            x = torch.cat([x, attended_skip], dim=1)
            #print(f"After concatenation shape: {x.shape}")
            
            # Apply decoder block
            x, _ = dec_block(x)
            #print(f"After decoder block shape: {x.shape}")
        
        x = self.final_conv(x)
        #print(f"\nFinal output shape: {x.shape}")
        return x


if __name__ == "__main__":
    # Model configuration
    input_channels = 7
    height, width = 720, 1440
    batch_size = 2
    base_filters = 8
    num_blocks = 5

    # Create model and print structure
    model = AttentionUNet(in_channels=input_channels, out_channels=1, num_blocks=num_blocks, base_channels=base_filters)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Test with dummy input
    print("\nTesting with dummy input:")
    input_tensor = torch.randn(batch_size, input_channels, height, width)
    print(f"Input shape: {input_tensor.shape}")
    
    output = model(input_tensor)
    print(f"\nFinal output shape: {output.shape}")




