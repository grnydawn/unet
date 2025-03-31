import torch
import torch.nn as nn
from typing import List, Tuple, Optional

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

class UNet(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 num_blocks: int = 5,
                 base_channels: int = 64):
        super().__init__()
        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        
        # Encoder path
        current_channels = in_channels
        encoder_channels = []
        for i in range(num_blocks):
            out_channels_block = base_channels * (2 ** i)
            # print(f"Encoder block {i}: in_channels={current_channels}, out_channels={out_channels_block}")
            self.encoder_blocks.append(
                UNetBlock(current_channels, out_channels_block, is_encoder=True)
            )
            current_channels = out_channels_block
            encoder_channels.append(current_channels)
        
        # Decoder path
        for i in range(num_blocks-1, 0, -1):
            in_channels_block = current_channels  + encoder_channels[i - 1]  # Sum of upsampled and skip connection channels
            out_channels_block = encoder_channels[i-1]  # Match the corresponding encoder block
            
            # print(f"Decoder block {num_blocks-i-1}: in_channels={in_channels_block}, out_channels={out_channels_block}")
            
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
        # print("\nEncoder path:")
        for i, block in enumerate(self.encoder_blocks):
            x, pre_pool = block(x)
            if i < len(self.encoder_blocks) - 1: 
                encoder_features.append(pre_pool)
            else:
                x = pre_pool
            # print(f"Encoder block {i} output shape: {x.shape}")
            # print(f"Encoder block {i} skip connection shape: {pre_pool.shape}")
        
        # print("\nDecoder path:")
        # Decoder path
        for i, (up_conv, dec_block) in enumerate(zip(self.up_convs, self.decoder_blocks)):
            # print(f"\nDecoder step {i}:")
            # print(f"Before up-conv shape: {x.shape}")
            
            # Upsample
            x = up_conv(x)
            # print(f"After up-conv shape: {x.shape}")
            
            # Get corresponding encoder features
            skip_connection = encoder_features.pop()
            # print(f"Skip connection shape: {skip_connection.shape}")

            # Handle dimension mismatch
            if x.shape[2:] != skip_connection.shape[2:]:
                diff_h = skip_connection.size()[2] - x.size()[2]
                diff_w = skip_connection.size()[3] - x.size()[3]
                x = nn.functional.pad(x, [diff_w//2, diff_w-diff_w//2,
                                       diff_h//2, diff_h-diff_h//2])
                # print(f"After padding: {x.shape}")            
            
            # Concatenate skip connection
            x = torch.cat([x, skip_connection], dim=1)
            # print(f"After concatenation shape: {x.shape}")
            
            # Apply decoder block
            x, _ = dec_block(x)
            # print(f"After decoder block shape: {x.shape}")
        
        x = self.final_conv(x)
        # print(f"\nFinal output shape: {x.shape}")
        return x


if __name__ == "__main__":
    # Model configuration
    input_channels = 7
    height, width = 720, 1440
    batch_size = 2
    base_filters = 8
    num_blocks = 5

    # Create model and print structure
    model = UNet(in_channels=input_channels, out_channels=1, num_blocks=num_blocks, base_channels=base_filters)
    # print("\nModel structure:")
    # print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Test with dummy input
    print("\nTesting with dummy input:")
    input_tensor = torch.randn(batch_size, input_channels, height, width)
    print(f"Input shape: {input_tensor.shape}")
    
    output = model(input_tensor)
    print(f"\nFinal output shape: {output.shape}")



