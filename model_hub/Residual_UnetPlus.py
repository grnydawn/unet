import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict

class UNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, is_encoder: bool = True):
        super().__init__()
        self.is_encoder = is_encoder
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )

        # Skip connection: If in_channels != out_channels, project input using 1x1 conv
        if in_channels != out_channels:
            self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        else:
            self.projection = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

        if is_encoder:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = self.projection(x)
        # print ("****", residual.shape)
        x = self.conv(x)
        # print ("****", x.shape)
        x += residual  # Add residual connection
        # print ("****", x.shape)
        x = self.relu(x)  # Apply activation after addition
        # print ("****", x.shape)

        if self.is_encoder:
            return self.pool(x), x  # Return pooled feature and skip connection
        return x, None

class ResidualUNetPlusPlus(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 num_blocks: int = 5,
                 base_channels: int = 64):
        super().__init__()
        self.num_blocks = num_blocks
        
        # Store all nested blocks in a ModuleDict
        self.blocks = nn.ModuleDict()
        
        # Store upsampling operations
        self.up_ops = nn.ModuleDict()
        
        # Store channel sizes for each node
        self.channels = {}
        
        # Initialize first column (encoder path)
        curr_channels = in_channels
        for i in range(num_blocks):
            out_ch = base_channels * (2 ** i)  # 2 ** min(i, 3)
            self.blocks[f"X{i}_0"] = UNetBlock(curr_channels, out_ch, is_encoder=True)
            self.channels[f"X{i}_0"] = out_ch
            curr_channels = out_ch
            
        # Initialize nested blocks
        for j in range(1, num_blocks):  # Skip connection level
            for i in range(num_blocks - j):  # Depth
                # Calculate input channels (sum of previous nodes at this level + upsampled features)
                in_ch = 0
                # Add channels from all previous nodes at this level
                for k in range(j):
                    in_ch += self.channels[f"X{i}_{k}"]
                # Add channels from upsampled feature
                in_ch += self.channels[f"X{i+1}_{j-1}"]
                
                out_ch = base_channels * (2 ** i)
                self.channels[f"X{i}_{j}"] = out_ch
                
                #print(f"Nested Block X{i}_{j}: in_channels={in_ch}, out_channels={out_ch}")
                
                # Create upsampling operation
                up_in_ch = self.channels[f"X{i+1}_{j-1}"]
                self.up_ops[f"up_{i}_{j}"] = nn.ConvTranspose2d(
                    up_in_ch, up_in_ch,
                    kernel_size=2,
                    stride=2
                )
                
                # Create nested block
                self.blocks[f"X{i}_{j}"] = UNetBlock(
                    in_ch,
                    out_ch,
                    is_encoder=False
                )
        
        # Final 1x1 convolution
        final_channels = self.channels[f"X0_{num_blocks-1}"]
        self.final_conv = nn.Conv2d(final_channels, out_channels, kernel_size=1)
    
    def handle_dimension_mismatch(self, x: torch.Tensor, target_shape: tuple) -> torch.Tensor:
        """Handle dimension mismatches by adding padding if necessary."""
        if x.shape[2:] != target_shape[2:]:
            diff_h = target_shape[2] - x.size()[2]
            diff_w = target_shape[3] - x.size()[3]
            
            if diff_h > 0 or diff_w > 0:
                x = F.pad(x, [
                    diff_w//2, diff_w - diff_w//2,
                    diff_h//2, diff_h - diff_h//2
                ])
                #print(f"Applied padding. New shape: {x.shape}")
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #print("\nForward pass:")
        #print(f"Input shape: {x.shape}")
        
        # Store all intermediate node outputs
        node_outputs = {}
        
        # First column (encoder path)
        curr_input = x
        for i in range(self.num_blocks):
            pooled, feature = self.blocks[f"X{i}_0"](curr_input)
            node_outputs[f"X{i}_0"] = feature
            if i < self.num_blocks:  # Don't pool at the last layer
                curr_input = pooled
            #print(f"Encoder block {i} output shape: {pooled.shape}")
            #print(f"Encoder block {i} feature shape: {feature.shape}")
        
        # Nested blocks
        for j in range(1, self.num_blocks):  # Skip connection level
            for i in range(self.num_blocks - j):  # Depth
                #print(f"\nProcessing nested block X{i}_{j}:")
                
                # Collect all previous nodes at this level
                prev_features = []
                
                # Get upsampled feature from lower level
                lower_feature = node_outputs[f"X{i+1}_{j-1}"]
                #print (lower_feature.shape)
                up_feature = self.up_ops[f"up_{i}_{j}"](lower_feature)
                #print(f"Upsampled feature shape: {up_feature.shape}")
                
                # Collect all dense connections
                for k in range(j):
                    skip_feature = node_outputs[f"X{i}_{k}"]
                    #print(f"Skip connection X{i}_{k} shape: {skip_feature.shape}")
                    
                    # Handle dimension mismatch
                    if up_feature.shape[2:] != skip_feature.shape[2:]:
                        up_feature = self.handle_dimension_mismatch(up_feature, skip_feature.shape)
                    
                    prev_features.append(skip_feature)
                
                prev_features.append(up_feature)
                
                # Concatenate all features
                combined_features = torch.cat(prev_features, dim=1)
                #print(f"Combined features shape: {combined_features.shape}")
                
                # Process through nested block
                output, _ = self.blocks[f"X{i}_{j}"](combined_features)
                node_outputs[f"X{i}_{j}"] = output
                #print(f"Nested block output shape: {output.shape}")
        
        # Get the final output (top-right node)
        final_output = node_outputs[f"X0_{self.num_blocks-1}"]
        #print(f"After decoder block shape: {x.shape}")
        
        # Final 1x1 convolution
        output = self.final_conv(final_output)
        #print(f"\nFinal output shape: {output.shape}")
        
        return output

if __name__ == "__main__":
    # Model configuration
    input_channels = 7
    height, width = 720, 1440
    batch_size = 2
    base_filters = 8
    num_blocks = 5

    # Create model
    model = ResidualUNetPlusPlus(
        in_channels=input_channels, 
        out_channels=1, 
        num_blocks=num_blocks, 
        base_channels=base_filters
    )
    
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Test with dummy input
    print("\nTesting with dummy input:")
    input_tensor = torch.randn(batch_size, input_channels, height, width)
    print(f"Input shape: {input_tensor.shape}")
    
    output = model(input_tensor)


