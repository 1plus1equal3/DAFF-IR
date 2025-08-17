import torch
import torch.nn as nn
import torch.nn.functional as F

# UpSampling & DownSampling blocks
## UpSampling block
class UpSample(nn.Module):
    def __init__(self, filters=64):
        super(UpSample, self).__init__()
        self.conv = nn.Conv2d(filters * 2, filters * 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x

## DownSampling block
class DownSample(nn.Module):
    def __init__(self, filters=64):
        super(DownSample, self).__init__()
        self.conv = nn.Conv2d(filters // 2, filters // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_unshuffle(x)
        return x
    
# Torch version of DConvBlock and DegradeProjection
class DConvBlock(nn.Module):
    def __init__(self, inshape, dim=64, expansion_factor=1.0, bias=False):
        super(DConvBlock, self).__init__()
        hidden_features = int(dim*expansion_factor)
        self.conv = nn.Conv2d(inshape, hidden_features, kernel_size=1, bias=bias)
        self.depthwise = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        x = self.depthwise(x)
        return x

class DegradeProjection(nn.Module):
    def __init__(self, dim, size, conv_dim=48, num_heads=1, bias=False):
        super(DegradeProjection, self).__init__()
        self.size = size
        self.num_heads = num_heads
        self.dense = nn.Linear(dim, size ** 2 * num_heads, bias=bias)
        self.conv_1 = DConvBlock(num_heads, conv_dim, bias=bias)
        self.conv_2 = DConvBlock(conv_dim, conv_dim, bias=bias)
        self.prj = nn.Conv2d(conv_dim, num_heads, 1, 1, bias=bias)

    def forward(self, x, true_size=None):
        x = self.dense(x)
        x = x.view(-1, self.num_heads, self.size, self.size)
        if true_size is not None:
            x = F.interpolate(x, size=(true_size, true_size), mode='bilinear', align_corners=False)
        x_t = self.conv_1(x)
        x1 = F.gelu(x_t)
        x2 = self.conv_2(x_t)
        x = self.prj(x1 * x2) + x
        x = torch.sigmoid(x)  # Apply sigmoid activation
        return x # Shape: (batch_size, num_heads, height, width)
    
# Attribute Fusion Channel Attention (AFCA)
class AFCA(nn.Module):
    def __init__(self, channel, dim, reduction=8, bias=False):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Project class embedding to channel space
        self.class_fc = nn.Linear(dim, channel, bias=bias)

        self.se = nn.Sequential(
            nn.Conv2d(channel * 2, channel // reduction, 1, bias=bias),
            nn.GELU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=bias)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, d_c):
        B, C, _, _ = x.size()
        d_proj = self.class_fc(d_c).view(B, C, 1, 1)

        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)

        # Concatenate class embedding along channel dimension
        max_fused = torch.cat([max_result, d_proj], dim=1)
        avg_fused = torch.cat([avg_result, d_proj], dim=1)

        max_out = self.se(max_fused)
        avg_out = self.se(avg_fused)

        output = self.sigmoid(max_out + avg_out)
        return output

# Attribute Fusion Spatial Attention (AFSA)
class AFSA(nn.Module):
    def __init__(self, dim, kernel_size=7, spatial_size=(16, 16), bias=False):  # specify expected HxW if fixed
        super().__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=bias)
        self.sigmoid = nn.Sigmoid()

        # Project class info to spatial map
        self.class_fc = nn.Linear(dim, spatial_size[0] * spatial_size[1], bias=bias)
        self.spatial_size = spatial_size

    def forward(self, x, d_c):
        B, _, H, W = x.size()
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)

        class_map = self.class_fc(d_c).view(B, 1, self.spatial_size[0], self.spatial_size[1])
        class_map = F.interpolate(class_map, size=(H, W), mode='bilinear', align_corners=False)

        result = torch.cat([max_result, avg_result, class_map], dim=1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output

# Attribute Fusion (AFCA + AFSA)
class AttributeFusion(nn.Module):
    def __init__(self, dim, channel, reduction=8, kernel_size=7, spatial_size=(16, 16), bias=False):
        super(AttributeFusion, self).__init__()
        self.afca = AFCA(channel, dim, reduction=reduction, bias=bias)
        self.afsa = AFSA(dim, kernel_size=kernel_size, spatial_size=spatial_size, bias=bias)

    def forward(self, x, d_c):
        x_1 = x * self.afca(x, d_c)
        x_1 = x_1 * self.afsa(x_1, d_c)
        return x_1 + x