import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .modules import UpSample, DownSample, DConvBlock, DegradeProjection
from .swin import WindowAttention, window_partition, window_reverse
import numbers

# Custom LayerNormalization
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
    
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def to_3d(self, x):
        # Convert 4D tensor to 3D tensor
        return rearrange(x, 'b c h w -> b (h w) c')
    
    def to_4d(self, x, h, w):
        # Convert 3D tensor back to 4D tensor
        return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

    def forward(self, x):
        h, w = x.shape[-2:]
        return self.to_4d(self.body(self.to_3d(x)), h, w)

# Torch version of cDARA
class cDARA(nn.Module):
    def __init__(self, filters, num_heads, bias, degrade_class=False, class_num=3, degrade_level=False):
        super(cDARA, self).__init__()
        self.filters = filters
        self.num_heads = num_heads
        self.bias = bias
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1), requires_grad=True)
        self.degrade_class = degrade_class
        self.degrade_level = degrade_level
        self.W_Q = DConvBlock(filters, filters, bias=self.bias)
        self.W_K = DConvBlock(filters, filters, bias=self.bias)
        self.W_V = DConvBlock(filters, filters, bias=self.bias)
        self.prj = nn.Conv2d(filters, filters, 1, bias=self.bias)
        self.degrade_prj = DegradeProjection(class_num, filters // num_heads, 48, num_heads, bias=self.bias)

    def forward(self, inputs, d_c=None, d_l=None):
        batch_size, height, width = inputs.shape[0], inputs.shape[2], inputs.shape[3]
        
        query = self.W_Q(inputs)                 # SHAPE: (batch_size, channels, height, width)
        key = self.W_K(inputs)                   # SHAPE: (batch_size, channels, height, width)
        value = self.W_V(inputs)                 # SHAPE: (batch_size, channels, height, width)

        query = rearrange(query, 'b (head c) h w -> b head c (h w)', head=self.num_heads)    # SHAPE: (batch_size, num_heads, channels, height * width)
        key = rearrange(key, 'b (head c) h w -> b head c (h w)', head=self.num_heads)        # SHAPE: (batch_size, num_heads, channels, height * width)
        value = rearrange(value, 'b (head c) h w -> b head c (h w)', head=self.num_heads)    # SHAPE: (batch_size, num_heads, channels, height * width)

        query = F.normalize(query, dim=-1)  # Normalize query
        key = F.normalize(key, dim=-1)      # Normalize key

        attn_score = query @ key.transpose(-2, -1)
        attn_score = attn_score * self.temperature # SHAPE: (batch_size, num_heads, channels, channels)

        if self.degrade_class:
            d_c_prj = self.degrade_prj(d_c)
            attn_score = attn_score * d_c_prj
        if self.degrade_level:
            d_l = d_l.view(batch_size, 1, 1, 1)
            attn_score = attn_score * d_l

        attn_score = attn_score.softmax(dim=-1)  # SHAPE: (batch_size, num_heads, channels, channels)
        output = attn_score @ value # SHAPE: (batch_size, num_heads, channels, height * width)

        output = rearrange(output, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=height, w=width) # SHAPE: (batch_size, channels, height, width)
        return self.prj(output)
        
# Torch (Swin) version of sDARA
class sDARA(nn.Module):
    def __init__(self, window_size, filters, num_heads, bias, degrade_class=False, class_num=3, degrade_level=False, shift_size=0):
        super(sDARA, self).__init__()
        self.shift_size = shift_size
        self.window_size = window_size
        self.attn = WindowAttention(
            window_size=(window_size, window_size),
            filters=filters,
            num_heads=num_heads,
            bias=bias,
            degrade_class=degrade_class,
            class_num=class_num,
            degrade_level=degrade_level
        )

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask
    

    def forward(self, input, d_c=None, d_l=None):
        B, C, H, W = input.shape

        input = rearrange(input, 'b c h w -> b h w c')
        # Cyclic shift
        if self.shift_size > 0:
            shifted_input = torch.roll(input, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_input = input

        # Partition into windows
        windows = window_partition(shifted_input, self.window_size)  # (num_windows*B, window_size, window_size, C)
        windows = windows.view(-1, self.window_size * self.window_size, C)

        # Calculate attention mask and apply window attention
        attn_mask = self.calculate_mask((H, W)).to(input.device)
        attn_windows = self.attn(windows, d_c, d_l, attn_mask)

        # Merge windows back to original shape
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)  # (num_windows*B, window_size, window_size, C)
        attn_windows = window_reverse(attn_windows, self.window_size, H, W) # (B, H, W, C)

        # Reverse cyclic shift
        if self.shift_size > 0:
            attn_windows = torch.roll(attn_windows, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        output = rearrange(attn_windows, 'b h w c -> b c h w')
        return output
    
# Fusion Feed Forward Network (FFN)
class FuseFFN(nn.Module):
    def __init__(self, filters, expansion_factor=2.0, bias=False):
        super(FuseFFN, self).__init__()
        self.dconv_1 = DConvBlock(filters * 2, filters, expansion_factor, bias=bias)
        self.dconv_2 = DConvBlock(filters * 2, filters, expansion_factor, bias=bias)
        self.conv = nn.Conv2d(int(filters * expansion_factor), filters, 1, bias=bias)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x_a = self.dconv_1(x)
        x_b = F.gelu(self.dconv_2(x))
        output = self.conv(x_a * x_b)
        return output
    
class SelectiveFuseFFN(nn.Module):
    def __init__(self, filters, expansion_factor=2.0, min_filters=32, reduction_ratio=8, bias=False):
        super(SelectiveFuseFFN, self).__init__()
        self.dconv_1 = DConvBlock(filters, filters, expansion_factor, bias=bias)
        self.dconv_2 = DConvBlock(filters, filters, expansion_factor, bias=bias)
        self.hidden_dim = int(filters * expansion_factor)
        self.reduce_filters = max(min_filters, self.hidden_dim // reduction_ratio)
        
        self.linear = nn.Linear(self.hidden_dim, self.reduce_filters, bias=bias)
        self.projections = nn.ModuleList([
            nn.Linear(self.reduce_filters, self.hidden_dim, bias=bias) for _ in range(2)
        ])
        self.conv = nn.Conv2d(self.hidden_dim, filters, kernel_size=1, bias=bias)

    def forward(self, x1, x2):
        x_a = F.gelu(self.dconv_1(x1))  # Shape: (bs, hidden_dim, h, w)
        x_b = F.gelu(self.dconv_2(x2))  # Shape: (bs, hidden_dim, h, w)
        x_feats = torch.stack([x_a, x_b], dim=1)  # Shape: (bs, 2, hidden_dim, h, w)
        pooled = (x_a + x_b).mean(dim=(2, 3))  # Shape: (bs, hidden_dim)
        x = self.linear(pooled)  # Shape: (bs, reduce_filters)
        weights = []
        for prj in self.projections:
            w = prj(x)  # Shape: (bs, hidden_dim)
            weights.append(w.view(-1, self.hidden_dim, 1, 1))  # (bs, hidden_dim, 1, 1)
        attn_weights = torch.stack(weights, dim=1)  # (bs, 2, hidden_dim, 1, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)  # Across the 2 paths
        x = (x_feats * attn_weights).sum(dim=1)  # (bs, hidden_dim, h, w)
        output = self.conv(x)  # (bs, filters, h, w)
        return output
    
class DAFF(nn.Module):
    def __init__(self, window_size, filters, c_heads, s_heads, expansion_factor, bias, norm_bias='WithBias', \
                  degrade_class=False, class_num=3, degrade_level=False, shift_size=0):
        super(DAFF, self).__init__()
        # Initialize degradation awareness regularization attention blocks
        self.c_dara = cDARA(
            filters, 
            c_heads, 
            bias, 
            degrade_class, 
            class_num, 
            degrade_level
        )
        self.s_dara = sDARA(
            window_size, 
            filters, 
            s_heads, 
            bias, 
            degrade_class, 
            class_num, 
            degrade_level,
            shift_size=shift_size
        ) 
        self.fuse_ffn = SelectiveFuseFFN(
            filters, 
            expansion_factor, 
            min_filters=48, 
            reduction_ratio=4, 
            bias=bias
        )
        self.norm1 = LayerNorm(filters, norm_bias)
        self.norm2 = LayerNorm(filters, norm_bias)
        self.norm3 = LayerNorm(filters, norm_bias)
        self.norm4 = LayerNorm(filters, norm_bias)

    def forward(self, inputs):
        input, d_c, d_l = inputs
        x1 = self.c_dara(self.norm1(input), d_c, d_l)
        x1 = self.norm2(x1 + input)
        x2 = self.s_dara(self.norm3(input), d_c, d_l)
        x2 = self.norm4(x2 + input)
        outputs = self.fuse_ffn(x1, x2) + input
        return outputs