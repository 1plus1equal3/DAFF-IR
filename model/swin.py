import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .modules import DegradeProjection

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    def __init__(self, window_size, filters, num_heads, bias, degrade_class=False, class_num=3, degrade_level=False):
        super(WindowAttention, self).__init__()
        self.window_size = window_size
        self.filters = filters
        self.num_heads = num_heads
        self.bias = bias
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1), requires_grad=True)
        self.degrade_class = degrade_class
        self.degrade_level = degrade_level
        self.W_Q = nn.Linear(filters, filters, bias=self.bias)
        self.W_K = nn.Linear(filters, filters, bias=self.bias)
        self.W_V = nn.Linear(filters, filters, bias=self.bias)
        self.prj = nn.Linear(filters, filters, bias=self.bias)
        self.degrade_prj = DegradeProjection(class_num, window_size[0] * window_size[1], 48, num_heads, bias=self.bias)

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        torch.nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, input, d_c=None, d_l=None, mask=None):
        B_, N, C = input.shape
        query = self.W_Q(input)  # SHAPE: (B_, N, C)
        key = self.W_K(input)    # SHAPE: (B_, N, C)
        value = self.W_V(input)  # SHAPE: (B_, N, C)

        # Reshape 
        query = rearrange(query, 'b n (head c) -> b head n c', head=self.num_heads) # SHAPE: (B_, num_heads, N, C)
        key = rearrange(key, 'b n (head c) -> b head n c', head=self.num_heads)     # SHAPE: (B_, num_heads, N, C)
        value = rearrange(value, 'b n (head c) -> b head n c', head=self.num_heads)   # SHAPE: (B_, num_heads, N, C)

        # Normalize Q & K
        query = F.normalize(query, dim=-1)  # Normalize query
        key = F.normalize(key, dim=-1)      # Normalize key

        # Compute attention scores
        attn_score = query @ key.transpose(-2, -1)  # SHAPE: (B_, num_heads, N, N)
        attn_score = attn_score * self.temperature  # Scale by temperature

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn_score = attn_score + relative_position_bias.unsqueeze(0) # SHAPE: (B_, num_heads, N, N)

        # If mask is provided, apply it
        if mask is not None:
            nW = mask.shape[0]
            attn_score = attn_score.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)  # SHAPE: (B_ // nW, nW, num_heads, N, N) 
            attn_score = attn_score.view(-1, self.num_heads, N, N)  # Reshape back to (B_, num_heads, N, N)

        # Use degradation information
        if self.degrade_class:
            d_c_prj = self.degrade_prj(d_c) # SHAPE: (B_ // nW, num_heads, N, N)
            d_c_prj = d_c_prj.unsqueeze(1).expand(-1, nW, -1, -1, -1) # SHAPE: (B_ // nW, nW, num_heads, N, N)
            attn_score = attn_score.view(B_ // nW, nW, self.num_heads, N, N) * d_c_prj  # SHAPE: (B_ // nW, nW, num_heads, N, N)
            attn_score = attn_score.view(-1, self.num_heads, N, N)
        if self.degrade_level:
            d_l = d_l.view(B_, 1, 1, 1)
            attn_score = attn_score * d_l

        # Compute attention score
        attn_score = F.softmax(attn_score, dim=-1)  # SHAPE: (B_, num_heads, N, N)
        # Compute attention output
        output = attn_score @ value  # SHAPE: (B_, num_heads, N, C)
        output = rearrange(output, 'b head n c -> b n (head c)', head=self.num_heads)  # SHAPE: (B_, N, C)
        # Final projection
        output = self.prj(output)  # SHAPE: (B_, N, C)
        return output