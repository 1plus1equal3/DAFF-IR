import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import UpSample, DownSample, AttributeFusion
from .dara import DAFF

# Encoder module for DAFF-IR
class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        # Parse configuration
        self.input_shape = config['input_shape']
        self.filters = config['filters']
        self.num = config['num']
        self.modules_num = config['modules_num']
        self.window_size = config['window_size']
        self.c_heads = config['c_heads']
        self.s_heads = config['s_heads']
        self.expansion_factor = config['expansion_factor']
        self.bias = config['bias']
        self.degrade_class = config['degrade_class']
        self.class_num = config['class_num']
        self.degrade_level = config['degrade_level']
        self.shift_size = config['shift_size']

        # Model modules
        ## Preprocess convolution layers
        self.patch_embedding = nn.Conv2d(self.input_shape[0], self.filters[0], kernel_size=3, stride=1, padding=1, bias=self.bias)
        ## Downsample layer
        self.downsamples = nn.ModuleList([DownSample(self.filters[i+1]) for i in range(self.num - 1)])
        self.downsamples.append(nn.Identity())  # Last layer does not downsample
        ## DAFF blocks
        self.daff_blocks = nn.ModuleList()
        for i in range(self.num):
            self.daff_blocks.append(
                nn.ModuleList(
                    [DAFF(
                        window_size=self.window_size[i],
                        filters=self.filters[i],
                        c_heads=self.c_heads[i],
                        s_heads=self.s_heads[i],
                        expansion_factor=self.expansion_factor,
                        bias=self.bias,
                        degrade_class=self.degrade_class,
                        class_num=self.class_num,
                        degrade_level=self.degrade_level,
                        shift_size=self.shift_size[i] if (m % 2 == 1) else 0
                    ) for m in range(self.modules_num[i])]
                )
            )
        ## Degrade classifier module
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.filters[-1], 128, bias=self.bias),
            nn.Linear(128, self.class_num, bias=self.bias),
            # nn.Softmax(dim=-1),  # Degradation class
        )
        ## Degrade estimator module
        self.estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.filters[-1], 128, bias=self.bias),
            nn.Linear(128, 1, bias=self.bias),
            # nn.Sigmoid(),  # Degradation level
        )

    def forward(self, x):
        # Initial convolution
        patches = self.patch_embedding(x)  # Shape: (batch_size, filters[0], height, width)
        x = patches
        # Temporary degradation tensors
        temp_1 = torch.zeros((x.shape[0], self.class_num), device=x.device)
        temp_2 = torch.zeros((x.shape[0], 1), device=x.device)  # For degrade_level
        # Immediate features
        feats = []
        # Encoder blocks
        for i in range(self.num):
            for j in range(self.modules_num[i]):
                x = self.daff_blocks[i][j]([x, temp_1, temp_2])
            feats.append(x) # Store features
            x = self.downsamples[i](x)  # Downsample if not the last block
        # Degradation classifier and estimator
        d_c = self.classifier(x.detach())  # Degradation class
        d_l = self.estimator(x.detach())  # Degradation level
        # Final output
        output = {
            'x': x,
            'input_patches': patches,
            'feats': feats,
            'd_c': d_c,
            'd_l': d_l,
        }
        return output

# Decoder module for DAFF-IR
class Decoder(nn.Module):
    def __init__(self, config, feat_shape):
        super(Decoder, self).__init__()
        # Parse configuration
        self.filters = config['filters']
        self.num = config['num']
        self.modules_num = config['modules_num']
        self.window_size = config['window_size']
        self.c_heads = config['c_heads']
        self.s_heads = config['s_heads']
        self.expansion_factor = config['expansion_factor']
        self.bias = config['bias']
        self.degrade_class = config['degrade_class']
        self.class_num = config['class_num']
        self.degrade_level = config['degrade_level']
        self.use_attr_fusion = config['use_attr_fusion']
        self.shift_size = config['shift_size']

        # Model modules
        ## Upsampling layers
        self.upsample_layers = nn.ModuleList([UpSample(self.filters[i]) for i in range(1, self.num)])
        self.upsample_layers.append(nn.Identity())  # Last layer does not upsample
        ## Attribute Fusion layers
        if self.use_attr_fusion:
            self.attrs_fusion = nn.ModuleList([
                AttributeFusion(
                    dim=self.class_num,
                    channel=self.filters[i] * 2,
                    reduction=4,
                    kernel_size=7,
                    spatial_size=(16, 16),
                    bias=self.bias
                ) for i in range(1, self.num)
            ])
        ## conv_1x1
        self.conv_1x1s = nn.ModuleList([
            nn.Conv2d(
                self.filters[i] * 2, 
                self.filters[i], 
                kernel_size=1, 
                stride=1, 
                bias=self.bias
            ) for i in range(1, self.num)
        ])
        ## DAFF blocks
        self.daff_blocks = nn.ModuleList()
        for i in range(1, self.num):
            self.daff_blocks.append(
                nn.ModuleList(
                    [DAFF(
                        window_size=self.window_size[i],
                        filters=self.filters[i],
                        c_heads=self.c_heads[i],
                        s_heads=self.s_heads[i],
                        expansion_factor=self.expansion_factor,
                        bias=self.bias,
                        degrade_class=self.degrade_class,
                        class_num=self.class_num,
                        degrade_level=self.degrade_level,
                        shift_size=self.shift_size[i] if (m % 2 == 1) else 0
                    ) for m in range(self.modules_num[i])]
                )
            )
        
    def forward(self, input, feat_inputs, d_c, d_l):
        # input, feat_inputs, d_c, d_l = inputs
        x = self.upsample_layers[0](input)  # Initial upsample
        # Decoder blocks
        for i in range(1, self.num):
            # Concatenate with corresponding feature
            x = torch.cat([x, feat_inputs[-1-i]], dim=1)
            if self.use_attr_fusion:
                # Apply attribute fusion
                x = self.attrs_fusion[i-1](x, d_c)
            # Apply 1x1 convolution
            x = self.conv_1x1s[i-1](x)
            # Apply DAFF blocks
            for j in range(self.modules_num[i]):
                x = self.daff_blocks[i-1][j]([x, d_c, d_l])
            x = self.upsample_layers[i](x)  # Upsample if not the last block
        return x

# Refiner module for DAFF-IR
class Refiner(nn.Module):
    def __init__(self, config):
        super(Refiner, self).__init__()
        # Parse configuration
        self.filters = config['filters']
        self.window_size = config['window_size']
        self.c_heads = config['c_heads']
        self.s_heads = config['s_heads']
        self.expansion_factor = config['expansion_factor']
        self.bias = config['bias']
        self.degrade_class = config['degrade_class']
        self.class_num = config['class_num']
        self.degrade_level = config['degrade_level']
        self.shift_size = config['shift_size']
        self.defocus = config['defocus']
        self.last_act = config['last_act']

        # Define possible last activation functions
        self.activations = {
            'sigmoid': torch.sigmoid,
            'tanh': torch.tanh,
            'relu': F.relu,
            'none': lambda x: x
        }
        if self.last_act not in self.activations:
            raise ValueError(f"Unsupported last activation function: {self.last_act}. Supported functions are: {list(self.activations.keys())}")


        # Model modules
        self.daff_block = nn.ModuleList(
            [DAFF(
                window_size=self.window_size,
                filters=self.filters,
                c_heads=self.c_heads,
                s_heads=self.s_heads,
                expansion_factor=self.expansion_factor,
                bias=self.bias,
                degrade_class=self.degrade_class,
                class_num=self.class_num,
                degrade_level=self.degrade_level,
                shift_size=self.shift_size if (m % 2 == 1) else 0
            ) for m in range(config['modules_num'])]
        )
        self.conv_output = nn.Conv2d(self.filters, 3, kernel_size=3, stride=1, padding=1, bias=self.bias)

    def forward(self, input, input_patches, d_c=None, d_l=None):
        # Forward through DAFF block
        x = input
        for block in self.daff_block:
            x = block([x, d_c, d_l])
        # If defocus is enabled, apply a defocus filter
        if self.defocus:
            x = x + input_patches  # Assuming input_patches is the defocus filter
        # Final output convolution
        output = self.conv_output(x)
        # Apply last activation if specified
        output = self.activations[self.last_act](output)
        return {
            'x': x, 
            'output': output
        }

# Decoder and Refiner combined module for DAFF-IR
class DecoderNRefiner(nn.Module):
    def __init__(self, decoder_config, refiner_config, feat_shape=None):
        super(DecoderNRefiner, self).__init__()
        # Initialize decoder and refiner
        self.decoder = Decoder(decoder_config, feat_shape)
        self.refiner = Refiner(refiner_config)

    def forward(self, input, feat_inputs, input_patches, d_c, d_l):
        # Forward through decoder
        x = self.decoder(input, feat_inputs, d_c, d_l)
        # Forward through refiner
        outputs = self.refiner(x, input_patches, d_c, d_l)
        return outputs

# Degradation-Aware Guided Filter (DAGF) module
class DAGF(nn.Module):
    def __init__(self, config, refiner_config):
        super(DAGF, self).__init__()
        # Parse configuration
        self.filters = config['filters']
        self.window_size = config['window_size']
        self.c_heads = config['c_heads']
        self.s_heads = config['s_heads']
        self.expansion_factor = config['expansion_factor']
        self.bias = config['bias']
        self.degrade_class = config['degrade_class']
        self.class_num = config['class_num']
        self.degrade_level = config['degrade_level']
        self.shift_size = config['shift_size']
        self.radius = config['radius']
        # Model modules
        self.refiner = Refiner(refiner_config)
        self.box_filter = nn.Conv2d(3, 3, kernel_size=3, padding=self.radius, dilation=self.radius, groups=3, bias=self.bias)
        self.box_filter.weight.data.fill_(1.0 / ((2*self.radius+1)**2))  # Normalize box filter
        self.map_conv_1 = nn.Conv2d(3, self.filters, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.map_conv_2 = nn.Conv2d(6, self.filters, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.map_conv_3 = nn.Conv2d(self.filters, 3, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.daff_blocks = nn.ModuleList([
            DAFF(
                window_size=self.window_size,
                filters=self.filters,
                c_heads=self.c_heads,
                s_heads=self.s_heads,
                expansion_factor=self.expansion_factor,
                bias=self.bias,
                degrade_class=self.degrade_class,
                class_num=self.class_num,
                degrade_level=self.degrade_level,
                shift_size=self.shift_size if (m % 2 == 1) else 0
            ) for m in range(config['modules_num'])
        ])

    def forward(self, x, input_patches, d_c=None, d_l=None):
        y = self.map_conv_1(x) + input_patches  # Initial mapping
        y = self.refiner(y, y, d_c, d_l)['output']
        # x: degraded image, y: clean image
        N = self.box_filter(x.new_ones(x.shape))  # Normalization factor
        # Calculate statistics of x and y
        mean_x = self.box_filter(x) / N
        mean_y = self.box_filter(y) / N
        cov_xy = self.box_filter(x * y) / N - mean_x * mean_y
        var_x = self.box_filter(x * x) / N - mean_x * mean_x
        # Expand for daff_blocks
        A = self.map_conv_2(torch.cat([cov_xy, var_x], dim=1))
        for i in range(len(self.daff_blocks)):
            A = self.daff_blocks[i]([A, d_c, d_l])
        A = self.map_conv_3(A)
        b = mean_y - A * mean_x
        output = A * x + b
        return output
    
##### Degradation-Aware Feature Fusion Image Restoration (DAFF-IR) Model #####
class DAFFIR(nn.Module):
    def __init__(self, encoder_config, decoder_config, refiner_config):
        super(DAFFIR, self).__init__()
        # Initialize encoder, decoder, and refiner
        self.encoder = Encoder(encoder_config)
        self.decoder_n_refiner = DecoderNRefiner(decoder_config, refiner_config)

    def forward(self, input):
        # Encoder forward pass
        encoder_output = self.encoder(input)
        encode_x = encoder_output['x']
        input_patches = encoder_output['input_patches']
        feats = encoder_output['feats']
        d_c = encoder_output['d_c']
        d_l = encoder_output['d_l']
        # Decoder and Refiner forward pass
        dr_output = self.decoder_n_refiner(encode_x, feats, input_patches, d_c, d_l)
        output = dr_output['output'] + input  # Add input to output
        # Final output
        outputs = {
            'degrade_outputs': [d_c, d_l],
            'restore_outputs': output,
            'input_patches': input_patches,
        }
        return outputs