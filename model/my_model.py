"""
original code from rwightman:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import math


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        # 保证img_size和patch_size都是tuple
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        # img_size = (img_size, img_size)
        # patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1] attn[i, j] 表示第 i 个 token 对第 j 个 token 的原始关注度
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x


class FeatureEnhance(nn.Module):
    def __init__(self):
        super(FeatureEnhance, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(2, 4, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        self.conv_final = nn.Conv2d(8, 1, kernel_size=1)

    def forward(self, x):
        residual = x
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.act3(self.conv3(x))
        x = self.conv_final(x)
        return x + residual

def haar_dwt_2d(x):
    # x: [B, 1, H, W], assume H and W even
    # Haar wavelet transform (unnormalized version for perfect reconstruction)
    # First, along rows
    low_row = (x[:, :, :, 0::2] + x[:, :, :, 1::2])
    high_row = (x[:, :, :, 0::2] - x[:, :, :, 1::2])
    # Then along columns
    LL = (low_row[:, :, 0::2, :] + low_row[:, :, 1::2, :])
    LH = (low_row[:, :, 0::2, :] - low_row[:, :, 1::2, :])
    HL = (high_row[:, :, 0::2, :] + high_row[:, :, 1::2, :])
    HH = (high_row[:, :, 0::2, :] - high_row[:, :, 1::2, :])
    return LL, LH, HL, HH

def haar_iwt_2d(LL, LH, HL, HH):
    """
    Inverse Haar Wavelet Transform 2D
    反小波变换：从四个子带重构原始图像
    
    Args:
        LL, LH, HL, HH: [B, 1, H/2, W/2] 四个子带
        
    Returns:
        x: [B, 1, H, W] 重构的图像
    """
    # 重构行方向
    low_row = torch.zeros(LL.shape[0], LL.shape[1], LL.shape[2] * 2, LL.shape[3], 
                          device=LL.device, dtype=LL.dtype)
    high_row = torch.zeros(HL.shape[0], HL.shape[1], HL.shape[2] * 2, HL.shape[3], 
                           device=HL.device, dtype=HL.dtype)
    
    # 逆变换列方向 (除以2恢复原始尺度)
    low_row[:, :, 0::2, :] = (LL + LH) / 2.0
    low_row[:, :, 1::2, :] = (LL - LH) / 2.0
    high_row[:, :, 0::2, :] = (HL + HH) / 2.0
    high_row[:, :, 1::2, :] = (HL - HH) / 2.0
    
    # 重构完整图像
    x = torch.zeros(low_row.shape[0], low_row.shape[1], low_row.shape[2], low_row.shape[3] * 2,
                    device=low_row.device, dtype=low_row.dtype)
    
    # 逆变换行方向 (除以2恢复原始尺度)
    x[:, :, :, 0::2] = (low_row + high_row) / 2.0
    x[:, :, :, 1::2] = (low_row - high_row) / 2.0
    
    return x

class MyModel(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, max_len=18, num_chars=68, decoder_depth=2, disc_hidden=256):
        super(MyModel, self).__init__()
        self.encoder = VisionTransformer(img_size, patch_size, in_c, num_classes,
                                         embed_dim, depth, num_heads, mlp_ratio, qkv_bias,
                                         qk_scale, representation_size, distilled, drop_ratio,
                                         attn_drop_ratio, drop_path_ratio, embed_layer, norm_layer, act_layer)
        self.sqrt_dim = int(math.sqrt(self.encoder.embed_dim))
        assert self.sqrt_dim ** 2 == self.encoder.embed_dim, "embed_dim must be a perfect square"
        self.fe = FeatureEnhance()
        # Decoder components
        self.max_len = max_len
        self.num_chars = num_chars
        self.extra_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, decoder_depth)]
        self.decoder_blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(decoder_depth)
        ])
        self.class_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_chars)
        )
        self.disc_head = nn.Sequential(
            nn.Linear(embed_dim, disc_hidden),
            nn.ReLU(),
            nn.Linear(disc_hidden, 1)
        )
        # Simple linear decoder to sequence logits [B, max_len * num_chars]
        self.sequence_head = nn.Sequential(
            nn.Linear(self.encoder.embed_dim, self.encoder.embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.encoder.embed_dim * 2, self.max_len * self.num_chars)
        )
        # Remove the old head as it's replaced by decoder
        # self.head = nn.Linear(self.encoder.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, memory_feature=None, memory_weight=0.5):
        """
        前向传播，支持历史记忆特征融合
        
        Args:
            x: [B, C, H, W] 输入图像
            memory_feature: [B, 1, sqrt_dim, sqrt_dim] 历史记忆特征（已经过DWT和FE）
                           如果为None，则不进行融合
            memory_weight: float, 历史记忆特征的权重（0-1）
                          融合公式: fused = (1-weight)*current + weight*memory
        
        Returns:
            char_probs: [B, max_len, num_chars] 字符概率
            disc_prob: [B, 1] 判别器概率
            current_feature: [B, 1, sqrt_dim, sqrt_dim] 当前特征（用于作为下次的memory）
        """
        vec = self.encoder.forward_features(x)  # [B, embed_dim]
        B = vec.shape[0]
        
        # Reshape to 2D image [B, 1, sqrt_dim, sqrt_dim]
        img = vec.view(B, 1, self.sqrt_dim, self.sqrt_dim)
        
        # Apply 2D DWT
        LL, LH, HL, HH = haar_dwt_2d(img)
        
        # Tile back to original size
        H_half = self.sqrt_dim // 2
        coeff = torch.zeros(B, 1, self.sqrt_dim, self.sqrt_dim, device=img.device)
        coeff[:, :, :H_half, :H_half] = LL
        coeff[:, :, :H_half, H_half:] = LH
        coeff[:, :, H_half:, :H_half] = HL
        coeff[:, :, H_half:, H_half:] = HH
        
        # Apply FE (特征增强)
        enhanced = self.fe(coeff)  # [B, 1, sqrt_dim, sqrt_dim]
        
        # 保存当前特征用于返回（可作为下次的历史记忆）
        current_feature = enhanced
        
        # 与历史记忆特征融合（加法融合）
        if memory_feature is not None:
            # 加权融合：fused = (1-weight)*current + weight*memory
            fused_feature = (1 - memory_weight) * enhanced + memory_weight * memory_feature
        else:
            fused_feature = enhanced
        
        # 分解融合后的特征为四个子带
        H_half = self.sqrt_dim // 2
        fused_LL = fused_feature[:, :, :H_half, :H_half]
        fused_LH = fused_feature[:, :, :H_half, H_half:]
        fused_HL = fused_feature[:, :, H_half:, :H_half]
        fused_HH = fused_feature[:, :, H_half:, H_half:]
        
        # 反小波变换 (IWT)
        reconstructed = haar_iwt_2d(fused_LL, fused_LH, fused_HL, fused_HH)  # [B, 1, sqrt_dim, sqrt_dim]
        
        # Flatten to vector [B, embed_dim]
        feature = reconstructed.view(B, -1)
        

        # 简单线性映射到序列输出
        sequence_logits = self.sequence_head(feature)  # [B, max_len * num_chars]
        char_logits = sequence_logits.view(B, self.max_len, self.num_chars)  # [B, max_len, num_chars]
        char_probs = char_logits.softmax(dim=-1)

        # # Create decoder input by repeating the feature for max_len positions
        # decoder_input = feature.unsqueeze(1).repeat(1, self.max_len, 1)  # [B, max_len, embed_dim]
        
        # # Add extra token
        # extra = self.extra_token.expand(B, -1, -1)
        # decoder_tokens = torch.cat([extra, decoder_input], dim=1)  # [B, max_len+1, embed_dim]
        
        # # Pass through decoder blocks
        # decoder_out = self.decoder_blocks(decoder_tokens)
        
        # # Character classification
        # char_logits = self.class_head(decoder_out[:, 1:])  # [B, max_len, num_chars]
        # char_probs = char_logits.softmax(dim=-1)
        # print(char_probs)
        
        # Discriminator output
        disc_logits = self.disc_head(feature)
        disc_prob = disc_logits.sigmoid()  # yes/no probability
        
        return char_probs, disc_prob, current_feature


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
