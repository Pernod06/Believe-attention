"""
Multi-View License Plate Recognition Model
多视图车牌识别模型 - 融合多张遮挡图像信息
"""
import torch
import torch.nn as nn
import math
from model.my_model import MyModel, VisionTransformer, Block, PatchEmbed
from functools import partial


class MultiViewFusion(nn.Module):
    """
    多视图特征融合模块
    """
    def __init__(self, embed_dim, num_views=5, fusion_type='attention'):
        """
        Args:
            embed_dim: 特征维度
            num_views: 视图数量（默认5张图像）
            fusion_type: 融合方式 ['attention', 'transformer', 'average', 'max']
        """
        super(MultiViewFusion, self).__init__()
        self.embed_dim = embed_dim
        self.num_views = num_views
        self.fusion_type = fusion_type
        
        if fusion_type == 'attention':
            # 使用注意力机制融合
            self.query = nn.Linear(embed_dim, embed_dim)
            self.key = nn.Linear(embed_dim, embed_dim)
            self.value = nn.Linear(embed_dim, embed_dim)
            self.scale = embed_dim ** -0.5
            self.out_proj = nn.Linear(embed_dim, embed_dim)
            
        elif fusion_type == 'transformer':
            # 使用Transformer层融合
            self.fusion_blocks = nn.Sequential(*[
                Block(dim=embed_dim, num_heads=8, mlp_ratio=4.0)
                for _ in range(2)
            ])
            self.view_embed = nn.Parameter(torch.zeros(1, num_views, embed_dim))
            nn.init.trunc_normal_(self.view_embed, std=0.02)
            
        elif fusion_type == 'weighted':
            # 可学习权重融合
            self.view_weights = nn.Parameter(torch.ones(num_views) / num_views)
            self.weight_proj = nn.Sequential(
                nn.Linear(embed_dim, num_views),
                nn.Softmax(dim=-1)
            )
    
    def forward(self, features):
        """
        Args:
            features: [B, num_views, embed_dim] 多个视图的特征
            
        Returns:
            fused_feature: [B, embed_dim] 融合后的特征
        """
        B, N, D = features.shape
        
        if self.fusion_type == 'average':
            # 简单平均
            return features.mean(dim=1)
        
        elif self.fusion_type == 'max':
            # 最大池化
            return features.max(dim=1)[0]
        
        elif self.fusion_type == 'attention':
            # 注意力融合
            Q = self.query(features.mean(dim=1, keepdim=True))  # [B, 1, D]
            K = self.key(features)  # [B, N, D]
            V = self.value(features)  # [B, N, D]
            
            # 计算注意力权重
            attn = (Q @ K.transpose(-2, -1)) * self.scale  # [B, 1, N]
            attn = attn.softmax(dim=-1)
            
            # 加权求和
            out = (attn @ V).squeeze(1)  # [B, D]
            out = self.out_proj(out)
            return out
        
        elif self.fusion_type == 'transformer':
            # Transformer融合
            features = features + self.view_embed
            features = self.fusion_blocks(features)
            # 取平均或第一个token
            return features.mean(dim=1)
        
        elif self.fusion_type == 'weighted':
            # 动态加权融合
            weights = self.weight_proj(features.mean(dim=1))  # [B, N]
            weights = weights.unsqueeze(-1)  # [B, N, 1]
            fused = (features * weights).sum(dim=1)  # [B, D]
            return fused


class MultiViewModel(nn.Module):
    """
    多视图车牌识别模型
    输入：5张部分遮挡的车牌图像
    输出：融合后的完整车牌识别结果
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, max_len=18, num_chars=68, decoder_depth=2, disc_hidden=256,
                 num_views=5, fusion_type='attention', share_encoder=True):
        """
        Args:
            ... (与MyModel相同的参数)
            num_views: 视图数量（默认5张图像）
            fusion_type: 融合方式 ['attention', 'transformer', 'average', 'max', 'weighted']
            share_encoder: 是否共享编码器权重
        """
        super(MultiViewModel, self).__init__()
        self.num_views = num_views
        self.share_encoder = share_encoder
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.num_chars = num_chars
        
        # 创建单视图模型
        if share_encoder:
            # 共享编码器：所有视图使用同一个编码器
            self.single_view_model = MyModel(
                img_size, patch_size, in_c, num_classes,
                embed_dim, depth, num_heads, mlp_ratio, qkv_bias,
                qk_scale, representation_size, distilled, drop_ratio,
                attn_drop_ratio, drop_path_ratio, embed_layer, norm_layer, act_layer,
                max_len, num_chars, decoder_depth, disc_hidden
            )
        else:
            # 不共享编码器：每个视图有独立的编码器
            self.view_models = nn.ModuleList([
                MyModel(
                    img_size, patch_size, in_c, num_classes,
                    embed_dim, depth, num_heads, mlp_ratio, qkv_bias,
                    qk_scale, representation_size, distilled, drop_ratio,
                    attn_drop_ratio, drop_path_ratio, embed_layer, norm_layer, act_layer,
                    max_len, num_chars, decoder_depth, disc_hidden
                )
                for _ in range(num_views)
            ])
        
        # 多视图融合模块
        self.fusion = MultiViewFusion(embed_dim, num_views, fusion_type)
        
        # 融合后的解码器
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        
        self.extra_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, decoder_depth)]
        self.decoder_blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(decoder_depth)
        ])
        
        # 输出头
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
        
        nn.init.trunc_normal_(self.extra_token, std=0.02)
    
    def extract_features(self, x):
        """
        从单张图像提取特征（在小波变换和特征增强之后）
        
        Args:
            x: [B, 3, H, W] 单张图像
            
        Returns:
            feature: [B, embed_dim] 提取的特征
        """
        if self.share_encoder:
            model = self.single_view_model
        else:
            model = self.view_models[0]  # 这里会在forward中正确处理
        
        # 使用encoder提取特征
        vec = model.encoder.forward_features(x)  # [B, embed_dim]
        B = vec.shape[0]
        
        # 应用小波变换和特征增强（复用MyModel的逻辑）
        sqrt_dim = int(math.sqrt(self.embed_dim))
        img = vec.view(B, 1, sqrt_dim, sqrt_dim)
        
        from model.my_model import haar_dwt_2d
        LL, LH, HL, HH = haar_dwt_2d(img)
        
        H_half = sqrt_dim // 2
        coeff = torch.zeros(B, 1, sqrt_dim, sqrt_dim, device=img.device)
        coeff[:, :, :H_half, :H_half] = LL
        coeff[:, :, :H_half, H_half:] = LH
        coeff[:, :, H_half:, :H_half] = HL
        coeff[:, :, H_half:, H_half:] = HH
        
        enhanced = model.fe(coeff)
        feature = enhanced.view(B, -1)
        
        return feature
    
    def forward(self, x):
        """
        Args:
            x: [B, num_views, 3, H, W] 多张遮挡图像
            
        Returns:
            char_probs: [B, max_len, num_chars] 字符概率
            disc_prob: [B, 1] 判别器输出
        """
        B, N, C, H, W = x.shape
        assert N == self.num_views, f"Expected {self.num_views} views, got {N}"
        
        # 提取每个视图的特征
        view_features = []
        for i in range(N):
            view_img = x[:, i, :, :, :]  # [B, 3, H, W]
            
            if self.share_encoder:
                feature = self.extract_features(view_img)
            else:
                # 使用对应的编码器
                vec = self.view_models[i].encoder.forward_features(view_img)
                sqrt_dim = int(math.sqrt(self.embed_dim))
                img = vec.view(B, 1, sqrt_dim, sqrt_dim)
                
                from model.my_model import haar_dwt_2d
                LL, LH, HL, HH = haar_dwt_2d(img)
                
                H_half = sqrt_dim // 2
                coeff = torch.zeros(B, 1, sqrt_dim, sqrt_dim, device=img.device)
                coeff[:, :, :H_half, :H_half] = LL
                coeff[:, :, :H_half, H_half:] = LH
                coeff[:, :, H_half:, :H_half] = HL
                coeff[:, :, H_half:, H_half:] = HH
                
                enhanced = self.view_models[i].fe(coeff)
                feature = enhanced.view(B, -1)
            
            view_features.append(feature)
        
        # 堆叠特征 [B, num_views, embed_dim]
        view_features = torch.stack(view_features, dim=1)
        
        # 融合多视图特征
        fused_feature = self.fusion(view_features)  # [B, embed_dim]
        
        # 解码器
        decoder_input = fused_feature.unsqueeze(1).repeat(1, self.max_len, 1)  # [B, max_len, embed_dim]
        extra = self.extra_token.expand(B, -1, -1)
        decoder_tokens = torch.cat([extra, decoder_input], dim=1)  # [B, max_len+1, embed_dim]
        
        decoder_out = self.decoder_blocks(decoder_tokens)
        
        # 输出
        char_logits = self.class_head(decoder_out[:, 1:])  # [B, max_len, num_chars]
        char_probs = char_logits.softmax(dim=-1)
        
        disc_logits = self.disc_head(decoder_out[:, 0])  # [B, 1]
        disc_prob = disc_logits.sigmoid()
        
        return char_probs, disc_prob


if __name__ == '__main__':
    # 测试模型
    print("Testing MultiViewModel...")
    
    model = MultiViewModel(
        img_size=224,
        patch_size=16,
        embed_dim=144,  # 12^2
        depth=4,
        num_heads=6,
        max_len=18,
        num_chars=68,
        num_views=5,
        fusion_type='attention',
        share_encoder=True
    )
    
    # 模拟输入：batch_size=2, num_views=5, 3通道, 224x224
    x = torch.randn(2, 5, 3, 224, 224)
    
    char_probs, disc_prob = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Char probs shape: {char_probs.shape}")  # [2, 18, 68]
    print(f"Disc prob shape: {disc_prob.shape}")     # [2, 1]
    print("✓ Model test passed!")



