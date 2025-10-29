"""
Example: Using MyModel with Memory Fusion
示例：使用MyModel的历史记忆融合功能

展示如何：
1. 处理单张图像（无历史记忆）
2. 处理多张图像序列（累积历史记忆）
3. 控制融合权重
"""
import torch
from model.my_model import MyModel


def example_single_image():
    """示例1：处理单张图像（无历史记忆）"""
    print("="*60)
    print("Example 1: Single Image (No Memory)")
    print("="*60)
    
    # 创建模型
    model = MyModel(
        img_size=224,
        patch_size=16,
        in_c=3,
        embed_dim=144,  # 12^2
        depth=4,
        num_heads=6,
        max_len=18,
        num_chars=68,
        decoder_depth=2
    )
    model.eval()
    
    # 单张图像
    x = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        # 不传入memory_feature，正常处理
        char_probs, disc_prob, current_feature = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output char_probs: {char_probs.shape}")  # [1, 18, 68]
    print(f"Output disc_prob: {disc_prob.shape}")    # [1, 1]
    print(f"Current feature (for next memory): {current_feature.shape}")  # [1, 1, 12, 12]
    print()


def example_sequential_images():
    """示例2：处理多张图像序列，累积历史记忆"""
    print("="*60)
    print("Example 2: Sequential Images with Memory Fusion")
    print("="*60)
    
    model = MyModel(
        img_size=224,
        patch_size=16,
        embed_dim=144,
        depth=4,
        num_heads=6,
        max_len=18,
        num_chars=68,
        decoder_depth=2
    )
    model.eval()
    
    # 模拟5张连续的遮挡图像
    num_views = 5
    images = [torch.randn(1, 3, 224, 224) for _ in range(num_views)]
    
    memory_feature = None  # 初始无历史记忆
    memory_weight = 0.5    # 历史记忆权重
    
    print(f"Processing {num_views} sequential images...")
    print()
    
    with torch.no_grad():
        for i, img in enumerate(images):
            if memory_feature is None:
                print(f"View {i}: No memory (first image)")
            else:
                print(f"View {i}: Fusing with memory (weight={memory_weight})")
            
            # 前向传播
            char_probs, disc_prob, current_feature = model(
                img,
                memory_feature=memory_feature,
                memory_weight=memory_weight
            )
            
            # 更新记忆：使用当前特征作为下一次的历史记忆
            memory_feature = current_feature
            
            print(f"  -> Predicted shape: {char_probs.shape}")
            print(f"  -> Updated memory: {memory_feature.shape}")
            print()
    
    print("Final prediction includes information from all 5 views!")
    print()


def example_fusion_strategies():
    """示例3：不同的融合策略"""
    print("="*60)
    print("Example 3: Different Fusion Strategies")
    print("="*60)
    
    model = MyModel(
        img_size=224,
        patch_size=16,
        embed_dim=144,
        depth=4,
        num_heads=6,
        max_len=18,
        num_chars=68,
        decoder_depth=2
    )
    model.eval()
    
    # 两张图像
    img1 = torch.randn(1, 3, 224, 224)
    img2 = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        # 处理第一张图像
        _, _, feature1 = model(img1)
        
        # 测试不同权重
        fusion_weights = [0.0, 0.3, 0.5, 0.7, 1.0]
        
        for weight in fusion_weights:
            char_probs, _, _ = model(img2, memory_feature=feature1, memory_weight=weight)
            
            if weight == 0.0:
                strategy = "Only current image"
            elif weight == 1.0:
                strategy = "Only memory"
            else:
                strategy = f"Blend (current:{1-weight:.1f}, memory:{weight:.1f})"
            
            print(f"Weight={weight:.1f}: {strategy}")
    
    print()
    print("💡 Tip: Adjust memory_weight based on:")
    print("  - 0.0-0.3: Trust current image more (clearer views)")
    print("  - 0.4-0.6: Balanced fusion")
    print("  - 0.7-1.0: Trust memory more (current heavily occluded)")
    print()


def example_moving_average_fusion():
    """示例4：滑动平均融合（更平滑的记忆更新）"""
    print("="*60)
    print("Example 4: Moving Average Fusion")
    print("="*60)
    
    model = MyModel(
        img_size=224,
        patch_size=16,
        embed_dim=144,
        depth=4,
        num_heads=6,
        max_len=18,
        num_chars=68,
        decoder_depth=2
    )
    model.eval()
    
    num_views = 5
    images = [torch.randn(1, 3, 224, 224) for _ in range(num_views)]
    
    # 使用滑动平均：每次保留更多历史
    accumulated_feature = None
    alpha = 0.7  # 历史保留率
    
    print(f"Using exponential moving average (alpha={alpha})")
    print()
    
    with torch.no_grad():
        for i, img in enumerate(images):
            # 当前图像的特征
            char_probs, disc_prob, current_feature = model(img)
            
            if accumulated_feature is None:
                # 第一张图像，直接使用
                accumulated_feature = current_feature
                print(f"View {i}: Initialize memory")
            else:
                # 滑动平均更新
                accumulated_feature = alpha * accumulated_feature + (1 - alpha) * current_feature
                print(f"View {i}: Update memory with EMA")
                
                # 使用累积特征重新推理
                char_probs, disc_prob, _ = model(
                    img,
                    memory_feature=accumulated_feature,
                    memory_weight=0.5
                )
            
            print(f"  -> Prediction shape: {char_probs.shape}")
    
    print()
    print("💡 Moving average helps maintain stable memory across sequence")
    print()


def main():
    """运行所有示例"""
    print("\n" + "="*60)
    print("MyModel Memory Fusion Examples")
    print("="*60)
    print()
    
    # Example 1: 单张图像
    example_single_image()
    
    # Example 2: 序列图像
    example_sequential_images()
    
    # Example 3: 不同融合策略
    example_fusion_strategies()
    
    # Example 4: 滑动平均
    example_moving_average_fusion()
    
    # 总结
    print("="*60)
    print("Summary")
    print("="*60)
    print()
    print("MyModel 支持历史记忆融合功能：")
    print()
    print("1. 输入参数：")
    print("   - x: 当前图像")
    print("   - memory_feature: 历史特征（可选）")
    print("   - memory_weight: 融合权重（0-1）")
    print()
    print("2. 输出：")
    print("   - char_probs: 字符预测")
    print("   - disc_prob: 判别器输出")
    print("   - current_feature: 当前特征（用于下次记忆）")
    print()
    print("3. 处理流程：")
    print("   当前输入 → Encoder → DWT → FE → ")
    print("   融合(with memory) → IWT → Decoder → 输出")
    print()
    print("4. 应用场景：")
    print("   - 多视图车牌识别（融合5张遮挡图像）")
    print("   - 视频序列处理（累积历史帧信息）")
    print("   - 渐进式识别（逐步改善预测）")
    print()
    print("="*60)


if __name__ == '__main__':
    main()



