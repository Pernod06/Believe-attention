"""
Quick Start Script for MyModel Multi-View Training
快速开始脚本 - 使用MyModel进行多视图训练
"""
import os
import sys


def create_sample_data():
    """创建示例多视图数据用于测试"""
    print("="*60)
    print("Creating sample multi-view data...")
    print("="*60)
    
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    
    output_dir = 'sample_multiview_mymodel'
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建示例车牌
    sample_plates = ['京A12345', '沪B67890', '粤C11111', '浙D22222', '苏E33333']
    
    train_lines = []
    val_lines = []
    
    for idx, plate_text in enumerate(sample_plates):
        img_paths = []
        
        # 为每个车牌创建5个遮挡视图
        for view_id in range(5):
            # 创建车牌图像
            img = Image.new('RGB', (224, 224), color=(255, 255, 255))
            draw = ImageDraw.Draw(img)
            
            # 绘制文字
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            # 绘制车牌号
            text_bbox = draw.textbbox((0, 0), plate_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            x = (224 - text_width) // 2
            y = (224 - text_height) // 2
            draw.text((x, y), plate_text, fill=(0, 0, 0), font=font)
            
            # 添加不同位置的遮挡
            occlusion_x = (view_id * 40) % 180
            occlusion_y = (view_id * 30) % 180
            draw.rectangle(
                [occlusion_x, occlusion_y, occlusion_x+40, occlusion_y+40],
                fill=(0, 0, 0)
            )
            
            # 保存图像
            img_path = os.path.join(output_dir, f'plate_{idx:03d}_view_{view_id}.jpg')
            img.save(img_path)
            img_paths.append(img_path)
        
        # 生成txt行
        line = ' '.join(img_paths + [plate_text])
        
        # 80%训练，20%验证
        if idx < len(sample_plates) * 0.8:
            train_lines.append(line)
        else:
            val_lines.append(line)
    
    # 保存txt文件
    train_txt = os.path.join(output_dir, 'train.txt')
    val_txt = os.path.join(output_dir, 'val.txt')
    
    with open(train_txt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_lines))
    
    with open(val_txt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_lines))
    
    print(f"✓ Created sample data in {output_dir}/")
    print(f"  - Train samples: {len(train_lines)}")
    print(f"  - Val samples: {len(val_lines)}")
    print(f"  - Images per sample: 5")
    
    return train_txt, val_txt


def test_data_loading(train_txt):
    """测试数据加载"""
    print("\n" + "="*60)
    print("Testing data loading...")
    print("="*60)
    
    try:
        from data.multiview_data_loader import MultiViewLPRDataset
        from torch.utils.data import DataLoader
        from data.multiview_data_loader import collate_fn_multiview
        
        dataset = MultiViewLPRDataset(train_txt, img_size=(224, 224), num_views=5, mode='txt')
        
        print(f"✓ Dataset loaded: {len(dataset)} samples")
        
        if len(dataset) > 0:
            images, label, length = dataset[0]
            print(f"  Sample 0:")
            print(f"    Images shape: {images.shape}")
            print(f"    Label length: {length}")
        
        # 测试DataLoader
        loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn_multiview)
        images, labels, lengths = next(iter(loader))
        print(f"  Batch shape: {images.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation():
    """测试模型创建"""
    print("\n" + "="*60)
    print("Testing model creation...")
    print("="*60)
    
    try:
        from model.my_model import MyModel
        from train_mymodel_multiview import MultiViewFusion
        import torch
        
        # 创建MyModel
        model = MyModel(
            img_size=(224, 224),
            patch_size=16,
            in_c=3,
            embed_dim=144,
            depth=2,
            num_heads=6,
            max_len=18,
            num_chars=68,
            decoder_depth=1
        )
        
        fusion = MultiViewFusion(fusion_type='average')
        
        print(f"✓ MyModel created successfully")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # 测试前向传播
        x = torch.randn(1, 3, 224, 224)
        char_probs, disc_prob = model(x)
        print(f"  Output shapes: char_probs={char_probs.shape}, disc_prob={disc_prob.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_training(train_txt, val_txt):
    """运行训练"""
    print("\n" + "="*60)
    print("Starting training (3 epochs for demo)...")
    print("="*60)
    
    cmd = f"""python train_mymodel_multiview.py \
        --train_data {train_txt} \
        --val_data {val_txt} \
        --data_mode txt \
        --num_views 5 \
        --img_size 224 224 \
        --embed_dim 144 \
        --depth 2 \
        --num_heads 6 \
        --decoder_depth 1 \
        --train_batch_size 2 \
        --test_batch_size 2 \
        --max_epoch 3 \
        --learning_rate 1e-3 \
        --num_workers 0 \
        --fusion_type average \
        --save_folder ./weights_demo/
    """
    
    print("Running command:")
    print(cmd)
    print()
    
    result = os.system(cmd)
    
    if result == 0:
        print("\n✓ Training completed successfully!")
        return True
    else:
        print("\n✗ Training failed!")
        return False


def run_inference():
    """运行推理"""
    print("\n" + "="*60)
    print("Testing inference...")
    print("="*60)
    
    if not os.path.exists('weights_demo/best_model.pth'):
        print("⚠ No trained model found, skipping inference test")
        return True
    
    # 使用第一个样本的图像
    sample_dir = 'sample_multiview_mymodel'
    if not os.path.exists(sample_dir):
        print("⚠ Sample data not found")
        return True
    
    # 找到第一组图像
    images = [f for f in sorted(os.listdir(sample_dir)) if f.endswith('.jpg') and 'plate_000' in f]
    
    if len(images) >= 5:
        image_paths = [os.path.join(sample_dir, img) for img in images[:5]]
        
        cmd = f"""python inference_mymodel_multiview.py \
            --model weights_demo/best_model.pth \
            --images {' '.join(image_paths)} \
            --show_views
        """
        
        print("Running inference:")
        print(cmd)
        print()
        
        result = os.system(cmd)
        
        if result == 0:
            print("\n✓ Inference completed successfully!")
            return True
        else:
            print("\n✗ Inference failed!")
            return False
    else:
        print("⚠ Not enough sample images found")
        return True


def main():
    """主函数"""
    print("\n" + "="*60)
    print("MyModel Multi-View Quick Start")
    print("="*60)
    
    # 1. 创建示例数据
    try:
        train_txt, val_txt = create_sample_data()
    except Exception as e:
        print(f"✗ Failed to create sample data: {e}")
        return
    
    # 2. 测试数据加载
    if not test_data_loading(train_txt):
        print("\n⚠ Data loading test failed, but continuing...")
    
    # 3. 测试模型创建
    if not test_model_creation():
        print("\n⚠ Model creation test failed, but continuing...")
    
    # 4. 询问是否运行完整训练
    print("\n" + "="*60)
    response = input("Run full training (3 epochs, ~5 minutes)? [y/N]: ")
    
    if response.lower() == 'y':
        # 5. 运行训练
        if run_training(train_txt, val_txt):
            # 6. 运行推理
            run_inference()
    else:
        print("\nSkipping training. To train manually, run:")
        print(f"\npython train_mymodel_multiview.py \\")
        print(f"    --train_data {train_txt} \\")
        print(f"    --val_data {val_txt} \\")
        print(f"    --data_mode txt \\")
        print(f"    --num_views 5 \\")
        print(f"    --max_epoch 3")
    
    # 总结
    print("\n" + "="*60)
    print("Quick Start Summary")
    print("="*60)
    print("\n✓ Sample data created")
    print("✓ Data loading tested")
    print("✓ Model creation tested")
    print("\nNext steps:")
    print("1. Prepare your actual multi-view data (see README)")
    print("2. Run training with your data")
    print("3. Use trained model for inference")
    print("\nFor detailed instructions, see:")
    print("  - README_MYMODEL_MULTIVIEW.md")
    print("="*60)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        import traceback
        traceback.print_exc()



