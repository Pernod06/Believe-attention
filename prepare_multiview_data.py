"""
Helper script to prepare multi-view data from existing single-view dataset
帮助脚本：从单视图数据集准备多视图数据
"""
import os
import argparse
from pathlib import Path
import shutil
from collections import defaultdict


def convert_single_to_multiview_txt(input_txt, output_txt, num_views=5):
    """
    将单视图txt文件转换为多视图格式
    
    如果一个车牌有多张图像，会被分组
    如果一个车牌只有1张图像，会被复制num_views次
    
    Args:
        input_txt: 输入txt文件（单视图格式）
                   格式: /path/to/image.jpg 京A12345
        output_txt: 输出txt文件（多视图格式）
                    格式: /path/img1.jpg /path/img2.jpg ... 京A12345
        num_views: 每个车牌的视图数量
    """
    print(f"Converting {input_txt} to multi-view format...")
    
    # 按车牌号分组图像
    plate_groups = defaultdict(list)
    
    with open(input_txt, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 2:
                continue
            
            img_path = parts[0]
            label = parts[-1]
            
            plate_groups[label].append(img_path)
    
    # 生成多视图数据
    output_lines = []
    
    for label, img_paths in plate_groups.items():
        # 如果图像数量 >= num_views，取前num_views个
        if len(img_paths) >= num_views:
            selected_imgs = img_paths[:num_views]
        else:
            # 如果图像数量 < num_views，重复填充
            selected_imgs = img_paths.copy()
            while len(selected_imgs) < num_views:
                selected_imgs.extend(img_paths)
            selected_imgs = selected_imgs[:num_views]
        
        # 生成一行
        line = ' '.join(selected_imgs + [label])
        output_lines.append(line)
    
    # 保存
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    
    print(f"✓ Converted {len(plate_groups)} plates to {output_txt}")
    print(f"  Each plate has {num_views} views")


def create_directory_structure(input_txt, output_dir, num_views=5, copy_images=False):
    """
    从txt文件创建目录结构
    
    Args:
        input_txt: 输入txt文件
        output_dir: 输出目录
        num_views: 视图数量
        copy_images: 是否复制图像（False则创建软链接）
    """
    print(f"Creating directory structure in {output_dir}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 按车牌号分组
    plate_groups = defaultdict(list)
    
    with open(input_txt, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 2:
                continue
            
            img_path = parts[0]
            label = parts[-1]
            
            plate_groups[label].append(img_path)
    
    # 创建目录结构
    for idx, (label, img_paths) in enumerate(plate_groups.items()):
        plate_dir = os.path.join(output_dir, f'plate_{idx:06d}')
        os.makedirs(plate_dir, exist_ok=True)
        
        # 保存标签
        with open(os.path.join(plate_dir, 'label.txt'), 'w', encoding='utf-8') as f:
            f.write(label)
        
        # 处理图像
        if len(img_paths) >= num_views:
            selected_imgs = img_paths[:num_views]
        else:
            selected_imgs = img_paths.copy()
            while len(selected_imgs) < num_views:
                selected_imgs.extend(img_paths)
            selected_imgs = selected_imgs[:num_views]
        
        # 复制或链接图像
        for view_idx, src_img in enumerate(selected_imgs):
            if not os.path.exists(src_img):
                print(f"Warning: Source image not found: {src_img}")
                continue
            
            # 获取文件扩展名
            ext = os.path.splitext(src_img)[1]
            dst_img = os.path.join(plate_dir, f'view_{view_idx}{ext}')
            
            if copy_images:
                shutil.copy2(src_img, dst_img)
            else:
                # 创建软链接
                if os.path.exists(dst_img):
                    os.remove(dst_img)
                os.symlink(os.path.abspath(src_img), dst_img)
    
    print(f"✓ Created {len(plate_groups)} plate directories in {output_dir}")


def analyze_multiview_data(data_file, mode='txt'):
    """
    分析多视图数据集
    
    Args:
        data_file: 数据文件路径
        mode: 'txt' 或 'directory'
    """
    print(f"\nAnalyzing multi-view data: {data_file}")
    print("="*60)
    
    if mode == 'txt':
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        num_samples = len([l for l in lines if l.strip()])
        
        # 检查第一行
        first_line = lines[0].strip().split()
        num_images = len(first_line) - 1
        
        print(f"Format: txt")
        print(f"Total samples: {num_samples}")
        print(f"Images per sample: {num_images}")
        
        # 统计标签
        labels = [line.strip().split()[-1] for line in lines if line.strip()]
        unique_labels = set(labels)
        print(f"Unique plates: {len(unique_labels)}")
        
        # 检查图像是否存在
        missing_count = 0
        for line in lines[:10]:  # 检查前10个样本
            parts = line.strip().split()
            for img_path in parts[:-1]:
                if not os.path.exists(img_path):
                    missing_count += 1
        
        if missing_count > 0:
            print(f"⚠ Warning: {missing_count} images not found in first 10 samples")
        else:
            print(f"✓ All images found in first 10 samples")
    
    elif mode == 'directory':
        plate_dirs = [d for d in os.listdir(data_file) 
                     if os.path.isdir(os.path.join(data_file, d))]
        
        num_samples = len(plate_dirs)
        
        print(f"Format: directory")
        print(f"Total samples: {num_samples}")
        
        # 检查第一个样本
        if num_samples > 0:
            first_dir = os.path.join(data_file, plate_dirs[0])
            views = [f for f in os.listdir(first_dir) if f.startswith('view_')]
            print(f"Images per sample: {len(views)}")
            
            # 检查label.txt
            label_files = sum([1 for d in plate_dirs 
                             if os.path.exists(os.path.join(data_file, d, 'label.txt'))])
            print(f"Samples with labels: {label_files}/{num_samples}")
    
    print("="*60)


def split_dataset(input_file, output_dir, train_ratio=0.8, mode='txt'):
    """
    划分训练集和验证集
    
    Args:
        input_file: 输入数据文件
        output_dir: 输出目录
        train_ratio: 训练集比例
        mode: 'txt' 或 'directory'
    """
    import random
    
    print(f"\nSplitting dataset (train={train_ratio:.1%}, val={1-train_ratio:.1%})...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if mode == 'txt':
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f if l.strip()]
        
        # 打乱
        random.shuffle(lines)
        
        # 划分
        split_idx = int(len(lines) * train_ratio)
        train_lines = lines[:split_idx]
        val_lines = lines[split_idx:]
        
        # 保存
        train_file = os.path.join(output_dir, 'train.txt')
        val_file = os.path.join(output_dir, 'val.txt')
        
        with open(train_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(train_lines))
        
        with open(val_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(val_lines))
        
        print(f"✓ Train: {len(train_lines)} samples → {train_file}")
        print(f"✓ Val:   {len(val_lines)} samples → {val_file}")
    
    elif mode == 'directory':
        # TODO: 实现目录模式的划分
        print("Directory mode split not implemented yet")


def main():
    parser = argparse.ArgumentParser(description='Prepare multi-view LPR data')
    parser.add_argument('command', choices=['convert', 'create_dirs', 'analyze', 'split'],
                       help='command to execute')
    parser.add_argument('--input', required=True, help='input file or directory')
    parser.add_argument('--output', help='output file or directory')
    parser.add_argument('--num_views', default=5, type=int, help='number of views')
    parser.add_argument('--mode', default='txt', choices=['txt', 'directory'], 
                       help='data mode')
    parser.add_argument('--copy_images', action='store_true', 
                       help='copy images instead of creating symlinks')
    parser.add_argument('--train_ratio', default=0.8, type=float, 
                       help='train/val split ratio')
    
    args = parser.parse_args()
    
    if args.command == 'convert':
        if not args.output:
            args.output = args.input.replace('.txt', '_multiview.txt')
        convert_single_to_multiview_txt(args.input, args.output, args.num_views)
    
    elif args.command == 'create_dirs':
        if not args.output:
            args.output = './multiview_data'
        create_directory_structure(args.input, args.output, args.num_views, args.copy_images)
    
    elif args.command == 'analyze':
        analyze_multiview_data(args.input, args.mode)
    
    elif args.command == 'split':
        if not args.output:
            args.output = './split_data'
        split_dataset(args.input, args.output, args.train_ratio, args.mode)


if __name__ == '__main__':
    # 示例用法
    import sys
    
    if len(sys.argv) == 1:
        print("Multi-View Data Preparation Tool")
        print("="*60)
        print("\nUsage examples:")
        print("\n1. Convert single-view txt to multi-view txt:")
        print("   python prepare_multiview_data.py convert \\")
        print("       --input train.txt \\")
        print("       --output train_multiview.txt \\")
        print("       --num_views 5")
        print("\n2. Create directory structure:")
        print("   python prepare_multiview_data.py create_dirs \\")
        print("       --input train.txt \\")
        print("       --output ./multiview_data \\")
        print("       --num_views 5")
        print("\n3. Analyze multi-view data:")
        print("   python prepare_multiview_data.py analyze \\")
        print("       --input train_multiview.txt \\")
        print("       --mode txt")
        print("\n4. Split dataset:")
        print("   python prepare_multiview_data.py split \\")
        print("       --input data.txt \\")
        print("       --output ./split_data \\")
        print("       --train_ratio 0.8")
        print("="*60)
    else:
        main()



