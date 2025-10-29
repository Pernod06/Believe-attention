"""
Multi-View Data Loader for existing MyModel
基于现有MyModel的多视图数据加载器
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
import glob
from data.load_data import CHARS


class MultiViewLPRDataset(Dataset):
    """
    多视图车牌数据集
    每个样本包含5张同一车牌的不同遮挡图像
    
    数据格式：
    1. txt模式：每行包含5张图像路径 + 车牌标签
       /path/img1.jpg /path/img2.jpg /path/img3.jpg /path/img4.jpg /path/img5.jpg 京A12345
    
    2. directory模式：
       plate_0001/
         view_0.jpg
         view_1.jpg
         ...
         view_4.jpg
         label.txt
    """
    
    def __init__(self, data_file, img_size=(224, 224), lpr_max_len=18, num_views=5, mode='txt', cache_images=False):
        """
        Args:
            data_file: 数据文件路径（txt文件或目录）
            img_size: 图像大小 (H, W)
            lpr_max_len: 车牌最大长度
            num_views: 视图数量
            mode: 'txt' 或 'directory'
            cache_images: 是否缓存图像到内存（加速但占用更多内存）
        """
        self.img_size = img_size
        self.lpr_max_len = lpr_max_len
        self.num_views = num_views
        self.mode = mode
        self.cache_images = cache_images
        
        # 加载数据
        self.samples = []
        self.image_cache = {} if cache_images else None
        # if mode == 'txt':
        self._load_from_txt(data_file)
        # elif mode == 'directory':
        #     self._load_from_directory(data_file)
        
        print(f"Loaded {len(self.samples)} multi-view samples ({num_views} views each)")
        if cache_images:
            print(f"Images will be cached in memory for faster loading")
    
    def _load_from_txt(self, txt_file):
        """从txt文件加载"""
        # 获取txt文件所在目录，用于处理相对路径
        txt_dir = "/home/pernod/data/CBLPRD_20percent"
        
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # 使用制表符分割（如果没有制表符则使用空格）
                parts = line.split('\t') if '\t' in line else line.split()
                if len(parts) == 2:
                    img_path_with_ext, label_str = parts
                    
                    # 去掉文件后缀（.jpg, .png等），作为目录名
                    # 例如：CBLPRD-330k/000472165.jpg -> CBLPRD-330k/000472165
                    img_path_no_ext = os.path.splitext(img_path_with_ext)[0]
                    
                    # 拼接完整目录路径
                    # /home/pernod/data/CBLPRD_20percent + CBLPRD-330k/000472165
                    full_dir_path = os.path.join(txt_dir, img_path_no_ext)
                    
                    # 检查目录是否存在
                    if not os.path.isdir(full_dir_path):
                        print(f"警告: 目录不存在，跳过: {full_dir_path}")
                        continue
                    
                    # 加载目录下的所有图片，按文件名排序
                    img_files = sorted(
                        glob.glob(os.path.join(full_dir_path, '*.jpg')) + 
                        glob.glob(os.path.join(full_dir_path, '*.png')) +
                        glob.glob(os.path.join(full_dir_path, '*.jpeg'))
                    )
                    
                    # 严格检查图片数量，不等于期望值则报错
                    if len(img_files) != self.num_views:
                        raise ValueError(
                            f"\n图片数量不匹配！\n"
                            f"  目录: {full_dir_path}\n"
                            f"  期望: {self.num_views} 张图片\n"
                            f"  实际: {len(img_files)} 张图片\n"
                            f"  找到的文件: {img_files}\n"
                            f"  请检查数据集完整性！"
                        )
                    
                    # 添加到样本列表
                    self.samples.append({'images': img_files, 'label': label_str})
    
    # def _load_from_directory(self, root_dir):
    #     """从目录加载"""
    #     for plate_dir in sorted(os.listdir(root_dir)):
    #         plate_path = os.path.join(root_dir, plate_dir)
    #         if not os.path.isdir(plate_path):
    #             continue
            
    #         label_file = os.path.join(plate_path, 'label.txt')
    #         if not os.path.exists(label_file):
    #             continue
            
    #         with open(label_file, 'r', encoding='utf-8') as f:
    #             label = f.read().strip()
            
    #         img_paths = []
    #         for i in range(self.num_views):
    #             for ext in ['.jpg', '.png']:
    #                 img_path = os.path.join(plate_path, f'view_{i}{ext}')
    #                 if os.path.exists(img_path):
    #                     img_paths.append(img_path)
    #                     break
            
    #         if len(img_paths) == self.num_views:
    #             self.samples.append({'images': img_paths, 'label': label})
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        """
        Returns:
            images: [num_views, 3, H, W] - 5张图像
            label: [label_length] - 标签索引
            length: 标签长度
        """
        sample = self.samples[index]
        img_paths = sample['images']  # 已经是完整的文件路径列表
        label_str = sample['label']
        
        # 加载所有视图
        images = []
        for img_path in img_paths:
            # 检查缓存
            if self.cache_images and img_path in self.image_cache:
                img = self.image_cache[img_path]
            else:
                img = cv2.imread(img_path)
                if img is None:
                    img = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
                else:
                    img = cv2.resize(img, (self.img_size[1], self.img_size[0]))  # (W, H)
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype('float32') / 255.0
                img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
                
                # 缓存图像
                if self.cache_images:
                    self.image_cache[img_path] = img
            
            images.append(img)
        
        images = np.stack(images, axis=0)  # [num_views, 3, H, W]
        
        # 处理标签
        label = []
        for char in label_str:
            if char in CHARS:
                label.append(CHARS.index(char))
        
        label = np.array(label[:self.lpr_max_len], dtype=np.int32)
        length = len(label)
        
        return images, label, length


def collate_fn_multiview(batch):
    """
    自定义collate函数
    
    Returns:
        images: [B, num_views, 3, H, W]
        labels: [sum(lengths)]
        lengths: [B]
    """
    images_list = []
    labels_list = []
    lengths_list = []
    
    for images, label, length in batch:
        images_list.append(torch.from_numpy(images))
        labels_list.extend(label)
        lengths_list.append(length)
    
    images_batch = torch.stack(images_list, 0)
    labels_batch = torch.from_numpy(np.asarray(labels_list).flatten().astype(np.float32))
    lengths_batch = torch.tensor(lengths_list, dtype=torch.long)
    
    return images_batch, labels_batch, lengths_batch



