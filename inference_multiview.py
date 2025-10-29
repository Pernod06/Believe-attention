"""
Inference script for Multi-View License Plate Recognition
多视图车牌识别推理脚本
"""
import torch
import numpy as np
import cv2
from model.multi_view_model import MultiViewModel
from data.load_data import CHARS
import os
import argparse


class MultiViewLPRRecognizer:
    """
    多视图车牌识别器
    """
    def __init__(self, model_path, device='cuda', img_size=(94, 24), 
                 lpr_max_len=18, num_views=5, fusion_type='attention'):
        """
        Args:
            model_path: 模型权重路径
            device: 推理设备
            img_size: 图像大小 (width, height)
            lpr_max_len: 车牌最大长度
            num_views: 视图数量
            fusion_type: 融合方式
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.img_size = img_size
        self.lpr_max_len = lpr_max_len
        self.num_views = num_views
        
        # 创建模型
        num_chars = len(CHARS)
        self.model = MultiViewModel(
            img_size=(img_size[1], img_size[0]),  # (H, W)
            patch_size=(4, 4),
            in_c=3,
            embed_dim=144,  # 根据训练时的配置调整
            depth=4,
            num_heads=6,
            max_len=lpr_max_len,
            num_chars=num_chars,
            decoder_depth=2,
            num_views=num_views,
            fusion_type=fusion_type,
            share_encoder=True
        )
        
        # 加载权重
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully!")
        print(f"Device: {self.device}")
        if 'accuracy' in checkpoint:
            print(f"Model accuracy: {checkpoint['accuracy']:.4f}")
    
    def preprocess_images(self, image_paths):
        """
        预处理多张图像
        
        Args:
            image_paths: 图像路径列表（长度为num_views）
            
        Returns:
            images: [1, num_views, 3, H, W] tensor
        """
        images = []
        
        for img_path in image_paths:
            # 读取图像
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Cannot read image: {img_path}")
            
            # 调整大小
            img = cv2.resize(img, self.img_size)
            
            # 转换为RGB并归一化
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype('float32')
            img = (img / 255.0 - 0.5) * 2.0  # [-1, 1]
            
            # 转换为CHW格式
            img = np.transpose(img, (2, 0, 1))
            images.append(img)
        
        # 堆叠并转换为tensor
        images = np.stack(images, axis=0)  # [num_views, 3, H, W]
        images = torch.from_numpy(images).unsqueeze(0)  # [1, num_views, 3, H, W]
        
        return images
    
    @torch.no_grad()
    def predict(self, image_paths, decode_method='greedy'):
        """
        预测车牌
        
        Args:
            image_paths: 图像路径列表（长度为num_views）
            decode_method: 解码方法 ['greedy', 'beam_search']
            
        Returns:
            result: 预测结果字典
        """
        if len(image_paths) != self.num_views:
            raise ValueError(f"Expected {self.num_views} images, got {len(image_paths)}")
        
        # 预处理
        images = self.preprocess_images(image_paths).float().to(self.device)
        
        # 前向传播
        char_probs, disc_prob = self.model(images)  # [1, max_len, num_chars]
        
        # 解码
        if decode_method == 'greedy':
            plate_text, confidence = self._greedy_decode(char_probs[0])
        elif decode_method == 'beam_search':
            plate_text, confidence = self._beam_search_decode(char_probs[0])
        else:
            raise ValueError(f"Unknown decode method: {decode_method}")
        
        result = {
            'plate_text': plate_text,
            'confidence': confidence,
            'has_plate_prob': disc_prob.item(),
            'raw_probs': char_probs[0].cpu().numpy()
        }
        
        return result
    
    def _greedy_decode(self, char_probs):
        """
        贪婪解码（类似CTC解码）
        
        Args:
            char_probs: [max_len, num_chars] tensor
            
        Returns:
            plate_text: 车牌文字
            confidence: 平均置信度
        """
        # 转换为numpy
        probs = char_probs.cpu().numpy()
        
        # 获取最大概率的字符
        pred_indices = [np.argmax(probs[i, :]) for i in range(probs.shape[0])]
        
        # 去除重复和blank
        decoded = []
        confidences = []
        pre_c = pred_indices[0]
        
        if pre_c != len(CHARS) - 1:  # 不是blank
            decoded.append(pre_c)
            confidences.append(probs[0, pre_c])
        
        for i, c in enumerate(pred_indices[1:], 1):
            if c == len(CHARS) - 1:  # blank
                pre_c = c
                continue
            if c != pre_c:  # 不重复
                decoded.append(c)
                confidences.append(probs[i, c])
                pre_c = c
        
        # 转换为文字
        plate_text = ''.join([CHARS[idx] for idx in decoded if idx < len(CHARS) - 1])
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return plate_text, avg_confidence
    
    def _beam_search_decode(self, char_probs, beam_width=5):
        """
        束搜索解码
        
        Args:
            char_probs: [max_len, num_chars] tensor
            beam_width: 束宽度
            
        Returns:
            best_text: 最佳车牌文字
            best_confidence: 最佳置信度
        """
        probs = char_probs.cpu().numpy()
        
        # 初始化束
        beams = [{'text': '', 'indices': [], 'score': 0.0}]
        
        for t in range(probs.shape[0]):
            new_beams = []
            
            for beam in beams:
                # 对每个可能的字符扩展
                for c in range(len(CHARS)):
                    if c == len(CHARS) - 1:  # 跳过blank
                        continue
                    
                    # 检查是否重复
                    if beam['indices'] and beam['indices'][-1] == c:
                        continue
                    
                    new_beam = {
                        'text': beam['text'] + CHARS[c],
                        'indices': beam['indices'] + [c],
                        'score': beam['score'] + np.log(probs[t, c] + 1e-8)
                    }
                    new_beams.append(new_beam)
            
            # 保留top-k
            new_beams = sorted(new_beams, key=lambda x: x['score'], reverse=True)
            beams = new_beams[:beam_width]
        
        # 返回最佳结果
        best_beam = beams[0]
        return best_beam['text'], np.exp(best_beam['score'] / len(best_beam['indices']))


def main():
    parser = argparse.ArgumentParser(description='Multi-View LPR Inference')
    parser.add_argument('--model', required=True, help='path to model checkpoint')
    parser.add_argument('--images', nargs='+', required=True, help='paths to 5 view images')
    parser.add_argument('--device', default='cuda', help='device to use')
    parser.add_argument('--decode', default='greedy', choices=['greedy', 'beam_search'], 
                       help='decoding method')
    parser.add_argument('--img_size', default=[94, 24], nargs=2, type=int, 
                       help='image size [width height]')
    parser.add_argument('--num_views', default=5, type=int, help='number of views')
    
    args = parser.parse_args()
    
    if len(args.images) != args.num_views:
        print(f"Error: Expected {args.num_views} images, got {len(args.images)}")
        return
    
    # 创建识别器
    recognizer = MultiViewLPRRecognizer(
        model_path=args.model,
        device=args.device,
        img_size=tuple(args.img_size),
        num_views=args.num_views
    )
    
    # 预测
    print("\nProcessing images...")
    for i, img_path in enumerate(args.images):
        print(f"  View {i}: {img_path}")
    
    result = recognizer.predict(args.images, decode_method=args.decode)
    
    # 输出结果
    print("\n" + "="*50)
    print("Recognition Result:")
    print("="*50)
    print(f"Plate Text:      {result['plate_text']}")
    print(f"Confidence:      {result['confidence']:.4f}")
    print(f"Has Plate Prob:  {result['has_plate_prob']:.4f}")
    print("="*50)


if __name__ == '__main__':
    # 示例用法
    if len(os.sys.argv) == 1:
        print("Usage examples:")
        print("\nBasic usage:")
        print("  python inference_multiview.py \\")
        print("    --model weights_multiview/best_model.pth \\")
        print("    --images view0.jpg view1.jpg view2.jpg view3.jpg view4.jpg")
        print("\nWith beam search:")
        print("  python inference_multiview.py \\")
        print("    --model weights_multiview/best_model.pth \\")
        print("    --images view0.jpg view1.jpg view2.jpg view3.jpg view4.jpg \\")
        print("    --decode beam_search")
    else:
        main()



