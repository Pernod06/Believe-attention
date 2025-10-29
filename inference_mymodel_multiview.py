"""
Inference script for MyModel with multi-view fusion
使用MyModel进行多视图车牌识别推理
"""
import torch
import numpy as np
import cv2
from model.my_model import MyModel
from train_mymodel_multiview import MultiViewFusion
from data.load_data import CHARS
import argparse


class MyModelMultiViewRecognizer:
    """
    基于MyModel的多视图车牌识别器
    """
    def __init__(self, model_path, device='cuda', img_size=(224, 224),
                 lpr_max_len=18, num_views=5, fusion_type='average'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.img_size = img_size
        self.lpr_max_len = lpr_max_len
        self.num_views = num_views
        
        num_chars = len(CHARS)
        
        # 创建MyModel
        self.model = MyModel(
            img_size=img_size,
            patch_size=16,
            in_c=3,
            embed_dim=144,
            depth=4,
            num_heads=6,
            max_len=lpr_max_len,
            num_chars=num_chars,
            decoder_depth=2
        )
        
        # 创建融合层
        self.fusion_layer = MultiViewFusion(fusion_type=fusion_type)
        
        # 加载权重
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.fusion_layer.load_state_dict(checkpoint['fusion_state_dict'])
        
        self.model.to(self.device)
        self.fusion_layer.to(self.device)
        
        self.model.eval()
        self.fusion_layer.eval()
        
        print(f"✓ Model loaded successfully!")
        if 'accuracy' in checkpoint:
            print(f"  Model accuracy: {checkpoint['accuracy']:.4f}")
    
    def preprocess_image(self, img_path):
        """预处理单张图像"""
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")
        
        img = cv2.resize(img, (self.img_size[1], self.img_size[0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype('float32') / 255.0
        img = np.transpose(img, (2, 0, 1))
        
        return img
    
    @torch.no_grad()
    def predict(self, image_paths):
        """
        预测车牌
        
        Args:
            image_paths: 图像路径列表（长度为num_views）
            
        Returns:
            result: 预测结果字典
        """
        if len(image_paths) != self.num_views:
            raise ValueError(f"Expected {self.num_views} images, got {len(image_paths)}")
        
        # 预处理所有图像
        images = []
        for img_path in image_paths:
            img = self.preprocess_image(img_path)
            images.append(img)
        
        images = np.stack(images, axis=0)
        images = torch.from_numpy(images).unsqueeze(0).float()  # [1, num_views, 3, H, W]
        images = images.to(self.device)
        
        B, N, C, H, W = images.shape
        
        # 对每个视图分别推理
        char_probs_list = []
        disc_probs_list = []
        
        for view_idx in range(N):
            view_img = images[:, view_idx, :, :, :]
            char_probs, disc_prob = self.model(view_img)
            char_probs_list.append(char_probs)
            disc_probs_list.append(disc_prob)
        
        # 融合
        fused_char_probs, fused_disc_prob = self.fusion_layer(char_probs_list, disc_probs_list)
        
        # 解码
        plate_text, confidence = self._greedy_decode(fused_char_probs[0])
        
        # 获取每个视图的预测（用于调试）
        view_predictions = []
        for i in range(N):
            view_text, view_conf = self._greedy_decode(char_probs_list[i][0])
            view_predictions.append({
                'text': view_text,
                'confidence': view_conf,
                'disc_prob': disc_probs_list[i][0].item()
            })
        
        result = {
            'plate_text': plate_text,
            'confidence': confidence,
            'has_plate_prob': fused_disc_prob[0].item(),
            'view_predictions': view_predictions
        }
        
        return result
    
    def _greedy_decode(self, char_probs):
        """贪婪解码"""
        probs = char_probs.cpu().numpy()
        
        pred_indices = [np.argmax(probs[i, :]) for i in range(probs.shape[0])]
        
        decoded = []
        confidences = []
        pre_c = pred_indices[0]
        
        if pre_c != len(CHARS) - 1:
            decoded.append(pre_c)
            confidences.append(probs[0, pre_c])
        
        for i, c in enumerate(pred_indices[1:], 1):
            if c == len(CHARS) - 1:
                pre_c = c
                continue
            if c != pre_c:
                decoded.append(c)
                confidences.append(probs[i, c])
                pre_c = c
        
        plate_text = ''.join([CHARS[idx] for idx in decoded if idx < len(CHARS) - 1])
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return plate_text, avg_confidence


def main():
    parser = argparse.ArgumentParser(description='Multi-View LPR Inference with MyModel')
    parser.add_argument('--model', required=True, help='path to model checkpoint')
    parser.add_argument('--images', nargs='+', required=True, help='paths to view images')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--img_size', default=[224, 224], nargs=2, type=int)
    parser.add_argument('--num_views', default=5, type=int)
    parser.add_argument('--fusion_type', default='average')
    parser.add_argument('--show_views', action='store_true', help='show individual view predictions')
    
    args = parser.parse_args()
    
    if len(args.images) != args.num_views:
        print(f"Error: Expected {args.num_views} images, got {len(args.images)}")
        return
    
    # 创建识别器
    recognizer = MyModelMultiViewRecognizer(
        model_path=args.model,
        device=args.device,
        img_size=tuple(args.img_size),
        num_views=args.num_views,
        fusion_type=args.fusion_type
    )
    
    # 预测
    print("\nProcessing images...")
    for i, img_path in enumerate(args.images):
        print(f"  View {i}: {img_path}")
    
    result = recognizer.predict(args.images)
    
    # 输出结果
    print("\n" + "="*60)
    print("Recognition Result:")
    print("="*60)
    print(f"Fused Plate:     {result['plate_text']}")
    print(f"Confidence:      {result['confidence']:.4f}")
    print(f"Has Plate Prob:  {result['has_plate_prob']:.4f}")
    
    if args.show_views:
        print("\nIndividual View Predictions:")
        print("-"*60)
        for i, view_pred in enumerate(result['view_predictions']):
            print(f"View {i}: {view_pred['text']:10s} "
                  f"(conf: {view_pred['confidence']:.3f}, "
                  f"disc: {view_pred['disc_prob']:.3f})")
    
    print("="*60)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) == 1:
        print("Usage:")
        print("\nBasic usage:")
        print("  python inference_mymodel_multiview.py \\")
        print("    --model weights_mymodel_multiview/best_model.pth \\")
        print("    --images view0.jpg view1.jpg view2.jpg view3.jpg view4.jpg")
        print("\nWith view details:")
        print("  python inference_mymodel_multiview.py \\")
        print("    --model weights_mymodel_multiview/best_model.pth \\")
        print("    --images view0.jpg view1.jpg view2.jpg view3.jpg view4.jpg \\")
        print("    --show_views")
    else:
        main()



