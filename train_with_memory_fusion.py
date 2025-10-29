"""
Training script for MyModel with Memory Fusion
使用MyModel的历史记忆融合功能进行多视图训练
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.my_model import MyModel
from data.multiview_data_loader import MultiViewLPRDataset, collate_fn_multiview
from data.load_data import CHARS
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
import argparse
import time
import os

import cv2
from PIL import Image, ImageDraw, ImageFont


def get_parser():
    parser = argparse.ArgumentParser(description='Training with Memory Fusion')
    parser.add_argument('--max_epoch', default=1, type=int)
    parser.add_argument('--img_size', default=[48, 128], help='the image size')
    parser.add_argument('--train_data', default='../CBLPRD-330k_v1/train.txt')
    parser.add_argument('--val_data', default='../CBLPRD-330k_v1/val.txt')
    parser.add_argument('--data_mode', default='txt', choices=['txt', 'directory'])
    parser.add_argument('--num_views', default=5, type=int, help='number of views per sample')
    parser.add_argument('--memory_weight', default=0.5, type=float, help='memory fusion weight')
    parser.add_argument('--fusion_strategy', default='sequential', 
                       choices=['sequential', 'average', 'ema'],
                       help='how to fuse multiple views')
    parser.add_argument('--ema_alpha', default=0.7, type=float, help='EMA alpha for memory update')
    parser.add_argument('--learning_rate', default=1e-2, type=float)
    parser.add_argument('--lpr_max_len', default=18, type=int)
    parser.add_argument('--train_batch_size', default=2048, type=int)
    parser.add_argument('--test_batch_size', default=1024, type=int)
    parser.add_argument('--num_workers', default=8, type=int)  # 增加workers加速数据加载
    parser.add_argument('--cache_images', default=False, type=bool, help='Cache images in memory for faster loading')
    parser.add_argument('--cuda', default=True, type=bool)
    parser.add_argument('--save_folder', default='./weights_memory_fusion/')
    parser.add_argument('--save_interval', default=5, type=int)
    
    # MyModel parameters
    parser.add_argument('--embed_dim', default=144, type=int)
    parser.add_argument('--depth', default=4, type=int)
    parser.add_argument('--num_heads', default=6, type=int)
    parser.add_argument('--decoder_depth', default=2, type=int)
    parser.add_argument('--show', default=False, type=bool, help='show test image and its predict result or not.')
    return parser.parse_args()


def fuse_multiview_with_memory(model, images, fusion_strategy='average', 
                                memory_weight=0.5, ema_alpha=0.7):
    """
    融合多视图图像使用记忆机制
    
    Args:
        model: MyModel实例
        images: [B, num_views, C, H, W] 多视图图像
        fusion_strategy: 融合策略
            - 'sequential': 顺序累积记忆
            - 'average': 所有视图特征平均
            - 'ema': 指数移动平均
        memory_weight: 记忆融合权重
        ema_alpha: EMA的alpha参数
        
    Returns:
        final_char_probs: [B, max_len, num_chars]
        final_disc_prob: [B, 1]
        all_features: list of features from each view
    """
    B, N, C, H, W = images.shape
    
    if fusion_strategy == 'sequential':
        # 顺序处理：每次迭代与历史记忆融合
        memory_feature = None
        all_features = []
        
        for view_idx in range(N):
            view_img = images[:, view_idx, :, :, :]
            char_probs, disc_prob, current_feature = model(
                view_img,
                memory_feature=memory_feature,
                memory_weight=memory_weight
            )
            all_features.append(current_feature)
            
            # 更新记忆为所有历史特征的平均值
            if len(all_features) > 1:
                memory_feature = torch.stack(all_features, dim=0).mean(dim=0)
            else:
                memory_feature = current_feature
        
        # 返回最后一次的预测（包含所有历史信息）
        return char_probs, disc_prob, all_features
    
    elif fusion_strategy == 'average':
        # 先获取所有视图的特征，然后平均
        all_features = []
        
        for view_idx in range(N):
            view_img = images[:, view_idx, :, :, :]
            _, _, current_feature = model(view_img)
            all_features.append(current_feature)
        
        # 平均所有特征
        avg_feature = torch.stack(all_features, dim=0).mean(dim=0)
        
        # 使用平均特征作为记忆，重新推理最后一张图像
        last_img = images[:, -1, :, :, :]
        char_probs, disc_prob, _ = model(
            last_img,
            memory_feature=avg_feature,
            memory_weight=1.0  # 完全使用平均特征
        )
        
        return char_probs, disc_prob, all_features
    
    elif fusion_strategy == 'ema':
        # 指数移动平均
        accumulated_feature = None
        all_features = []
        
        for view_idx in range(N):
            view_img = images[:, view_idx, :, :, :]
            _, _, current_feature = model(view_img)
            all_features.append(current_feature)
            
            if accumulated_feature is None:
                accumulated_feature = current_feature
            else:
                # EMA更新
                accumulated_feature = ema_alpha * accumulated_feature + (1 - ema_alpha) * current_feature
        
        # 使用累积特征推理最后一张图像
        last_img = images[:, -1, :, :, :]
        char_probs, disc_prob, _ = model(
            last_img,
            memory_feature=accumulated_feature,
            memory_weight=memory_weight
        )
        
        return char_probs, disc_prob, all_features
    
    else:
        raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")


def Greedy_Decode_Eval_Memory(model, dataset, args):
    """评估函数（使用记忆融合）"""
    epoch_size = len(dataset) // args.test_batch_size
    if epoch_size == 0:
        epoch_size = 1
        
    batch_iterator = iter(DataLoader(
        dataset,
        args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_multiview
    ))
    
    Tp = 0
    Tn_1 = 0
    Tn_2 = 0
    t1 = time.time()
    
    model.eval()
    
    with torch.no_grad():
        for i in range(epoch_size):
            try:
                images, labels, lengths = next(batch_iterator)
                
                # 解析标签
                start = 0
                targets = []
                for length in lengths:
                    label = labels[start:start+length].cpu().numpy()
                    targets.append(label)
                    start += length

                imgs = images.numpy().copy()
                
                if args.cuda:
                    images = Variable(images.cuda())
                else:
                    images = Variable(images)
                
                # 多视图融合
                char_probs, _, _ = fuse_multiview_with_memory(
                    model, images,
                    fusion_strategy=args.fusion_strategy,
                    memory_weight=args.memory_weight,
                    ema_alpha=args.ema_alpha
                )
                
                # 贪婪解码
                probs = char_probs.cpu().detach().numpy()
                preb_labels = []
                
                for j in range(probs.shape[0]):
                    preb = probs[j, :, :]
                    preb_label = [np.argmax(preb[k, :]) for k in range(preb.shape[0])]
                    
                    # 去重和去blank
                    no_repeat_blank_label = []
                    pre_c = preb_label[0]
                    if pre_c != len(CHARS) - 1:
                        no_repeat_blank_label.append(pre_c)
                    
                    for c in preb_label:
                        if (pre_c == c) or (c == len(CHARS) - 1):
                            if c == len(CHARS) - 1:
                                pre_c = c
                            continue
                        no_repeat_blank_label.append(c)
                        pre_c = c
                    
                    preb_labels.append(no_repeat_blank_label)
                    
                
                # 计算准确率
                for j, label in enumerate(preb_labels):
                    if j >= len(targets):
                        break
                    if args.show:
                        # imgs shape: [B, num_views, C, H, W]
                        # 取第一个视图显示: imgs[j, 0]
                        show(imgs[j, 0], label, targets[j])
                    if len(label) != len(targets[j]):
                        Tn_1 += 1
                        continue
                    if np.array_equal(targets[j], np.array(label)):
                        Tp += 1
                    else:
                        Tn_2 += 1
                        
            except StopIteration:
                break
            except Exception as e:
                print(f"Error: {e}")
                continue
    
    total = Tp + Tn_1 + Tn_2
    Acc = Tp * 1.0 / total if total > 0 else 0.0
    t2 = time.time()
    
    print(f"[Info] Test Accuracy: {Acc:.4f} [Correct:{Tp}, Wrong_Length:{Tn_1}, Wrong_Char:{Tn_2}, Total:{total}]")
    print(f"[Info] Test Speed: {(t2 - t1) / len(dataset):.4f}s per sample")
    
    return Acc


def train_with_memory(args):
    """训练主函数"""
    os.makedirs(args.save_folder, exist_ok=True)
    
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据
    print("Loading datasets...")
    train_dataset = MultiViewLPRDataset(
        args.train_data,
        img_size=tuple(args.img_size),
        lpr_max_len=args.lpr_max_len,
        num_views=args.num_views,
        mode=args.data_mode,
        cache_images=args.cache_images
    )
    
    val_dataset = MultiViewLPRDataset(
        args.val_data,
        img_size=tuple(args.img_size),
        lpr_max_len=args.lpr_max_len,
        num_views=args.num_views,
        mode=args.data_mode,
        cache_images=args.cache_images
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=collate_fn_multiview
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # 创建模型
    print("Creating MyModel...")
    num_chars = len(CHARS)
    
    model = MyModel(
        img_size=tuple(args.img_size),
        patch_size=16,
        in_c=3,
        num_classes=num_chars,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        max_len=args.lpr_max_len,
        num_chars=num_chars,
        decoder_depth=args.decoder_depth
    ).to(device)
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Fusion strategy: {args.fusion_strategy}")
    print(f"Memory weight: {args.memory_weight}")
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=2e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch)
    
    # 损失函数
    ctc_loss = nn.CTCLoss(blank=len(CHARS)-1, zero_infinity=True)
    disc_criterion = nn.BCELoss()
    
    best_acc = 0.0
    
    print("\nStarting training...")
    print("="*80)
    
    for epoch in range(args.max_epoch):
        model.train()
        total_loss = 0
        num_batches = 0
        batch_count = 0  # 计数器用于控制打印频率
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.max_epoch}")
        
        for images, labels, label_lengths in pbar:
            images = images.float().to(device)
            labels = labels.long().to(device)
            label_lengths = label_lengths.to(device)
            
            B, N, C, H, W = images.shape
            
            # 多视图融合
            char_probs, disc_prob, _ = fuse_multiview_with_memory(
                model, images,
                fusion_strategy=args.fusion_strategy,
                memory_weight=args.memory_weight,
                ema_alpha=args.ema_alpha
            )
            
            # CTC Loss 计算
            # char_probs: [B, max_len, num_chars] with softmax already applied
            # Convert to log probabilities for CTC
            char_log_probs = torch.log(char_probs + 1e-8)  # Add epsilon to avoid log(0)
            char_log_probs = char_log_probs.permute(1, 0, 2)  # [max_len, B, num_chars]
            
            # 每100个batch打印一次预测结果
            if batch_count % 100 == 0:
                with torch.no_grad():
                    # 取第一个样本进行解码显示
                    pred_indices = char_probs[0].argmax(dim=-1).cpu().numpy()  # [max_len]
                    # 去重和去blank
                    pred_chars = []
                    prev_idx = -1
                    for idx in pred_indices:
                        if idx != prev_idx and idx != len(CHARS) - 1:
                            pred_chars.append(CHARS[idx])
                        prev_idx = idx
                    pred_str = ''.join(pred_chars)
                    
                    # 获取真实标签
                    label_str = ''.join([CHARS[int(labels[i].item())] for i in range(label_lengths[0])])
                    
                    print(f"\n  [Batch {batch_count}] Pred: {pred_str:15s} | True: {label_str}")
            
            # Input lengths (all sequences are max_len)
            input_lengths = torch.full((B,), args.lpr_max_len, dtype=torch.long, device=device)
            
            # CTC loss expects labels as flat tensor
            char_loss = ctc_loss(char_log_probs, labels, input_lengths, label_lengths)
            
            # 判别器损失
            has_plate = torch.ones(B, dtype=torch.float, device=device)
            disc_loss = disc_criterion(disc_prob.squeeze(-1), has_plate)
            
            # 总损失
            loss = char_loss + 0.1 * disc_loss
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            batch_count += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\n[Epoch {epoch+1}/{args.max_epoch}] Train Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
        
        # 验证
        print("Evaluating...")
        val_acc = Greedy_Decode_Eval_Memory(model, val_dataset, args)
        
        scheduler.step()
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(args.save_folder, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': val_acc,
                'fusion_strategy': args.fusion_strategy,
                'memory_weight': args.memory_weight
            }, save_path)
            print(f"✓ Saved best model (Acc: {val_acc:.4f})")
        
        # 定期保存
        if (epoch + 1) % args.save_interval == 0:
            save_path = os.path.join(args.save_folder, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': val_acc
            }, save_path)
            print(f"✓ Saved checkpoint")
        
        print("="*80)
    
    print(f"\nTraining completed! Best accuracy: {best_acc:.4f}")

def show(img, label, target):
    # 检查并调整图像维度
    if len(img.shape) == 3:
        if img.shape[0] == 3:  # (C, H, W) -> (H, W, C)
            img = np.transpose(img, (1, 2, 0))
        # else: already (H, W, C)
    
    img *= 128.
    img += 127.5
    img = img.astype(np.uint8)

    lb = ""
    for i in label:
        lb += CHARS[i]
    tg = ""
    for j in target.tolist():
        tg += CHARS[int(j)]

    flag = "T" if lb == tg else "F"
    # img = cv2.putText(img, lb, (0,16), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (0, 0, 255), 1)
    img = cv2ImgAddText(img, lb, (0, 0))
    
    # 保存图片到文件而不是显示
    import os
    os.makedirs("./test_results", exist_ok=True)
    filename = f"./test_results/{flag}_{tg}_{lb}.jpg"
    cv2.imwrite(filename, img)
    print("target: ", tg, " ### {} ### ".format(flag), "predict: ", lb, f" [Saved: {filename}]")
    
    # 测试结果
    # exit()

def cv2ImgAddText(img, text, pos, textColor=(255, 0, 0), textSize=12):
    if (isinstance(img, np.ndarray)):  # detect opencv format or not
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype("data/NotoSansCJK-Regular.ttc", textSize, encoding="utf-8")
    draw.text(pos, text, textColor, font=fontText)

    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    args = get_parser()
    
    print("Configuration:")
    print("-" * 60)
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    print("-" * 60)
    
    train_with_memory(args)

