import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import CTCLoss
from model.my_model import MyModel
from data.load_data import LPRDataLoader, CHARS
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
import argparse
import time
import cv2
from PIL import Image, ImageDraw, ImageFont

def collate_fn(batch):
    imgs = []
    labels = []
    lengths = []
    for _, sample in enumerate(batch):
        img, label, length = sample
        imgs.append(torch.from_numpy(img))
        labels.extend(label)
        lengths.append(length)
    labels = np.asarray(labels).flatten().astype(np.float32)

    return (torch.stack(imgs, 0), torch.from_numpy(labels), torch.tensor(lengths, dtype=torch.long))


def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--max_epoch', default=15, help='epoch to train the network')
    parser.add_argument('--img_size', default=[94, 24], help='the image size')
    parser.add_argument('--train_img_dirs', default="../CBLPRD-330k_v1/train.txt", help='the train images path')
    parser.add_argument('--test_img_dirs', default="../CBLPRD-330k_v1/val.txt", help='the test images path')
    parser.add_argument('--dropout_rate', default=0.5, help='dropout rate.')
    parser.add_argument('--learning_rate', default=0.01, help='base value of learning rate.')
    parser.add_argument('--lpr_max_len', default=18, help='license plate number max length.')
    parser.add_argument('--train_batch_size', default=1024, help='training batch size.')
    parser.add_argument('--test_batch_size', default=512, help='testing batch size.')
    parser.add_argument('--phase_train', default=True, type=bool, help='train or test phase flag.')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
    parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
    parser.add_argument('--save_interval', default=2000, type=int, help='interval for save model state dict')
    parser.add_argument('--test_interval', default=2000, type=int, help='interval for evaluate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=2e-5, type=float, help='Weight decay for SGD')
    parser.add_argument('--lr_schedule', default=[4, 8, 12, 14, 16], help='schedule for learning rate.')
    parser.add_argument('--save_folder', default='./weights/', help='Location to save checkpoint models')
    # parser.add_argument('--pretrained_model', default='./weights/Final_LPRNet_model.pth', help='pretrained base model')
    parser.add_argument('--pretrained_model', default='', help='test/LPRNet_Pytorch/weights/Final_LPRNet_model.pth')
    parser.add_argument('--show', default=False, type=bool, help='show test image and its predict result or not.')

    args = parser.parse_args()

    return args

def train(
        train_txt,
        val_txt,
        img_size=(48, 128),
        lpr_max_len=18,
        batch_size=256,
        epochs=100,
        lr=1e-2,
        device='cuda'
):
    argus = get_parser()
    # 数据集
    train_dataset = LPRDataLoader(train_txt, img_size, lpr_max_len)
    val_dataset = LPRDataLoader(val_txt, img_size, lpr_max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)

    num_classes = len(CHARS)

    model = MyModel(
        img_size=(img_size[1], img_size[0]),
        patch_size=(8, 8),
        in_c=3,
        num_classes=lpr_max_len * num_classes,  # Keep for encoder compatibility
        embed_dim=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        max_len=lpr_max_len,
        num_chars=num_classes,
        decoder_depth=2,
        disc_hidden=256
    ).to(device)

    # 损失函数和优化器
    ctc_loss = CTCLoss(blank=len(CHARS)-1, zero_infinity=True)
    bce_loss = nn.BCELoss()  # For discriminator
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_ctc_loss = 0
        total_disc_loss = 0
        
        for images, labels, label_lengths in tqdm(train_loader, desc="Training", unit="batch"):
            images = torch.tensor(images, dtype=torch.float32).to(device)
            labels = torch.tensor(labels, dtype=torch.long).to(device)

            optimizer.zero_grad()
            
            # Forward pass - returns (char_probs, disc_prob, current_feature)
            char_probs, disc_prob, _ = model(images)
            
            # CTC Loss for character recognition
            batch_size = char_probs.size(0)
            # char_probs: [B, max_len, num_chars]
            # Convert to log probabilities for CTC
            char_logits = torch.log(char_probs + 1e-8)  # Add epsilon to avoid log(0)
            char_logits = char_logits.permute(1, 0, 2)  # [max_len, B, num_chars]

            input_lengths = torch.full((batch_size,), lpr_max_len, dtype=torch.long)
            loss_ctc = ctc_loss(char_logits, labels, input_lengths, label_lengths)
            
            # Discriminator Loss (假设所有真实样本标签为1)
            disc_targets = torch.ones_like(disc_prob)
            loss_disc = bce_loss(disc_prob, disc_targets)
            
            # 总损失 (可以调整权重)
            loss = loss_ctc + 0.1 * loss_disc
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_ctc_loss += loss_ctc.item()
            total_disc_loss += loss_disc.item()

        avg_loss = total_loss / len(train_loader)
        avg_ctc = total_ctc_loss / len(train_loader)
        avg_disc = total_disc_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f} (CTC: {avg_ctc:.4f}, Disc: {avg_disc:.4f})")

        # 验证
        model.eval()
        print("Final test Accuracy:")
        Greedy_Decode_Eval(model, val_dataset, argus)
        # with torch.no_grad():
        #     for images, labels, label_lengths in val_dataloader:
        #         images = images.to(device)
        #         labels = labels.to(device)
        #         outputs = model(images)
        #         batch_size = outputs.size(0)
        #         max_label_len = labels.shape[1]
        #         outputs = outputs.view(batch_size, -1, num_classes)
        #         preds = outputs[:, :max_label_len, :].argmax(dim=2)
        #         correct += (preds == labels).all(dim=1).sum().item()
        #         total += labels.size(0)
        # print(f"Val Acc: {correct/total:.4f}")

def Greedy_Decode_Eval(Net, datasets, args):
    epoch_size = len(datasets) // args.test_batch_size
    batch_iterator = iter(DataLoader(datasets, args.test_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn))

    Tp = 0
    Tn_1 = 0
    Tn_2 = 0
    t1 = time.time()
    
    for i in range(epoch_size):
        # load train data
        images, labels, lengths = next(batch_iterator)
        start = 0
        targets = []
        for length in lengths:
            label = labels[start:start+length].cpu().numpy()  # Convert to numpy array here
            targets.append(label)
            start += length

        imgs = images.numpy().copy()

        if args.cuda:
            images = Variable(images.cuda())
        else:
            images = Variable(images)

        # forward
        # MyModel returns (char_probs, disc_prob, current_feature)
        char_probs, _, _ = Net(images)
        
        # char_probs is already [B, max_len, num_chars] with softmax applied
        prebs = char_probs.view(args.test_batch_size, args.lpr_max_len, len(CHARS))
        
        # greedy decode
        prebs = prebs.cpu().detach().numpy()
        preb_labels = list()
        
        for i in range(prebs.shape[0]):
            preb = prebs[i, :, :]
            preb_label = list()
            preb_label = [np.argmax(preb[j, :]) for j in range(preb.shape[0])]
            no_repeat_blank_label = list()
            pre_c = preb_label[0]
            if pre_c != len(CHARS) - 1:
                no_repeat_blank_label.append(pre_c)
            for c in preb_label: # dropout repeate label and blank label
                if (pre_c == c) or (c == len(CHARS) - 1):
                    if c == len(CHARS) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c
            preb_labels.append(no_repeat_blank_label)
            
        for i, label in enumerate(preb_labels):
            if args.show:
                show(imgs[i], label, targets[i])
            if len(label) != len(targets[i]):
                Tn_1 += 1
                continue
            if np.array_equal(targets[i], np.array(label)):  # Use np.array_equal for comparison
                Tp += 1
            else:
                Tn_2 += 1

    Acc = Tp * 1.0 / (Tp + Tn_1 + Tn_2)
    print("[Info] Test Accuracy: {} [{}:{}:{}:{}]".format(Acc, Tp, Tn_1, Tn_2, (Tp+Tn_1+Tn_2)))
    t2 = time.time()
    print("[Info] Test Speed: {}s 1/{}]".format((t2 - t1) / len(datasets), len(datasets)))



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
    
    try:
        # img = cv2.putText(img, lb, (0,16), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (0, 0, 255), 1)
        img = cv2ImgAddText(img, lb, (0, 0))
        
        # 保存图片到文件而不是显示
        import os
        os.makedirs("./test_results", exist_ok=True)
        filename = f"./test_results/{flag}_{tg}_{lb}.jpg"
        cv2.imwrite(filename, img)
        print("target: ", tg, " ### {} ### ".format(flag), "predict: ", lb, f" [Saved: {filename}]")
    except Exception as e:
        print(f"Error saving image: {e}, img.shape={img.shape}")
    
    # 测试结果
    exit()

def cv2ImgAddText(img, text, pos, textColor=(255, 0, 0), textSize=12):
    if (isinstance(img, np.ndarray)):  # detect opencv format or not
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype("data/NotoSansCJK-Regular.ttc", textSize, encoding="utf-8")
    draw.text(pos, text, textColor, font=fontText)

    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

if __name__ == "__main__":
    train_txt = "/home/pernod/CBLPRD-330k_v1/train.txt"
    val_txt = "/home/pernod/CBLPRD-330k_v1/val.txt"
    train(train_txt, val_txt)
