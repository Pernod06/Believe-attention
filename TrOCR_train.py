import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer, VisionEncoderDecoderModel, ViTFeatureExtractor, ViTModel
from tqdm import tqdm
from data.load_data import LPRDataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "microsoft/trocr-base-handwritten"

feature_extractor = ViTFeatureExtractor.from_pretrained("microsoft/trocr-base-handwritten")
tokenizer = AutoTokenizer.from_pretrained("microsoft/trocr-base-handwritten")

model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # TrOCR 的标准化参数
])

def parse_args():
    parser = argparse.ArgumentParser(description="TrOCR 车牌识别训练脚本")
    parser.add_argument("--train_txt", type=str, required=True, help="训练集文件路径")
    parser.add_argument("--val_txt", type=str, required=True, help="验证集文件路径")
    parser.add_argument("--root_dir", type=str, required=True, help="图像根目录")
    parser.add_argument("--output_dir", type=str, default="trocr_output", help="模型保存路径")
    parser.add_argument("--batch_size", type=int, default=16, help="批量大小")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--lr", type=float, default=3e-4, help="学习率")
    return parser.parse_args()

def collate_fn(batch):
    images = [item["pixel_values"] for item in batch]
    labels = [item["labels"] for item in batch]
    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    return {"pixel_values": pixel_values, "labels": labels}

def create_dataloader(txt_file, root_dir, transform, batch_size):
    dataset = LPRDataLoader(
        txt_file=txt_file,
        root_dir=root_dir,
        transform=transform
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )


def train_one_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(data_loader, desc="Training"):
        images = batch["pixel_values"].to(device)
        labels = tokenizer(batch["labels"], return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
        # vision_embeds 可以作为 TrOCRForCausalLM 的 inputs_embeds
        outputs = model(pixel_values=images, labels=labels)

        loss = outputs.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
    return total_loss / len(data_loader)


def validate(model, data_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validation"):
            images = batch["pixel_values"].to(device)
            labels = tokenizer(batch["labels"], return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
            # vision_embeds 可以作为 TrOCRForCausalLM 的 inputs_embeds
            outputs = model(pixel_values=images, labels=labels)

            loss = outputs.loss
            total_loss += loss.item()
    return total_loss / len(data_loader)


def main():
    args = parse_args()

    # 创建数据集和 DataLoader
    train_loader = create_dataloader(args.train_txt, args.root_dir, image_transform, args.batch_size)
    val_loader = create_dataloader(args.val_txt, args.root_dir, image_transform, args.batch_size)

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # 输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(os.path.join(args.output_dir, "best_model"))
            tokenizer.save_pretrained(os.path.join(args.output_dir, "best_model"))
            print("Saved best model.")

    # 保存最终模型
    model.save_pretrained(os.path.join(args.output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final_model"))
    print("Training complete.")

if __name__ == "__main__":
    main()