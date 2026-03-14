import torch
import open_clip
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torch.optim as optim
from tqdm import tqdm

class ImageTextDataset(Dataset):
    def __init__(self, image_dir, text_dir, transform, tokenizer):
        self.image_dir = image_dir
        self.text_dir = text_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.captions_per_image = 5 # 每张图像对应的文本数量，根据实际情况调整

        image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        self.base_names = [os.path.splitext(f)[0] for f in image_files]

        self.num_images = len(self.base_names)
        self.total_samples = self.num_images * self.captions_per_image

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        img_idx = idx // self.captions_per_image
        cap_idx = idx % self.captions_per_image
        base = self.base_names[img_idx]

        img_path = os.path.join(self.image_dir, base + '.jpg')
        if not os.path.exists(img_path):
            img_path = os.path.join(self.image_dir, base + '.png')
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        txt_path = os.path.join(self.text_dir, base + '.txt')
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        text = lines[cap_idx] if cap_idx < len(lines) else ""

        tokens = self.tokenizer(text)
        return image, tokens[0]


def parse_args():
    parser = argparse.ArgumentParser(description="Train use open clip")
    parser.add_argument("--train_h5", type=str, default="./Data/test.h5")
    #parser.add_argument("--test_h5", type=str, default="./Data/test.h5")
    parser.add_argument("--save_dir", type=str, default="./result_openclip_full")
    parser.add_argument("--model_name", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="./clip_weights/open_clip_pytorch_model.bin")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--captions_per_image", type=int, default=5)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def train():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载模型和预处理
    model, _, preprocess_train = open_clip.create_model_and_transforms(
        args.model_name,
        pretrained=args.pretrained,
        device=device
    )
    tokenizer = open_clip.get_tokenizer(args.model_name)

    # 创建 Dataset 和 DataLoader
    train_dataset = ImageTextDataset(
        image_dir=args.image_dir,
        text_dir=args.text_dir,
        transform=preprocess_train,
        tokenizer=tokenizer
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 定义损失函数和优化器
    criterion = open_clip.loss.ClipLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 训练循环
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for images, texts in pbar:
            images = images.to(device)
            texts = texts.to(device)

            image_features, text_features, logit_scale = model(images, texts)
            loss = criterion(image_features, text_features, logit_scale)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

    # 保存模型
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "finetuned_clip.pth")
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存至 {save_path}")


if __name__ == "__main__":
    train()