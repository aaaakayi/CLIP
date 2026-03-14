import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from transformers import AutoTokenizer

from TextTransformer import TextTransformerForCLIP
from Transformer.model.encoder import transfomer_encoder
from ViT import ViT, Transformer


class NoiseMultimodalDataset(Dataset):
    """Synthetic multimodal noise dataset for sanity checks."""

    def __init__(self, size, seq_len, vocab_size, image_size=224, seed=42):
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.image_size = image_size
        self.seed = seed

        # Pre-generate binary labels so train/val splits are deterministic.
        rng = np.random.default_rng(seed)
        self.labels = rng.integers(0, 2, size=size, dtype=np.int64)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Deterministic per-sample noise.
        g = torch.Generator().manual_seed(self.seed + idx)

        image = torch.randn(3, self.image_size, self.image_size, generator=g)

        # Random sequence length in [8, seq_len], then pad to seq_len.
        cur_len = int(torch.randint(8, self.seq_len + 1, (1,), generator=g).item())
        input_ids = torch.randint(999, self.vocab_size, (cur_len,), generator=g, dtype=torch.long)
        input_ids = torch.cat([input_ids, torch.zeros(self.seq_len - cur_len, dtype=torch.long)], dim=0)

        attention_mask = torch.zeros(self.seq_len, dtype=torch.long)
        attention_mask[:cur_len] = 1

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, input_ids, attention_mask, label


class CLIPBinaryClassifier(nn.Module):
    def __init__(self, image_encoder, text_encoder, hidden_dim=512):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

        image_dim = image_encoder.embed_dim
        text_dim = text_encoder.encoder.d_model

        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, images, input_ids, attention_mask):
        image_feat = self.image_encoder(images)
        text_feat = self.text_encoder(input_ids, attention_mask)

        image_feat = nn.functional.normalize(self.image_proj(image_feat), dim=-1)
        text_feat = nn.functional.normalize(self.text_proj(text_feat), dim=-1)

        x = torch.cat([image_feat, text_feat], dim=-1)
        return self.classifier(x)


def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, input_ids, attention_mask, labels in loader:
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            logits = model(images, input_ids, attention_mask)
            loss = criterion(logits, labels)

            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()

    return total_loss / len(loader), correct / total


def main():
    parser = argparse.ArgumentParser(description="Noise binary sanity check (expected ~50% val acc)")
    parser.add_argument("--size", type=int, default=22500)
    parser.add_argument("--seq_len", type=int, default=77)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("./Bert_tokenizer")
    vocab_size = tokenizer.vocab_size

    dataset = NoiseMultimodalDataset(
        size=args.size,
        seq_len=args.seq_len,
        vocab_size=vocab_size,
        seed=args.seed,
    )

    val_size = int(args.size * args.val_ratio)
    train_size = args.size - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    vit_transformer = Transformer(d_model=768, num_heads=6, num_layers=6)
    image_encoder = ViT(
        patch=16,
        channels=3,
        embed_dim=768,
        token_size=(224 // 16) ** 2,
        img_size=224,
        transformer=vit_transformer,
    )

    text_backbone = transfomer_encoder(
        num_layers=6,
        vocab_size=vocab_size,
        num_heads=6,
        dropout=0.1,
        d_model=768,
    )
    text_encoder = TextTransformerForCLIP(text_backbone, cls_token_id=101)

    model = CLIPBinaryClassifier(image_encoder, text_encoder, hidden_dim=512).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for images, input_ids, attention_mask, labels in pbar:
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            logits = model(images, input_ids, attention_mask)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(train_loss=f"{loss.item():.4f}")

        train_loss = total_loss / len(train_loader)
        val_loss, val_acc = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc * 100:.2f}%"
        )

    print("\nSanity target: validation accuracy should stay around 50% on pure noise.")


if __name__ == "__main__":
    main()
