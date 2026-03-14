import argparse
import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import open_clip
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoTokenizer
from torchvision import transforms

from Dataset.Dataset import ImageTextDataset
from Transformer.model.encoder import transfomer_encoder
from module.CLIP import CLIP
from module.TextTransformer import TextTransformerForCLIP


OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CLIP with open_clip visual encoder and custom text encoder"
    )
    parser.add_argument("--train_h5", type=str, default="./Data/train.h5")
    parser.add_argument("--test_h5", type=str, default="./Data/test.h5")
    parser.add_argument("--tokenizer_dir", type=str, default="./Bert_tokenizer")
    parser.add_argument(
        "--pretrained",
        type=str,
        default="./clip_weights/open_clip_pytorch_model.bin",
    )
    parser.add_argument("--model_name", type=str, default="ViT-B-32")
    parser.add_argument("--save_dir", type=str, default="./result")
    parser.add_argument("--images_per_batch", type=int, default=8)
    parser.add_argument("--captions_per_image", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--text_width", type=int, default=768)
    parser.add_argument("--text_layers", type=int, default=12)
    parser.add_argument("--text_heads", type=int, default=12)
    parser.add_argument("--text_dropout", type=float, default=0.1)
    parser.add_argument(
        "--accum_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps. Effective batch size = images_per_batch * captions_per_image * accum_steps.",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank for vision encoder. 0 disables LoRA (vision encoder fully frozen).",
    )
    parser.add_argument(
        "--lora_scale",
        type=float,
        default=1.0,
        help="Scaling factor for LoRA output (often 1.0 or 2.0).",
    )
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class LoRALinear(nn.Module):
    """Wraps nn.Linear with Low-Rank Adaptation. Original layer is frozen."""

    def __init__(self, linear: nn.Linear, r: int, lora_scale: float = 1.0):
        super().__init__()
        self.linear = linear
        self.linear.requires_grad_(False)
        in_f, out_f = linear.in_features, linear.out_features
        self.lora_A = nn.Parameter(torch.zeros(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_scale = lora_scale
        self.r = r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        out = out + self.lora_scale * (x @ self.lora_A.T @ self.lora_B.T)
        return out


def apply_lora_to_visual_encoder(visual_encoder: nn.Module, r: int, lora_scale: float = 1.0) -> int:
    """
    Freeze the open_clip visual encoder and inject LoRA into transformer blocks.
    Expects visual_encoder.visual to have .transformer.resblocks (VisionTransformer) or .resblocks.
    Replaces nn.Linear in attn.out_proj and mlp.c_fc, mlp.c_proj.
    Returns the number of LoRA-injected layers.
    """
    visual = getattr(visual_encoder, "visual", visual_encoder)
    visual.requires_grad_(False)
    transformer = getattr(visual, "transformer", visual)
    resblocks = getattr(transformer, "resblocks", None)
    if resblocks is None:
        return 0
    count = 0
    for block in resblocks:
        if hasattr(block, "attn") and hasattr(block.attn, "out_proj") and isinstance(block.attn.out_proj, nn.Linear):
            block.attn.out_proj = LoRALinear(block.attn.out_proj, r, lora_scale)
            count += 1
        if hasattr(block, "mlp"):
            if hasattr(block.mlp, "c_fc") and isinstance(block.mlp.c_fc, nn.Linear):
                block.mlp.c_fc = LoRALinear(block.mlp.c_fc, r, lora_scale)
                count += 1
            if hasattr(block.mlp, "c_proj") and isinstance(block.mlp.c_proj, nn.Linear):
                block.mlp.c_proj = LoRALinear(block.mlp.c_proj, r, lora_scale)
                count += 1
    return count


def resolve_image_size(image_size):
    if isinstance(image_size, tuple):
        if len(image_size) == 2 and image_size[0] == image_size[1]:
            return int(image_size[0])
        raise ValueError(f"Unsupported image_size: {image_size}")
    return int(image_size)


def build_transforms(image_size):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.CenterCrop(image_size),
        transforms.Normalize(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
    ])
    return train_transform, eval_transform


def split_train_val_indices_by_image(dataset_len, captions_per_image=5, val_ratio=0.1, seed=42):
    if dataset_len % captions_per_image != 0:
        raise ValueError("Dataset size must be divisible by captions_per_image")

    num_images = dataset_len // captions_per_image
    image_ids = list(range(num_images))
    rng = random.Random(seed)
    rng.shuffle(image_ids)

    val_count = max(1, int(num_images * val_ratio))
    val_image_ids = set(image_ids[:val_count])

    train_indices = []
    val_indices = []
    for image_id in image_ids:
        start = image_id * captions_per_image
        target = val_indices if image_id in val_image_ids else train_indices
        target.extend(range(start, start + captions_per_image))

    return train_indices, val_indices


class GroupedBatchSampler:
    def __init__(self, dataset_len, images_per_batch, captions_per_image=5, shuffle=True, drop_last=True):
        if dataset_len % captions_per_image != 0:
            raise ValueError("dataset_len must be divisible by captions_per_image")
        self.dataset_len = dataset_len
        self.images_per_batch = images_per_batch
        self.captions_per_image = captions_per_image
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_groups = dataset_len // captions_per_image

    def __iter__(self):
        group_ids = list(range(self.num_groups))
        if self.shuffle:
            random.shuffle(group_ids)

        current_groups = []
        for group_id in group_ids:
            current_groups.append(group_id)
            if len(current_groups) == self.images_per_batch:
                yield self._expand_groups(current_groups)
                current_groups = []

        if current_groups and not self.drop_last:
            yield self._expand_groups(current_groups)

    def __len__(self):
        if self.drop_last:
            return self.num_groups // self.images_per_batch
        return math.ceil(self.num_groups / self.images_per_batch)

    def _expand_groups(self, group_ids):
        batch_indices = []
        for group_id in group_ids:
            start = group_id * self.captions_per_image
            batch_indices.extend(range(start, start + self.captions_per_image))
        return batch_indices


class OpenCLIPVisualEncoder(nn.Module):
    def __init__(self, model_name, pretrained, device):
        super().__init__()
        if not os.path.exists(pretrained):
            raise FileNotFoundError(f"open_clip checkpoint not found: {pretrained}")

        clip_model, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=device,
        )
        self.visual = clip_model.visual
        self.embed_dim = getattr(self.visual, "output_dim", None)
        if self.embed_dim is None:
            text_projection = getattr(clip_model, "text_projection", None)
            if text_projection is None:
                raise AttributeError("Cannot infer visual output dimension from open_clip model")
            self.embed_dim = text_projection.shape[-1]
        self.image_size = resolve_image_size(getattr(self.visual, "image_size", 224))

    def forward(self, images):
        return self.visual(images)


def build_group_ids(batch_size, captions_per_image, device):
    if batch_size % captions_per_image != 0:
        raise ValueError("Batch size must be divisible by captions_per_image")
    num_groups = batch_size // captions_per_image
    return torch.arange(num_groups, device=device).repeat_interleave(captions_per_image)


def siglip_loss_with_group(image_feat, text_feat, group_ids, logit_scale):
    """
    image_feat: (N, D) 图像特征（已归一化）
    text_feat: (N, D) 文本特征（已归一化）
    group_ids: (N,) 每个样本所属的图片组 ID（相同 ID 表示同一图片的不同描述，它们互为正样本）
    logit_scale: 可学习的温度参数（标量）
    """
    scale = logit_scale.exp().clamp(max=100.0)  # 缩放因子
    logits = scale * (image_feat @ text_feat.T)   # (N, N)

    # 构造标签矩阵：正样本位置为 1，负样本为 0
    labels = (group_ids.unsqueeze(1) == group_ids.unsqueeze(0)).float()

    # 二元交叉熵损失（平均所有位置）
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    return loss

@torch.no_grad()
def evaluate_loss(model, data_loader, device, captions_per_image, amp_enabled):
    model.eval()
    total_loss = 0.0

    for images, input_ids, attention_mask in data_loader:
        images = images.to(device, non_blocking=True)
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        group_ids = build_group_ids(images.size(0), captions_per_image, device)

        with autocast(enabled=amp_enabled):
            image_feat, text_feat = model(images, input_ids, attention_mask)
            loss = siglip_loss_with_group(
                image_feat=image_feat,
                text_feat=text_feat,
                group_ids=group_ids,
                logit_scale=model.logit_scale,
            )
        total_loss += loss.item()

    return total_loss / max(len(data_loader), 1)


def save_loss_curve(train_losses, val_losses, save_dir):
    plt.figure(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, marker="o", linestyle="-", label="train")
    plt.plot(epochs, val_losses, marker="s", linestyle="-", label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.close()


def main():
    args = parse_args()
    seed_everything(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    print(f"device: {device}")

    visual_encoder = OpenCLIPVisualEncoder(
        model_name=args.model_name,
        pretrained=args.pretrained,
        device=device,
    )
    train_transform, eval_transform = build_transforms(visual_encoder.image_size)

    train_base_dataset = ImageTextDataset(args.train_h5, transform=train_transform)
    dataset_len = len(train_base_dataset)
    train_indices, val_indices = split_train_val_indices_by_image(
        dataset_len,
        captions_per_image=args.captions_per_image,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    val_base_dataset = ImageTextDataset(args.train_h5, transform=eval_transform)
    test_base_dataset = ImageTextDataset(args.test_h5, transform=eval_transform)

    train_dataset = Subset(train_base_dataset, train_indices)
    val_dataset = Subset(val_base_dataset, val_indices)
    test_dataset = Subset(test_base_dataset, list(range(len(test_base_dataset))))

    train_sampler = GroupedBatchSampler(
        dataset_len=len(train_dataset),
        images_per_batch=args.images_per_batch,
        captions_per_image=args.captions_per_image,
        shuffle=True,
        drop_last=True,
    )
    val_sampler = GroupedBatchSampler(
        dataset_len=len(val_dataset),
        images_per_batch=args.images_per_batch,
        captions_per_image=args.captions_per_image,
        shuffle=False,
        drop_last=False,
    )
    test_sampler = GroupedBatchSampler(
        dataset_len=len(test_dataset),
        images_per_batch=args.images_per_batch,
        captions_per_image=args.captions_per_image,
        shuffle=False,
        drop_last=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=amp_enabled,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=amp_enabled,
        persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=test_sampler,
        num_workers=args.num_workers,
        pin_memory=amp_enabled,
        persistent_workers=args.num_workers > 0,
    )

    if len(train_loader) == 0:
        raise ValueError(
            "train_loader is empty. Reduce --images_per_batch or use a larger training set."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, local_files_only=True)
    encoder = transfomer_encoder(
        num_layers=args.text_layers,
        vocab_size=tokenizer.vocab_size,
        num_heads=args.text_heads,
        dropout=args.text_dropout,
        d_model=args.text_width,
    )
    cls_token_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else 101
    text_encoder = TextTransformerForCLIP(encoder, cls_token_id=cls_token_id)

    model = CLIP(
        image_encoder=visual_encoder,
        text_encoder=text_encoder,
        embed_dim=args.embed_dim,
    ).to(device)

    nn.init.normal_(model.image_proj.weight, std=0.02)
    nn.init.zeros_(model.image_proj.bias)
    nn.init.normal_(model.text_proj.weight, std=0.02)
    nn.init.zeros_(model.text_proj.bias)
    nn.init.constant_(model.logit_scale, np.log(1 / 0.07))

    lora_r = max(0, int(args.lora_r))
    lora_scale = float(args.lora_scale)
    if lora_r > 0:
        num_lora = apply_lora_to_visual_encoder(
            model.image_encoder, r=lora_r, lora_scale=lora_scale
        )
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        print(
            f"ViT frozen; LoRA applied (r={lora_r}, scale={lora_scale}): {num_lora} layers. "
            f"Trainable params: {n_trainable:,} / {n_total:,}"
        )
    else:
        for p in model.image_encoder.parameters():
            p.requires_grad = False
        print("ViT fully frozen (no LoRA).")

    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=amp_enabled)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    accum_steps = max(1, int(args.accum_steps))
    effective_batch = args.images_per_batch * args.captions_per_image * accum_steps
    print(
        f"Gradient accumulation: {accum_steps} step(s). "
        f"Effective batch size (samples): {effective_batch} "
        f"(= {args.images_per_batch} images × {args.captions_per_image} captions × {accum_steps} accum)."
    )

    epoch_bar = tqdm(range(args.epochs), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        total_loss = 0.0
        num_batches = 0
        accum_counter = 0

        for images, input_ids, attention_mask in train_loader:
            images = images.to(device, non_blocking=True)
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            group_ids = build_group_ids(images.size(0), args.captions_per_image, device)

            if accum_counter == 0:
                optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=amp_enabled):
                image_feat, text_feat = model(images, input_ids, attention_mask)
                loss = siglip_loss_with_group(
                    image_feat=image_feat,
                    text_feat=text_feat,
                    group_ids=group_ids,
                    logit_scale=model.logit_scale,
                )
                scaled_loss = loss / accum_steps

            scaler.scale(scaled_loss).backward()
            accum_counter += 1
            total_loss += loss.item()
            num_batches += 1

            if accum_counter == accum_steps:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                with torch.no_grad():
                    model.logit_scale.clamp_(max=np.log(100.0))
                accum_counter = 0

        if accum_counter > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                model.logit_scale.clamp_(max=np.log(100.0))

        scheduler.step()

        train_loss = total_loss / max(num_batches, 1)
        val_loss = evaluate_loss(
            model=model,
            data_loader=val_loader,
            device=device,
            captions_per_image=args.captions_per_image,
            amp_enabled=amp_enabled,
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        logit_scale_value = model.logit_scale.exp().item()
        epoch_bar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            val_loss=f"{val_loss:.4f}",
            lr=f"{current_lr:.2e}",
            scale=f"{logit_scale_value:.3f}",
        )
        print(
            f"Epoch {epoch + 1}/{args.epochs}, "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"LR: {current_lr:.2e}, "
            f"logit_scale(exp): {logit_scale_value:.3f}"
        )

        latest_path = os.path.join(args.save_dir, "latest_model.pth")
        torch.save(model.state_dict(), latest_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save(model.state_dict(), best_path)

    save_loss_curve(train_losses, val_losses, args.save_dir)

    test_loss = evaluate_loss(
        model=model,
        data_loader=test_loader,
        device=device,
        captions_per_image=args.captions_per_image,
        amp_enabled=amp_enabled,
    )
    print(f"Final Test Loss: {test_loss:.4f}")


if __name__ == "__main__":
    main()