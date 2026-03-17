import os
import argparse
import numpy as np

import torch
import open_clip
import torch.nn as nn
from transformers import AutoTokenizer

from Dataset.Dataset import ImageTextDataset
from Transformer.model.encoder import transfomer_encoder
from module.CLIP import CLIP
from module.TextTransformer import TextTransformerForCLIP
from train import apply_lora_to_visual_encoder,resolve_image_size


def load_trained_clip_model(args, device, ckpt_path=None, strict=True):
    """
    Rebuild the same CLIP architecture used in training and load a saved state_dict.

    Notes:
    - args fields must match training-time model hyperparameters.
    - LoRA config (lora_r, lora_scale) must also match if LoRA was enabled.
    """
    if ckpt_path is None:
        ckpt_path = os.path.join(args.save_dir, "best_model.pth")

    visual_encoder = OpenCLIPVisualEncoder(
        model_name=args.model_name,
        pretrained=args.pretrained,
        device=device,
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

    lora_r = max(0, int(args.lora_r))
    lora_scale = float(args.lora_scale)
    if lora_r > 0:
        apply_lora_to_visual_encoder(model.image_encoder, r=lora_r, lora_scale=lora_scale)
        model.to(device)
    else:
        for p in model.image_encoder.parameters():
            p.requires_grad = False

    state_dict = torch.load(ckpt_path, map_location=device)
    load_msg = model.load_state_dict(state_dict, strict=strict)
    model.eval()
    return model, load_msg


def parse_args():
    """
        保持和train中一致的参数，方便加载模型和测试
    """
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
    parser.add_argument("--images_per_batch", type=int, default=15)
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
        default=4,
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

def count_parameters(model):
    """
    统计并打印模型的参数量。
    
    Args:
        model (nn.Module): PyTorch 模型
        
    Returns:
        tuple: (total_params, trainable_params) 单位：百万
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f} M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f} M)")
    
    return total_params / 1e6, trainable_params / 1e6


def count_parameters_per_module(model):
    """打印模型中每个子模块的参数量"""
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"{name}: {params:,} parameters ({params/1e6:.2f} M)")


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


def origin_model(args, device):
    visual_encoder = OpenCLIPVisualEncoder(
        model_name=args.model_name,
        pretrained=args.pretrained,
        device=device,
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

    return model