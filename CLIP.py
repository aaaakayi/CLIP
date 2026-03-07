import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CLIP(nn.Module):
    def __init__(self, image_encoder, text_encoder, embed_dim=512):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

        image_dim = image_encoder.embed_dim
        text_dim = text_encoder.encoder.d_model

        # 投影层，将特征映射到共享的对比空间
        self.image_proj = nn.Linear(image_dim, embed_dim)
        self.text_proj = nn.Linear(text_dim, embed_dim)

        # 可学习的温度参数，初始化为 log(1/0.07)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, images, input_ids, attention_mask=None):
        # 图像特征 [B, image_dim]
        image_feat = self.image_encoder(images)
        # 文本特征 [B, text_dim]
        text_feat = self.text_encoder(input_ids, attention_mask)

        # 投影并 L2 归一化
        image_feat = F.normalize(self.image_proj(image_feat), dim=-1)
        text_feat = F.normalize(self.text_proj(text_feat), dim=-1)

        return image_feat, text_feat

def clip_loss(image_feat, text_feat, logit_scale):
    # 计算相似度矩阵 (batch_size, batch_size)
    logits = logit_scale * image_feat @ text_feat.T
    labels = torch.arange(len(logits), device=logits.device)
    loss_i = F.cross_entropy(logits, labels)      # image -> text
    loss_t = F.cross_entropy(logits.T, labels)    # text -> image
    return (loss_i + loss_t) / 2