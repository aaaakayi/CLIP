import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from Dataset import ImageTextDataset
from ViT import ViT,Transformer
from TextTransformer import TextTransformerForCLIP
from Transformer.model.encoder import transfomer_encoder
from CLIP import CLIP
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

def clip_loss(image_feat, text_feat, logit_scale):
    """
    image_feat, text_feat: 已经 L2 归一化的特征，形状 [batch, embed_dim]
    logit_scale: 可学习的温度参数（标量）
    """
    # 计算相似度矩阵 (batch, batch)
    logits = logit_scale * image_feat @ text_feat.T
    labels = torch.arange(len(logits), device=logits.device)
    loss_i = F.cross_entropy(logits, labels)      # image -> text
    loss_t = F.cross_entropy(logits.T, labels)    # text -> image
    return (loss_i + loss_t) / 2


def main():
    # 超参数
    batch_size = 32
    epochs = 1
    lr = 5e-4
    embed_dim = 512

    if torch.cuda.is_available():
        device = 'cuda'
        print('cuda可用，使用cuda')
    else:
        device = 'cpu'
        print('cuda不可用，使用cpu')

    # h5_path
    h5_path = './Data/h5_3.h5'

    # 创建 Dataset 实例（只传入路径）
    train_dataset = ImageTextDataset(h5_path)

    # 创建 DataLoader，设置 num_workers>0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # ViT
    transformer = Transformer(d_model=768, num_heads=12, num_layers=12)
    image_encoder = ViT(
        patch = 16,
        channels = 3,
        embed_dim = 768,
        token_size = (224//16)**2,  # (pic_h//patch)*(pic_w//patch) 224//16
        img_size = 224,
        transformer = transformer
    )


    # Text Encoder
    tokenizer = AutoTokenizer.from_pretrained('./Bert_tokenizer')
    vocab_size = tokenizer.vocab_size
    encoder = transfomer_encoder(
        num_layers=12,
        vocab_size=vocab_size,
        num_heads=12,
        dropout=0.1,
        d_model=768,
    )
    text_encoder = TextTransformerForCLIP(encoder, cls_token_id=101)  # 使用正确的 cls_token_id

    # CLIP model
    clip_model = CLIP(image_encoder, text_encoder, embed_dim).to(device)

    # 优化器
    optimizer = optim.AdamW(clip_model.parameters(), lr=lr)

    # 训练
    for epoch in range(epochs):
        total_loss = 0
        # 使用 tqdm 包装 DataLoader
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for images, input_ids, attention_mask in loop:
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            image_feat, text_feat = clip_model(images, input_ids, attention_mask)
            loss = clip_loss(image_feat, text_feat, clip_model.logit_scale.exp())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            clip_model.logit_scale.data.clamp_(max=np.log(100))

            total_loss += loss.item()
            # 更新进度条显示当前 batch 的损失
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")
        torch.save(clip_model.state_dict(), f"clip_epoch{epoch + 1}.pt")

if __name__ == "__main__":
    main()