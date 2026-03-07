# 构建文本所需的Transformer
# encoder only
import torch
import torch.nn as nn
from Transformer.model.encoder import transfomer_encoder
from Transformer.model.tools import create_masks

class TextTransformerForCLIP(nn.Module):
    def __init__(self, transfomer_encoder, cls_token_id):
        super().__init__()
        self.encoder = transfomer_encoder
        self.cls_token_id = cls_token_id

    def forward(self, input_ids, attention_mask=None):
        B, seq_len = input_ids.shape
        # 创建 [CLS] token
        cls_ids = torch.full((B, 1), self.cls_token_id, device=input_ids.device)
        input_ids = torch.cat([cls_ids, input_ids], dim=1)  # [B, seq_len+1]

        if attention_mask is not None:
            # 假设 attention_mask 形状为 [B, seq_len]（二维）
            # 转换为 [B, 1, 1, seq_len]（四维）
            attention_mask = attention_mask.bool()
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, seq_len]
            # 创建 cls_mask（对应 [CLS] token 的位置，始终有效）
            cls_mask = torch.ones(B, 1, 1, 1, dtype=attention_mask.dtype, device=attention_mask.device)
            # 在最后一个维度拼接
            attention_mask = torch.cat([cls_mask, attention_mask], dim=-1)  # [B, 1, 1, seq_len+1]

            # 如果编码器需要 valid_len，可计算（原始有效长度 + 1）
            # valid_len = attention_mask_orig.sum(dim=1) + 1  # 但这里没有原始 mask，可从 attention_mask 去掉 cls 部分再计算
            valid_len = None  # 先设为 None，视内部实现而定
        else:
            valid_len = None

        x, _ = self.encoder(input_ids, valid_len=valid_len, mask=attention_mask)
        cls_out = x[:, 0, :]  # 取 [CLS] 输出
        return cls_out

if __name__ == "__main__":
    x = torch.randint(0, 10000, (2, 50))
    pad = torch.zeros((2,50),dtype=torch.int32)
    x = torch.cat([x,pad],dim=1)
    mask = create_masks(src_seq=x, pad_token_id=0, device='cpu')
    encoder = transfomer_encoder(
        num_layers = 12,
        vocab_size = 10000,
        num_heads = 12,
        dropout = 0.1,
        d_model = 768,
    )
    text_transformer = TextTransformerForCLIP(
        transfomer_encoder = encoder,
        cls_token_id = 0,
    )
    print(x.shape)
    print(mask['src_mask'].shape)
    output =text_transformer(input_ids = x,attention_mask = mask['src_mask'])

    print(output.shape)
