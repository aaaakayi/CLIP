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
            # tools.mask_softmax 约定: True 表示被屏蔽。
            if attention_mask.dim() == 2:
                # HF/BERT 常见约定: 1=有效token, 0=padding。
                attention_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
            elif attention_mask.dim() == 4:
                # 兼容已是 [B,1,1,L] 的padding掩码。
                attention_mask = attention_mask.bool()
            else:
                raise ValueError(f"Unsupported attention_mask dim: {attention_mask.dim()}")

            # CLS 不应被屏蔽，因此该位为 False。
            cls_mask = torch.zeros(B, 1, 1, 1, dtype=torch.bool, device=attention_mask.device)
            attention_mask = torch.cat([cls_mask, attention_mask], dim=-1)  # [B, 1, 1, seq_len+1]

            # valid_len 包含 CLS 位置。
            valid_len = (~attention_mask.squeeze(1).squeeze(1)).sum(dim=-1)
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
