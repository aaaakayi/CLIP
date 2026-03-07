# Vision Transformer (ViT)
import torch
import torch.nn as nn
import torch.nn.functional as F
from Transformer.model.transfomer_cell import transfomer_cell

# 定义符合ViT的Transformer 只需要一个encoder 无需词嵌入
class Transformer(nn.Module):
    def __init__(self,num_layers,num_heads,d_model,dropout=0.1):
        super().__init__()
        self.Sequential = nn.Sequential()
        for i in range(num_layers):
            self.Sequential.add_module(
                "block" + str(i),
                transfomer_cell(num_heads,dropout,d_model)
            )
    def forward(self,x):
        for blk in self.Sequential:
            x,_ = blk(x,x,x)
        return x

# 1.划分图片为patch
class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                          # [B, embed_dim, H', W']
        x = x.flatten(2)                           # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)                      # [B, num_patches, embed_dim]
        return x

class ViT(nn.Module):
    def __init__(
            self,
            patch,
            channels,
            embed_dim,
            token_size, # (pic_h//patch)*(pic_w//patch)
            img_size,
            transformer
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size = patch, in_chans=channels, embed_dim=embed_dim)
        self.transformer = transformer
        self.embed_dim = embed_dim
        self.norm = nn.LayerNorm(embed_dim)
        self.CLS_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn((1, token_size + 1, embed_dim)))

    def forward(self,x):
        # x : B,C,h,w
        B = x.shape[0]
        x = self.patch_embed(x) # x: B,token_size,embed_dim

        # 拓展CLS_token至batch
        CLS_tokens = self.CLS_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat([CLS_tokens,x],dim=1)

        # pos_embed
        x = x + self.pos_embed

        # norm
        x = self.norm(x)

        # transformer
        x = self.transformer(x)

        # 取出 [CLS] 对应的输出
        cls_out = x[:, 0, :]  # [B, embed_dim]
        return cls_out

if __name__ == "__main__":
    patch_embed = PatchEmbed(img_size=224, patch_size=16, in_chans=3, embed_dim=768)
    transformer = Transformer(d_model=768, num_heads=12, num_layers=12)
    token_size = (224//16)**2
    vit = ViT(
        patch = 16,
        channels = 3,
        embed_dim = 768,
        token_size = token_size,  # (pic_h//patch)*(pic_w//patch)
        img_size=224,
        transformer = transformer
    )

    # 测试
    x = torch.randn(2, 3, 224, 224)
    output = vit(x)
    print(output.shape)  # 应为 [2, 768]