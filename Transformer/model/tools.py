"""
    包含一些transformer实现所需工具 掩码softmax函数,点积注意力汇聚函数,多头注意力机制,位置编码
"""
import torch
import math
import torch.nn as nn
import torch.nn.functional as F


def mask_softmax(x, valid_lens=None, mask=None):
    """
    实现Masked Softmax，支持各种形状的掩码

    参数:
    x: 注意力分数张量，形状为 [batch_size * num_heads, num_queries, num_keys]
        注意：在多头注意力中，x已经是重塑后的形状，batch维度已包含num_heads
    valid_lens: 有效长度，用于创建填充掩码，形状为 (batch_size,) 或 (batch_size, num_queries)
    mask: 外部传入的掩码，形状可以是：
          - 2D: (num_queries, num_keys)
          - 3D: (batch_size, num_queries, num_keys)
          - 4D: (batch_size, 1, 1, num_keys) 或 (batch_size, 1, num_queries, num_keys)

    返回:
    裁剪后进行softmax的值，形状与x相同
    """
    # x的形状：在MultiHeadAttention中，x已经是(batch_size * num_heads, num_queries, num_keys)
    # 我们需要通过mask的batch_size来推断num_heads
    batch_size_num_heads, num_queries, num_keys = x.shape

    # 处理外部传入的掩码
    if mask is not None:
        # 记录原始掩码形状
        original_mask_shape = mask.shape

        # 情况1：4维掩码 (从create_masks返回)
        if mask.dim() == 4:
            batch_size = mask.shape[0]  # 原始batch_size
            num_heads = batch_size_num_heads // batch_size

            # 确保能整除
            if batch_size_num_heads % batch_size != 0:
                raise ValueError(
                    f"掩码的batch_size={batch_size}无法整除x的batch_size={batch_size_num_heads}"
                )

            # 子情况1.1: (batch_size, 1, 1, num_keys) -> 用于编码器或编码器-解码器注意力
            if mask.shape[1] == 1 and mask.shape[2] == 1:
                # 扩展掩码：(batch_size, 1, 1, num_keys) -> (batch_size, num_heads, num_queries, num_keys)
                mask = mask.expand(-1, num_heads, num_queries, -1)
                # 重塑：(batch_size, num_heads, num_queries, num_keys) -> (batch_size * num_heads, num_queries, num_keys)
                mask = mask.reshape(batch_size_num_heads, num_queries, num_keys)

            # 子情况1.2: (batch_size, 1, num_queries, num_keys) -> 用于解码器自注意力
            elif mask.shape[1] == 1:
                # 扩展掩码：(batch_size, 1, num_queries, num_keys) -> (batch_size, num_heads, num_queries, num_keys)
                mask = mask.expand(-1, num_heads, -1, -1)
                # 重塑
                mask = mask.reshape(batch_size_num_heads, num_queries, num_keys)

            # 子情况1.3: (batch_size, num_queries, num_queries, num_keys) -> 不常见，但处理一下
            elif mask.shape[2] == num_queries:
                # 假设num_queries = num_keys
                mask = mask.squeeze(1) if mask.shape[1] == 1 else mask
                if mask.dim() == 4:
                    # 压缩到3D
                    mask = mask.mean(dim=1)  # 或者其他聚合方式
                mask = mask.expand(-1, num_heads, -1, -1).reshape(batch_size_num_heads, num_queries, num_keys)

        # 情况2：3维掩码 (batch_size, num_queries, num_keys)
        elif mask.dim() == 3:
            batch_size = mask.shape[0]
            num_heads = batch_size_num_heads // batch_size

            if batch_size_num_heads % batch_size == 0:
                # 扩展掩码到多头
                mask = mask.unsqueeze(1).expand(-1, num_heads, -1, -1)
                mask = mask.reshape(batch_size_num_heads, num_queries, num_keys)
            else:
                # 如果不能整除，假设mask已经是正确的形状
                if mask.shape != x.shape:
                    # 尝试广播
                    try:
                        mask = mask.expand_as(x)
                    except:
                        raise ValueError(
                            f"3D掩码形状{mask.shape}无法广播到x形状{x.shape}，且batch_size不匹配"
                        )

        # 情况3：2维掩码 (num_queries, num_keys)
        elif mask.dim() == 2:
            # 直接扩展到需要的形状
            mask = mask.unsqueeze(0).expand(batch_size_num_heads, -1, -1)

        else:
            raise ValueError(f"不支持的掩码维度: {mask.dim()}")

    # 初始化最终掩码
    final_mask = torch.zeros_like(x, dtype=torch.bool, device=x.device)

    # 添加外部掩码（如果有）
    if mask is not None:
        # 确保掩码形状匹配
        if mask.shape != x.shape:
            try:
                mask = mask.expand_as(x)
            except RuntimeError:
                raise ValueError(f"掩码形状{mask.shape}无法广播到x形状{x.shape}")

        # 注意：我们的掩码是True表示要屏蔽
        final_mask = final_mask | mask

    # 添加填充掩码（如果有valid_lens）
    if valid_lens is not None:
        # 确定batch_size
        if mask is not None and mask.dim() >= 3:
            # 从掩码推断batch_size
            batch_size = mask.shape[0] if mask.dim() == 3 else mask.shape[0]
        else:
            # 从x推断：假设x的第一维是batch_size * num_heads
            # 我们需要知道num_heads才能知道batch_size
            # 这里我们假设num_heads=1，或者使用有效长度推断
            batch_size = batch_size_num_heads

        # 确保valid_lens的形状正确
        if valid_lens.dim() == 1:
            # (batch_size,) -> (batch_size, 1, 1)
            valid_lens = valid_lens.view(-1, 1, 1)
            # 扩展到每个查询
            valid_lens = valid_lens.expand(-1, num_queries, 1)
        elif valid_lens.dim() == 2:
            # (batch_size, num_queries) -> (batch_size, num_queries, 1)
            valid_lens = valid_lens.unsqueeze(-1)

        # 创建位置索引
        positions = torch.arange(num_keys, device=x.device)
        positions = positions.view(1, 1, -1).expand(batch_size, num_queries, -1)

        # 创建填充掩码：位置 >= 有效长度的位置应该被屏蔽
        # 注意：valid_lens需要扩展到num_keys维度
        padding_mask = positions >= valid_lens.expand(-1, -1, num_keys)

        # 扩展填充掩码到多头（如果需要）
        if padding_mask.shape[0] < batch_size_num_heads:
            # 计算num_heads
            num_heads = batch_size_num_heads // padding_mask.shape[0]
            if batch_size_num_heads % padding_mask.shape[0] == 0:
                padding_mask = padding_mask.unsqueeze(1).expand(-1, num_heads, -1, -1)
                padding_mask = padding_mask.reshape(batch_size_num_heads, num_queries, num_keys)

        # 合并掩码
        final_mask = final_mask | padding_mask

    # 应用掩码：将被屏蔽的位置设置为负无穷
    x_masked = x.masked_fill(final_mask, -1e9)

    # 应用softmax
    result = F.softmax(x_masked, dim=-1)

    return result

class DotProductAttention(nn.Module):
    """
    缩放点积注意力 评分函数a(k,q) = (q^T @ k)/sqrt(d)，d: 查询和键的长度，适用于查询和键长度相同的情况
    :param
    :return
    """
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self, queries, keys, values, mask=None, valid_lens=None):
        d = queries.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = mask_softmax(scores, valid_lens = valid_lens, mask = mask)
        return torch.bmm(self.dropout(self.attention_weights), values), self.attention_weights


class MultiHeadAttention(nn.Module):
    #简化计算成本，
    def __init__(self,
                 num_heads,
                 dropout,
                 num_hiddens,
                 q_size,
                 k_size,
                 v_size,
                 bias=False
        ):
        super(MultiHeadAttention,self).__init__()
        self.num_heads = num_heads
        self.attention =  DotProductAttention(dropout) # 注意力汇聚函数

        #需要投影到一个维度
        self.Wq = nn.Linear(q_size,num_hiddens,bias=bias)
        self.Wk = nn.Linear(k_size,num_hiddens,bias=bias)
        self.Wv = nn.Linear(v_size,num_hiddens,bias=bias)

        self.Wo = nn.Linear(num_hiddens,num_hiddens,bias=bias)


    def forward(self,q,k,v,valid_len=None,mask=None):
        # 首先要把数据改成 batch_size*num_heads , seq_len , num_hiddens/num_heads
        # 经过线性变化再拆分，然后进行注意力汇聚
        q, k, v = self.Wq(q), self.Wk(k), self.Wv(v)

        q = Unpack_base_head(q,self.num_heads)
        k = Unpack_base_head(k, self.num_heads)
        v = Unpack_base_head(v, self.num_heads)

        # 此时q,k,v形状变成 batch_size * num_heads,seq_len,num_hiddens/num_heads
        # valid_len需要匹配成相应的形状
        if valid_len is not None:
            valid_len = torch.repeat_interleave(valid_len,repeats=self.num_heads, dim=0)

        #注意力汇聚
        output,self.attention_weights = self.attention(
            queries = q,
            keys = k,
            values = v,
            mask = mask,
            valid_lens = valid_len
        )

        output = Merge_base_head(output,self.num_heads)
        output = self.Wo(output)

        #流程：q,k,v -> 线性变化投射到同一维度 -> 根据head拆分 -> valid_len(如果需要)根据拆分后的情况调整 -> 注意力汇聚 -> 根据head合并输出 -> 输出经过线性变换输出
        return output,self.attention_weights


def Unpack_base_head(x,num_heads):
    #根据头进行拆分
    x = x.reshape(x.shape[0],x.shape[1],num_heads,-1) # shape -> batch_size,seq_len,num_heads,features_size/num_heads
    x = x.permute(0,2,1,3) #shape -> batch_size,num_heads,seq_len,features_size/num_heads
    return x.reshape(-1,x.shape[2],x.shape[3]) #shape -> batch_size * num_heads,seq_len,features_size/num_heads

def Merge_base_head(x,num_heads):
    #根据头进行合并，此时根据线性变化，x shape -> batch_size * num_heads,seq_len,num_hiddens/num_heads
    x = x.reshape(-1,num_heads,x.shape[1],x.shape[2]) # shape -> batch_size,num_heads,seq_len,num_hiddens
    x = x.permute(0,2,3,1) # shape -> batch_size,seq_len,num_hiddens,num_heads
    return x.reshape(x.shape[0],x.shape[1],-1) #shape -> batch_size,seq_len,num_hiddens * num_heads


class position_ecoder(nn.Module):
    def __init__(self,dropout,num_hiddens,max_len=1000):
        super(position_ecoder,self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1,max_len,num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


def create_masks(src_seq=None, tgt_seq=None, pad_token_id=0, device='cpu'):
    """
    创建Transformer所需的所有掩码

    参数:
    src_seq: 源序列，形状为 (batch_size, src_len)
    tgt_seq: 目标序列，形状为 (batch_size, tgt_len)
    pad_token_id: 填充标记ID

    返回:
    src_mask: 源序列填充掩码，用于编码器，形状为 (batch_size, 1, 1, src_len)
    tgt_mask: 目标序列掩码（前瞻+填充），用于解码器自注意力，形状为 (batch_size, 1, tgt_len, tgt_len)
    memory_mask: 编码器输出掩码，用于解码器交叉注意力，形状为 (batch_size, 1, 1, src_len)
    src_valid_len: 源序列有效长度，形状为 (batch_size,)
    tgt_valid_len: 目标序列有效长度，形状为 (batch_size,)

    支持单独更新tgt_seq的mask,能够满足推理工作的需求
    """
    result = {
        'src_mask': None,
        'tgt_mask': None,
        'memory_mask': None,
        'src_valid_len': None,
        'tgt_valid_len': None
    }

    if src_seq is not None:
        batch_size, src_len = src_seq.shape

        # 创建源序列填充掩码（用于编码器）
        src_mask = (src_seq == pad_token_id).unsqueeze(1).unsqueeze(2)  # [B,1,1,src_len]

        # 编码器输出掩码（用于解码器交叉注意力）
        memory_mask = src_mask.clone()

        # 源序列有效长度
        src_valid_len = (src_seq != pad_token_id).sum(dim=1)  # [B]

        # 更新结果
        result['src_mask'] = src_mask
        result['memory_mask'] = memory_mask
        result['src_valid_len'] = src_valid_len

    if tgt_seq is not None:
        batch_size, tgt_len = tgt_seq.shape

        # 创建目标序列填充掩码
        tgt_padding_mask = (tgt_seq == pad_token_id).unsqueeze(1).unsqueeze(2)  # [B,1,1,tgt_len]

        # 创建前瞻掩码（上三角矩阵）
        look_ahead_mask = torch.triu(
            torch.ones((tgt_len, tgt_len), device=device),
            diagonal=1
        ).bool().unsqueeze(0).unsqueeze(0)  # [1,1,tgt_len,tgt_len]

        # 扩展到批次维度
        look_ahead_mask = look_ahead_mask.expand(batch_size, -1, -1, -1)  # [B,1,tgt_len,tgt_len]

        # 扩展填充掩码到注意力矩阵形状
        tgt_padding_mask_expanded = tgt_padding_mask.expand(-1, -1, tgt_len, -1)  # [B,1,tgt_len,tgt_len]

        # 合并目标序列掩码（填充+前瞻）
        tgt_mask = tgt_padding_mask_expanded | look_ahead_mask

        # 目标序列有效长度
        tgt_valid_len = (tgt_seq != pad_token_id).sum(dim=1)  # [B]

        # 更新结果
        result['tgt_mask'] = tgt_mask
        result['tgt_valid_len'] = tgt_valid_len

    return result