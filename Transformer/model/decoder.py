import torch
import torch.nn as nn
from .tools import MultiHeadAttention,position_ecoder
from .transfomer_cell import transfomer_cell,FFN,add_norm



class transfomer_decoder_cell(nn.Module):
    """
    transfomer decoder cell:
    包含解码器自注意力、“编码器-解码器”注意力和基于位置的前馈网络
    cell部分依旧可用,此时cell的输入变成了 解码器自注意力的输出作为q,编码器的输出作为k,v
    流程如下：
    输入x -> 掩码自注意力 -> add&norm -> 编码器-解码器注意力（其中k和v来自编码器输出，q来自上一个add&norm的输出） -> add&norm -> FFN -> add&norm -> 输出
    复用cell 部分代码，简化成输入x -> 掩码自注意力 -> add&norm -> cell 此时cell输入需要修改
    """
    def __init__(
            self,
            num_heads,
            dropout,
            d_model,
            **kwargs
    ):
        super(transfomer_decoder_cell,self).__init__(**kwargs)
        self.multiattention1 = MultiHeadAttention(
            num_heads = num_heads,
            dropout = dropout,
            num_hiddens = d_model,
            q_size = d_model,
            k_size = d_model,
            v_size = d_model,
            bias=False
        ) #编码器自注意力模块

        self.add_norm1 = add_norm(
            normalized_shape = d_model,
            dropout = dropout
        )

        self.cell = transfomer_cell(
            num_heads = num_heads,
            dropout = dropout,
            d_model = d_model,
        ) #cell 依旧可用，仅修改q为解码器自注意力输出,k,v为encoder输出,维持d_model一致

    def forward(self,decoder_input, encoder_output,
                mask=None,valid_len=None, #decoder输入序列的掩蔽方式
                encoder_mask=None, encoder_valid_len=None #原encoder序列的掩蔽方式
        ):
        # 实际上 q为编码器输出 k=v=解码器自注意力输出
        residual1 = decoder_input
        y1,self.attention_weights = self.multiattention1(decoder_input,decoder_input,decoder_input,valid_len=valid_len,mask=mask)
        x2 = self.add_norm1(x=residual1,y=y1) #编码器自注意力输出

        # --- 第2步：编码器-解码器注意力 + FFN ---
        # 使用复用的transformer_cell
        # q = x2 (解码器自注意力输出)
        # k = v = encoder_output (编码器输出)
        output, self.cross_attention_weights = self.cell(
            q = x2, k = encoder_output, v = encoder_output, valid_len = encoder_valid_len, mask = encoder_mask
        )

        return output, self.attention_weights, self.cross_attention_weights


class transfomer_decoder(nn.Module):
    """
    基于transformer decoder cell 构建的 transfomer decoder
    """
    def __init__(
            self,
            vocab_size,
            num_layers,
            num_heads, dropout, d_model,
            **kwargs
    ):
        super(transfomer_decoder,self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size,d_model)
        self.pos_encoding = position_ecoder(dropout, num_hiddens=d_model, max_len=1000)
        self.sequential = nn.Sequential()
        for i in range(num_layers):
            self.sequential.add_module(
                "block" + str(i),
                transfomer_decoder_cell(num_heads,dropout,d_model)
            )

        # self.output_layer = nn.Linear(d_model, vocab_size),解码器中不需要输出拓展成vocab_size

    def forward(self,
                decoder_input, encoder_output,
                tgt_mask=None, tgt_valid_len=None, # 解码器掩码
                memory_mask=None, memory_valid_len=None # 原编码器掩码
        ):
        self.attention_weights = []
        self.cross_attention_weights = []
        q = self.embedding(decoder_input)
        q = self.pos_encoding(q)
        i = 1
        for blk in self.sequential:
            q, attention_weights, cross_attention_weights = blk(
                decoder_input=q, encoder_output=encoder_output,
                mask=tgt_mask, valid_len=tgt_valid_len,
                encoder_mask=memory_mask, encoder_valid_len=memory_valid_len
            )
            self.attention_weights.append(
                (attention_weights,f"decoder block{i}",True,"Reds")
            )
            self.cross_attention_weights.append(
                (cross_attention_weights, f"decoder block{i}", True, "Reds")
            )
            i += 1
        output = q
        return output,self.attention_weights,self.cross_attention_weights