import torch
import torch.nn as nn
import torch.nn.functional as F
from .tools import create_masks
from .decoder import transfomer_decoder
from .encoder import transfomer_encoder

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,  # 源语言词汇表大小
        tgt_vocab_size: int,  # 目标语言词汇表大小
        num_layers: int,  # 编码器和解码器层数（通常相同）
        d_model: int,   # 模型维度
        num_heads: int,   # 注意力头数
        dropout: float, # dropout概率
        max_src_len: int,  # 源序列最大长度
        max_tgt_len: int,  # 目标序列最大长度
        **kwargs
    ): # d_model需要被num_heads整除
        super(Transformer, self).__init__(**kwargs)

        self.create_masks = create_masks

        #初始化编码器
        self.encoder = transfomer_encoder(
            num_layers=num_layers,
            vocab_size=src_vocab_size,
            num_heads=num_heads,
            dropout=dropout,
            d_model=d_model
        )

        #初始化解码器
        self.decoder = transfomer_decoder(
            num_layers=num_layers,
            vocab_size=tgt_vocab_size,
            num_heads=num_heads,
            dropout=dropout,
            d_model=d_model
        )

        #最终输出层
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        #参数初始化
        self._init_parameters()

        self.pad_token_id = 0 # 填充id
        self.max_src_len = max_src_len # 源序列最大长度
        self.max_tgt_len = max_tgt_len # 目标序列最大长度

    def _init_parameters(self):
        """初始化模型参数（使用Xavier初始化）"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
            self,
            src_seq: torch.Tensor,  # 源序列 [batch_size, src_len]
            tgt_seq: torch.Tensor,  # 目标序列 [batch_size, tgt_len]
    ):
        #创建掩码
        masks = self.create_masks(
            src_seq=src_seq,
            tgt_seq=tgt_seq,
            pad_token_id=self.pad_token_id,
            device=src_seq.device
        )

        # 编码器前向传播
        encoder_output, encoder_attention_weights = self.encoder(
            x=src_seq,
            valid_len = masks['src_valid_len'],
            mask = masks['src_mask']
        )

        # 解码器前向传播
        decoder_output, decoder_self_attention_weights, cross_attention_weights = self.decoder(
            decoder_input=tgt_seq,
            encoder_output=encoder_output,
            tgt_mask=masks['tgt_mask'],
            tgt_valid_len = masks['tgt_valid_len'],
            memory_mask = masks['memory_mask'],
            memory_valid_len = masks['src_valid_len']
        )

        # 最终输出
        logits = self.output_projection(decoder_output)

        # 注意力权重 用于可视化
        attention_weights = {
            'encoder': encoder_attention_weights,
            'decoder_self': decoder_self_attention_weights,
            'decoder_cross': cross_attention_weights
        }

        return logits, attention_weights

    @torch.no_grad()
    def generate(
            self,
            src_seq: torch.Tensor,
            max_len: int = 50,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0,
            repetition_penalty: float = 1.0,
            bos_token_id: int = 1,  # 起始标记ID
            eos_token_id: int = 2,  # 结束标记ID
            **kwargs
    ):
        """
        自回归生成目标序列（推理阶段）

        参数:
            src_seq: 源序列 [batch_size, src_len]
            max_len: 最大生成长度
            temperature: 温度参数（控制随机性）
            top_k: top-k采样参数
            top_p: top-p（核）采样参数
            repetition_penalty: 重复惩罚
            bos_token_id: 起始标记ID
            eos_token_id: 结束标记ID
        """
        self.eval()  # 确保在评估模式

        batch_size = src_seq.size(0)
        device = src_seq.device

        mask = self.create_masks(
            src_seq=src_seq, tgt_seq=None, pad_token_id=self.pad_token_id, device=device
        )
        # 以下掩码只需要计算一次
        src_mask = mask['src_mask']
        src_valid_len = mask['src_valid_len']
        memory_mask = mask['memory_mask']

        # 编码源序列
        encoder_output, _ = self.encoder(
            x=src_seq,
            mask=src_mask
        )

        # 初始化目标序列（以bos_token开始）
        generated = torch.full(
            (batch_size, 1), bos_token_id,
            dtype=torch.long, device=device
        )

        # 3. 生成循环
        for step in range(max_len - 1):

            # 每次循环需要重新计算目标序列掩码
            # 因为generated在不断变长
            mask = create_masks(
                src_seq=None,tgt_seq=generated, pad_token_id=self.pad_token_id, device=device
            )

            # 需要随目标序列更新
            tgt_mask = mask['tgt_mask']
            tgt_valid_len = mask['tgt_valid_len']

            # 解码器前向传播
            decoder_output, _, _ = self.decoder(
                decoder_input=generated,
                encoder_output=encoder_output,
                tgt_mask=tgt_mask,
                tgt_valid_len=tgt_valid_len,
                memory_mask=memory_mask,
                memory_valid_len=src_valid_len
            )

            # 获取最后一个位置的logits
            logits = self.output_projection(decoder_output[:, -1, :])

            # 应用采样策略
            next_token = self._sample_next_token(
                logits=logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                generated_tokens=generated
            )

            # 添加到生成序列
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

            # 检查是否所有序列都结束
            if torch.all(generated[:, -1] == eos_token_id):
                break

        return generated

    def _sample_next_token(
            self,
            logits: torch.Tensor,
            temperature: float,
            top_k: int,
            top_p: float,
            repetition_penalty: float,
            generated_tokens: torch.Tensor
    ):
        """采样下一个token"""
        # 1. 应用重复惩罚
        if repetition_penalty != 1.0:
            for token_id in torch.unique(generated_tokens):
                logits[:, token_id] /= repetition_penalty

        # 2. 应用温度
        if temperature != 1.0:
            logits = logits / temperature

        # 3. Top-k采样
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = -float('Inf')

        # 4. Top-p（核）采样
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # 移除累积概率超过top_p的token
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = -float('Inf')

        # 5. 从softmax分布中采样
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)

        return next_token