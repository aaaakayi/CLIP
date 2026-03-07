import torch
import torch.nn as nn
from .tools import MultiHeadAttention




#1.基于位置的前馈网络 FNN
class FFN(nn.Module):
    """
    基于位置的前馈网络(Feed-Forward Network,FFN)
    目的在于将自注意力输出的特征进行进一步处理, 增加模型的表达能力, 提供额外的非线性

    """
    def __init__(self,input,num_hiddens,output,**kwargs):
        super(FFN,self).__init__(**kwargs)
        #一般来说,input=output,而num_hiddens应该比他们大(例如4倍)
        self.layer1 = nn.Linear(in_features=input,out_features=num_hiddens)
        self.layer2 = nn.Linear(in_features=num_hiddens,out_features=output)

    def forward(self,x):
        return self.layer2(torch.relu(self.layer1(x)))


#2.残差连接和层规范化(add&norm)
class add_norm(nn.Module):
    """
        残差连接和层规范化(add & norm)
        解决梯度消失/爆炸问题，使深层网络可训练
        稳定训练过程，加速收敛
        实现稳定高效的深度网络训练
    """
    def __init__(self,normalized_shape,dropout,**kwargs):
        super(add_norm,self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self,x,y):
        return self.ln(self.dropout(y)+x) #x,y形状应该一致


#3.transfomer encoder
class transfomer_cell(nn.Module):
    """
    x -> MultiHeadAttention -> y -> add&norm(dropout(y)+x) -> x' -> FFN -> y' -> add&norm(dropout(y')+x') -> x"
      -> next transfomer_cell
    """
    def __init__(self,
                 num_heads,dropout,d_model,
                 **kwargs
    ):
        super(transfomer_cell,self).__init__(**kwargs)
        #初始化多头注意力模型,一般来说q_size = k_size = v_size = num_hiddens = d_model
        self.multiheadAttention = MultiHeadAttention(
            num_heads,
            dropout,
            num_hiddens = d_model,
            q_size = d_model,
            k_size = d_model,
            v_size = d_model,
            bias = False
        )
        #初始化FFN
        self.ffn = FFN(input = d_model,num_hiddens = 4*d_model,output = d_model) #这里直接采用经典的涉及方式,简化参数
        #初始化add & norm
        self.addnorm = add_norm(normalized_shape = d_model,dropout = dropout)

    def forward(self,q, k, v, valid_len=None, mask=None):
        residual1 = q
        y,self.attention_weights = self.multiheadAttention(q,k,v,valid_len = valid_len, mask = mask) #编码过程不需要掩码
        x2 = self.addnorm(x=residual1,y=y)

        residual2 = x2
        y2 = self.ffn(x2)
        output = self.addnorm(x=residual2,y=y2)
        return output,self.attention_weights