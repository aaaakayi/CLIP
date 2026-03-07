from .transfomer_cell import transfomer_cell
import torch.nn as nn
from .tools import position_ecoder

class transfomer_encoder(nn.Module):
    """
    基于transfomer_cell构建transfomer_encoder
    """
    def __init__(self,
                 num_layers,vocab_size,
                 num_heads, dropout, d_model,
                 **kwargs
    ):
        super(transfomer_encoder,self).__init__(**kwargs)
        self.d_model = d_model
        self.sequentia = nn.Sequential()
        self.embeded = nn.Embedding(vocab_size,d_model)
        self.pos_encoding = position_ecoder(dropout,num_hiddens=d_model,max_len=1000)
        for i in range(num_layers):
            self.sequentia.add_module(
                "block" + str(i),
                transfomer_cell(num_heads,dropout,d_model)
            )
    def forward(self,x,valid_len=None,mask=None):
        x = self.embeded(x)
        x = self.pos_encoding(x)
        self.attention_weight = []
        i = 1
        for blk in self.sequentia:
            x,attention_weight = blk(x,x,x,valid_len=valid_len, mask=mask)
            self.attention_weight.append(
                (attention_weight,f"enocde block{i}",True,"Reds")
            )
            i += 1

        return x,self.attention_weight