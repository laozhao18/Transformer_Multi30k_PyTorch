import torch
import torch.nn as nn
from layers.MultiHeadAttention import MultiHeadAttention
from layers.PositionwiseFeedForward import PositionwiseFeedForward
from layers.LayerNorm import LayerNorm

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, hidden, drop_prob, device):
        super().__init__()
        self.device = device
        self.self_attention = MultiHeadAttention(d_model, n_head, device)
        self.norm1 = LayerNorm(d_model, device)
        self.dropout1 = nn.Dropout(p=drop_prob)
        
        self.enc_dec_attention = MultiHeadAttention(d_model, n_head, device)
        self.norm2 = LayerNorm(d_model, device)
        self.dropout2 = nn.Dropout(p=drop_prob)
        
        self.ffn = PositionwiseFeedForward(d_model, hidden, drop_prob, device)
        self.norm3 = LayerNorm(d_model, device)
        self.dropout3 = nn.Dropout(p=drop_prob)
        
    def forward(self, enc, dec, src_mask, trg_mask):
        src_mask = src_mask.to(self.device)
        trg_mask = trg_mask.to(self.device)
        # 1. 计算自我注意力分数
        _x = dec
        x = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask)
        
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(_x + x)
        
        if enc is not None:
            # 3. 计算encoder-decoder attention分数
            _x = x
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)
            
            # 4. add and norm
            x = self.dropout2(x)
            x = self.norm2(_x + x)
            
        # 5. 通过逐点前馈神经网络
        _x = x
        x = self.ffn(x)
        
        # 6. add and norm
        x = self.dropout3(x)
        x = self.norm3(_x + x)
        
        return x