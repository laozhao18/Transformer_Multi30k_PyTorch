import torch
import torch.nn as nn
from layers.PositionwiseFeedForward import PositionwiseFeedForward
from layers.ScaleDotProductAttention import ScaleDotProductAttention
from layers.MultiHeadAttention import MultiHeadAttention
from layers.LayerNorm import LayerNorm

class EncoderLayer(nn.Module):
    def __init__(self, d_model, max_len, batch_size, hidden, n_head, drop_prob, device):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.batch_size = batch_size
        self.device = device
        
        self.attention = MultiHeadAttention(d_model, n_head, device)
        self.norm1 = LayerNorm(d_model, device)
        self.dropout1 = nn.Dropout(p=drop_prob)
        
        self.ffn = PositionwiseFeedForward(d_model, hidden, drop_prob, device)
        self.norm2 = LayerNorm(d_model, device)
        self.dropout2 = nn.Dropout(p=drop_prob)
        
    def forward(self, x, src_mask):
        _x = x
        src_mask = src_mask.to(self.device)
        x = self.attention(q=x, k=x, v=x, mask=src_mask)
        
        # add and norm
        x = self.dropout1(x)
        x = self.norm1(_x + x)
        
        # 进入到FFN
        _x = x
        x = self.ffn(x)
        
        # add and norm
        x = self.dropout2(x)
        x = self.norm2(_x + x)
        
        return x