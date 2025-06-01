import torch
import torch.nn as nn
from layers.ScaleDotProductAttention import ScaleDotProductAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, device):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.device = device
        self.attention = ScaleDotProductAttention(self.d_model, self.n_head, self.device)
        # 建立三个线性变换矩阵
        self.w_q = nn.Linear(d_model, d_model, device=device)
        self.w_k = nn.Linear(d_model, d_model, device=device)
        self.w_v = nn.Linear(d_model, d_model, device=device)
        self.w_concat = nn.Linear(d_model, d_model, device=device)
        
    def forward(self, q, k, v, mask=None):
        # 函数接收的参数q, k, v是三维的[batch_size, max_len, d_model]
        # 函数接收的都是词向量，因此需要先将词向量进行线性变换
        q = q.to(self.device)
        k = k.to(self.device)
        v = v.to(self.device)
        
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        
        # 分别对三个张量做分割变化
        q, k, v = self.split(q), self.split(k), self.split(v)
        
        out, attention = self.attention(q, k, v, mask=mask)
        
        out = self.concat(out)
        out = self.w_concat(out)
        
        return out
        
    def split(self, x):
        
        batch_size, max_len, _ = x.size()
        
        d_tensor = self.d_model // self.n_head
        
        x = x.view(batch_size, max_len, self.n_head, d_tensor).transpose(1, 2)
        
        return x
        
    def concat(self, x):
        
        batch_size, _, max_len, d_tensor = x.size()
        
        d_model = self.n_head * d_tensor
        
        x = x.transpose(1, 2).contiguous().view(batch_size, max_len, d_model)
        
        return x