import torch
import torch.nn as nn
import math

class ScaleDotProductAttention(nn.Module):
    def __init__(self, d_model, n_head, device):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.device = device
        self.n_head = n_head
        self.d_model = d_model
        
    def forward(self, q, k, v, mask=None):
        # 1. 接收的q, k, v都是四维张量[batch_size, n_head, max_len, d_tensor]
        # 2. 将三个张量转移至指定的硬件上
        # 3. 接收的mask是四维张量[batch_size, n_head, max_len, max_len]
        batch_size, _, max_len, d_tensor = q.size()
        q = q.to(self.device)
        k = k.to(self.device)
        v = v.to(self.device)
        
        k_t = k.transpose(2, 3)
        
        k_t = k_t.to(self.device)
        
        # score张量的维度是[batch_size, n_head, max_len, max_len]
        score = torch.matmul(q, k_t) / math.sqrt(d_tensor)
        score = score.to(self.device)
        
        if mask is not None:
            score = score.masked_fill(~mask, -65500)
            
        score = self.softmax(score)
        v = torch.matmul(score, v)
        
        return v, score