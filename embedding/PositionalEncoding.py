import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, batch_size, device):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.batch_size = batch_size
        self.device = device
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False

        # 以下代码是生成位置编码
        pos = torch.arange(0, self.max_len, device=self.device)
        pos = pos.float().unsqueeze(dim=1)
        # 此时pos的维度是[max_len, 1]
        
        _2i = torch.arange(0, self.d_model, step=2, device=self.device)
        _2i = _2i.unsqueeze(dim=0)
        # 此时_2i的维度是[1, d_model/2]
        
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / self.d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / self.d_model)))
        
        self.encoding = self.encoding.unsqueeze(dim=0)
        self.register_buffer("pos_encoding", self.encoding)
        
    def forward(self, x):
        a, b = x.size()
        return self.pos_encoding[:a, :b, :self.d_model]