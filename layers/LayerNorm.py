import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, d_model, device, eps=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model, device=device))
        self.beta = nn.Parameter(torch.zeros(d_model, device=device))
        self.eps = eps
        self.device = device
        
    def forward(self, x):
        x = x.to(self.device)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out