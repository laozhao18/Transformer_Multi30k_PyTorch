import torch
import torch.nn as nn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob, device):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden, device=device)
        self.linear2 = nn.Linear(hidden, d_model, device=device)
        self.dropout = nn.Dropout(p=drop_prob)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x