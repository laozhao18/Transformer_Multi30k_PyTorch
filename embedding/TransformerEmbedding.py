import torch
import torch.nn as nn
from embedding.TokenEmbedding import TokenEmbedding
from embedding.PositionalEncoding import PositionalEncoding

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, batch_size, drop_prob, pad_id, device):
        super().__init__()
        self.tok = TokenEmbedding(vocab_size, d_model, pad_id)
        self.poe = PositionalEncoding(d_model, max_len, batch_size, device)
        self.dropout = nn.Dropout(p=drop_prob)
        self.device = device
        self.tok = self.tok.to(device)
        self.poe = self.poe.to(device)
        
    def forward(self, x):
        x = x.to(self.device)
        tok = self.tok(x).to(self.device)
        poe = self.poe(x).to(self.device)
        return self.dropout(tok + poe)