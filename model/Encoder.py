import torch
import torch.nn as nn
from blocks.EncoderLayer import EncoderLayer
from embedding.TransformerEmbedding import TransformerEmbedding
'''
Encoder这里的forward函数直接输入tokenID序列，是一个二维张量[batch_size, max_len]
'''

class Encoder(nn.Module):
    def __init__(self, enc_voc_size, d_model, max_len, batch_size, n_head, n_layers, ffn_hidden, drop_prob, pad_id, device):
        super().__init__()
        self.device = device
        self.emb = TransformerEmbedding(vocab_size=enc_voc_size,
                                        d_model=d_model,
                                        max_len=max_len,
                                        batch_size=batch_size,
                                        drop_prob=drop_prob,
                                        pad_id=pad_id,
                                        device=device)
                                        
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                     max_len=max_len,
                                     batch_size=batch_size,
                                     hidden=ffn_hidden,
                                     n_head=n_head,
                                     drop_prob=drop_prob,
                                     device=device) for _ in range(n_layers)])
                                     

    def forward(self, x, src_mask):
        x = self.emb(x)
        
        for layer in self.layers:
            x = layer(x, src_mask)
            
        return x
    