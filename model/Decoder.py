import torch
import torch.nn as nn
from blocks.DecoderLayer import DecoderLayer
from embedding.TransformerEmbedding import TransformerEmbedding

class Decoder(nn.Module):
    def __init__(self, dec_voc_size, d_model, max_len, batch_size, n_head, n_layers, ffn_hidden, drop_prob, pad_id, device):
        super().__init__()
        self.emb = TransformerEmbedding(vocab_size=dec_voc_size,
                                        d_model=d_model,
                                        max_len=max_len,
                                        batch_size=batch_size,
                                        drop_prob=drop_prob,
                                        pad_id=pad_id,
                                        device=device)
                                        
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  n_head=n_head,
                                                  hidden=ffn_hidden,
                                                  drop_prob=drop_prob,
                                                  device=device) for _ in range(n_layers)])
                                                  
        self.linear = nn.Linear(d_model, dec_voc_size, device=device)
    def forward(self, enc, trg_tokenID, src_mask, trg_mask):
        trg = self.emb(trg_tokenID)
        
        for layer in self.layers:
            trg = layer(enc, trg, src_mask, trg_mask)
            
        output = self.linear(trg)
        
        return output