# Transformer_Multi30K_PyTorch
Reproducing the Transformer paper using the Multi30K dataset.

可以跳转到[CSDN博客](https://blog.csdn.net/weixin_52054839/article/details/147606285?spm=1011.2415.3001.10575&sharefrom=mp_manage_link)看更加详细的讲解。

```
import torch
import torch.nn as nn

from model.Encoder import Encoder
from model.Decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, pad_idx, enc_voc_size, dec_voc_size, d_model, max_len, batch_size, n_head, n_layers, ffn_hidden, drop_prob, device):
        super().__init__()
        self.device = device
        self.pad = pad_idx
        self.n_head = n_head
        self.max_len = max_len
        self.batch_size = batch_size
        
        self.encoder = Encoder(enc_voc_size=enc_voc_size,
                               d_model=d_model,
                               max_len=max_len,
                               batch_size=batch_size,
                               n_head=n_head,
                               n_layers=n_layers,
                               ffn_hidden=ffn_hidden,
                               drop_prob=drop_prob,
                               pad_id=pad_idx,
                               device=device)
                               
        self.decoder = Decoder(dec_voc_size=dec_voc_size,
                               d_model=d_model,
                               max_len=max_len,
                               batch_size=batch_size,
                               n_head=n_head,
                               n_layers=n_layers,
                               ffn_hidden=ffn_hidden,
                               drop_prob=drop_prob,
                               pad_id=pad_idx,
                               device=device)
                              
    # encoder编码器接收的是tokenID序列
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc = self.encoder(src, src_mask)
        output = self.decoder(enc, trg, src_mask, trg_mask)
        return output
        
    def make_src_mask(self, src):
        src_mask = (src != self.pad)
        src_mask1 = src_mask.unsqueeze(1)
        src_mask2 = src_mask.unsqueeze(2)
        src_mask = (src_mask1 & src_mask2)
        src_mask = src_mask.unsqueeze(1)
        
        src_mask = src_mask.expand(src.shape[0], self.n_head, -1, -1)
        return src_mask
        
    def make_trg_mask(self, trg):
        # 创建无效字符的掩码
        trg_pad_mask = (trg != self.pad).to(self.device)
        trg_pad1 = trg_pad_mask.unsqueeze(1).to(self.device)
        trg_pad2 = trg_pad_mask.unsqueeze(2).to(self.device)
        trg_pad_mask = (trg_pad1 & trg_pad2).unsqueeze(1)
        trg_pad_mask = trg_pad_mask.expand(trg.shape[0], self.n_head, -1, -1)

        # 创建因果编码
        trg_tri_mask = torch.tril(torch.ones(self.max_len, self.max_len)).bool().to(self.device)
        trg_tri_mask = trg_tri_mask.unsqueeze(0).unsqueeze(0)
        trg_tri_mask = trg_tri_mask.expand(trg.shape[0], self.n_head, -1, -1)
        trg_mask = trg_pad_mask & trg_tri_mask
        return trg_mask
```
