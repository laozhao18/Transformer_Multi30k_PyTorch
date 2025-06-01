# Transformer_Multi30K_PyTorch
## Reproducing the Transformer paper using the Multi30K dataset.

## 可以跳转到[CSDN博客](https://blog.csdn.net/weixin_52054839/article/details/147606285?spm=1011.2415.3001.10575&sharefrom=mp_manage_link)看更加详细的讲解。
# 注意，以下代码不是完整的代码，只是**主体结构Transformer, Encoder和Decoder**，以及比较重要的**缩放点积注意力机制(ScaleDotProductAttention)**

## Transformer

```python
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

## Encoder
```python
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
    
```
## Decoder
```python
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
```

## ScaleDotProductAttention
```python
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
```
