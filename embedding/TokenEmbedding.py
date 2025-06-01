import torch 
import torch.nn as nn
import math

"""
参数意义：
vocab_size: 词汇表的大小
d_model: 词嵌入向量的维度，通常取512
torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None)
padding_idx参数是一个索引，如果选定，那么此处的词向量全部设为0，并且不更新它的梯度
"""

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model, pad_id):
        super().__init__(vocab_size, d_model)
        self.emb = nn.Embedding(num_embeddings=vocab_size,
                                embedding_dim=d_model,
                                padding_idx=pad_id)
        self.d_model = d_model
        
    def forward(self, x):
        return self.emb(x) * math.sqrt(self.d_model)