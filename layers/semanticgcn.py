import math
import torch
import torch.nn.functional as F

from torch import nn
from layers.layers import SimpleGraphConvolutionLayer, select, clones


class SemanticGCN(nn.Module):
    """
    SemanticGCN
    """
    def __init__(self, args, input_dim, num_layers):
        super(SemanticGCN, self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.hidden_dim = input_dim // args.sem_att_heads
        self.num_layers = num_layers
        self.attention_heads = args.sem_att_heads
        self.dropout = args.sem_att_dropout
        self.k = args.sem_top_k
        self.gcn = SimpleGraphConvolutionLayer(self.input_dim, self.input_dim, self.dropout)
        self.attn = MultiHeadAttention(self.attention_heads,
                                       self.input_dim,
                                       aspect_aware=True,
                                       dropout=self.dropout)
        self.dense = nn.Linear(self.input_dim, self.hidden_dim)
        self.gat_w = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        self.gat_dropout = nn.Dropout(self.dropout)
        self.leakyrelu = nn.LeakyReLU(args.sem_alpha)
        self.layerNorm = nn.LayerNorm(self.input_dim)
        self.bias = nn.Parameter(torch.FloatTensor(self.num_layers, self.hidden_dim))

    def forward(self, input, adj, src_mask, aspect_mask):
        src_mask = src_mask.unsqueeze(-2)
        # aspect feature
        aspect_mask = aspect_mask.unsqueeze(-1).repeat(1, 1, self.input_dim)  # mask for aspect
        aspect = (input * aspect_mask).mean(dim=1)  # aspects average pooling
        for i in range(self.num_layers):
            att_adj = self.selectTopK(input, aspect, src_mask, i + 1)  # Top-k selection
            # att_adj = self.attn(input, input, aspect, src_mask)  # aspect aware
            input = input.unsqueeze(1).repeat(1, self.attention_heads, 1, 1)  # (B, att_heads, seq_len, hidden_dim)
            input = self.dense(input)  # FC:hidden_dim->hidden_dim/heads (B, att_heads, seq_len, hidden_dim/heads)
            input = torch.matmul(att_adj, input) + self.bias[i]
            input = F.relu(input)
            input = torch.cat([i.squeeze(1) for i in torch.split(input, 1, dim=1)], dim=-1)  # cat multi_heads
        return input

    def selectTopK(self, input, aspect, src_mask, layers):
        """
        select top-k scores
        """
        attn = self.attn(input, input, aspect, src_mask)
        if layers != 1:
            probability = F.softmax(attn.sum(dim=(-2, -1)), dim=0)
            max_idx = torch.argmax(probability, dim=1)
            attn_sum = torch.stack([attn[i][max_idx[i]] for i in range(len(max_idx))], dim=0)
        else:
            attn_sum = torch.sum(attn, dim=1)
        att_tensor = select(attn_sum, self.k).unsqueeze(1).repeat(1, self.attention_heads, 1, 1) * attn
        return F.softmax(att_tensor, dim=-1)

class MultiHeadAttention(nn.Module):
    """
    MHA
    """
    def __init__(self, h, d_model, aspect_aware=False, relation_aware=False, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0, "The input dimension needs to be divisible by the number of attention heads."
        self.d_k = d_model // h
        self.h = h
        self.aspect_aware = aspect_aware
        self.relation_aware = relation_aware
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)
        if aspect_aware:
            self.weight_m = nn.Parameter(torch.Tensor(self.h, self.d_k, self.d_k))
            self.bias_m = nn.Parameter(torch.Tensor(1))
            self.dense = nn.Linear(d_model, self.d_k)
        if relation_aware:
            self.linear_query = nn.Linear(d_model, d_model)
            self.linear_key = nn.Linear(d_model, self.d_k)

    def forward(self, query, key, aspect=None, mask=None):
        if mask is not None:
            mask = mask[:, :, :query.size(1)]
            mask = mask.unsqueeze(1)  # (B, 1, 1, seq)

        nbatches = query.size(0)
        if self.relation_aware:
            query = self.linear_query(query).view(nbatches, -1, self.h, self.d_k)  # (B, H, seq, dim)
            key = self.linear_key(key)  # key: relation adj(B, seq, seq, dim)
        else:
            query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                          for l, x in zip(self.linears, (query, key))]  # multiple heads dim: d/z
        # aspect aware attention
        if self.aspect_aware:
            batch, a_dim = aspect.size()[0], aspect.size()[1]
            aspect = aspect.unsqueeze(1).expand(batch, self.h, a_dim)  # (batch, heads, dim)
            aspect = self.dense(aspect)  # (batch, heads, dim/heads)
            aspect = aspect.unsqueeze(2).expand(batch, self.h, query.size()[2],
                                                self.d_k)  # (batch_size, heads, seq, dim)
            attn = self.attention(query, key, aspect=aspect, mask=mask, dropout=self.dropout)
        else:
            attn = self.attention(query, key, mask=mask, dropout=self.dropout)  # self-att score
        return attn

    def attention(self, query, key, aspect=None, mask=None, dropout=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        # aspect-aware attention
        if self.aspect_aware:
            batch = len(scores)
            p = self.weight_m.size(0)
            max = self.weight_m.size(1)
            weight_m = self.weight_m.unsqueeze(0).expand(batch, p, max, max)  # (B, heads, dim, dim)
            # attention scores
            aspect_scores = torch.tanh(torch.add(torch.matmul(torch.matmul(aspect, weight_m), key.transpose(-2, -1)), self.bias_m))  # [16,5,41,41]
            scores = torch.add(scores, aspect_scores)  # self-attn + aspect-aware attn

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e18)
        if not self.relation_aware:  # using relation do not choose the softmax
            scores = F.softmax(scores, dim=-1)
            if dropout is not None:
                scores = dropout(scores)
        return scores