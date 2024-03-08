import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicLSTM(nn.Module):
    '''
    LSTM which can hold variable length sequence, use like TensorFlow's RNN(input, lenght...).
    '''
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0,
                 bidirectional=False, only_use_last_hidden_state=False, rnn_type='LSTM'):
        super(DynamicLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.only_use_last_hidden_state = only_use_last_hidden_state
        self.rnn_type = rnn_type
        
        if self.rnn_type == 'LSTM':
            self.RNN = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                               bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'GRU':
            self.RNN = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                              bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'RNN':
            self.RNN = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                              bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
    
    def forward(self, x, x_len):
        '''
        sequence -> sort -> pad and pack -> process using RNN -> unpack -> unsort
        '''
        '''sort'''
        x_sort_idx = torch.sort(x_len, descending=True)[1].long()
        x_unsort_idx = torch.sort(x_sort_idx)[1].long()
        x_len = x_len[x_sort_idx]
        x = x[x_sort_idx]
        '''pack'''
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=self.batch_first)
        ''' process '''
        if self.rnn_type == 'LSTM':
            out_pack, (ht, ct) = self.RNN(x_emb_p, None)
        else:
            out_pack, ht = self.RNN(x_emb_p, None)
            ct = None
        '''unsort'''
        ht = ht[:, x_unsort_idx]
        if self.only_use_last_hidden_state:
            return ht
        else:
            out, _ = torch.nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=self.batch_first)
            if self.batch_first:
                out = out[x_unsort_idx]
            else:
                out = out[:, x_unsort_idx]
            if self.rnn_type == 'LSTM':
                ct = ct[:, x_unsort_idx]
            return out, (ht, ct)

class SqueezeEmbedding(nn.Module):
    '''
    Squeeze sequence embedding length to the longest one in the batch
    '''
    def __init__(self, batch_first=True):
        super(SqueezeEmbedding, self).__init__()
        self.batch_first = batch_first
    
    def forward(self, x, x_len):
        '''
        sequence -> sort -> pad and pack -> unpack -> unsort
        '''
        '''sort'''
        x_sort_idx = torch.sort(x_len, descending=True)[1].long()
        x_unsort_idx = torch.sort(x_sort_idx)[1].long()
        x_len = x_len[x_sort_idx]
        x = x[x_sort_idx]
        '''pack'''
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=self.batch_first)
        '''unpack'''
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(x_emb_p, batch_first=self.batch_first)
        if self.batch_first:
            out = out[x_unsort_idx]
        else:
            out = out[:, x_unsort_idx]
        return out

class SoftAttention(nn.Module):
    '''
    Attention Mechanism for ATAE-LSTM
    '''
    def __init__(self, hidden_dim, embed_dim):
        super(SoftAttention, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.w_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_p = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_x = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.weight = nn.Parameter(torch.Tensor(hidden_dim+embed_dim))
    
    def forward(self, h, aspect):
        hx = self.w_h(h)
        vx = self.w_v(aspect)
        hv = torch.tanh(torch.cat((hx, vx), dim=-1))
        ax = torch.unsqueeze(F.softmax(torch.matmul(hv, self.weight), dim=-1), dim=1)
        rx = torch.squeeze(torch.bmm(ax, h), dim=1)
        hn = h[:, -1, :]
        hs = torch.tanh(self.w_p(rx)+self.w_x(hn))
        return hs


class Attention(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='dot_product', dropout=0):
        ''' Attention Mechanism
        :param embed_dim:
        :param hidden_dim:
        :param out_dim:
        :param n_head: num of head (Multi-Head Attention)
        :param score_function: scaled_dot_product / mlp (concat) / bi_linear (general dot)
        :return (?, q_len, out_dim,)
        '''
        super(Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.w_k = nn.Linear(embed_dim, n_head * hidden_dim)
        self.w_q = nn.Linear(embed_dim, n_head * hidden_dim)
        self.proj = nn.Linear(n_head * hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim*2))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:  # dot_product / scaled_dot_product
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k, q):
        if len(q.shape) == 2:
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:
            k = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]  # ?
        k_len = k.shape[1]
        q_len = q.shape[1]
        kx = self.w_k(k).view(mb_size, k_len, self.n_head, self.hidden_dim)
        kx = kx.permute(2, 0, 1, 3).contiguous().view(-1, k_len, self.hidden_dim)
        qx = self.w_q(q).view(mb_size, q_len, self.n_head, self.hidden_dim)
        qx = qx.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.hidden_dim)
        if self.score_function == 'dot_product':
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qx, kt)
        elif self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx, kt)
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'mlp':
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            kq = torch.cat((kxx, qxx), dim=-1)
            score = F.tanh(torch.matmul(kq, self.weight))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qw, kt)
        else:
            raise RuntimeError('invalid score_function')
        score = F.softmax(score, dim=-1)
        output = torch.bmm(score, kx)
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)
        output = self.proj(output)
        output = self.dropout(output)
        return output, score

class GraphConvolutionLayer(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=-1, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GraphConvolutionLayer(nn.Module):
    """
    Simple GCN layer
    """
    def __init__(self, in_features, out_features, dropout=0.2):
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Linear(in_features, out_features)
        self.gcn_drop = nn.Dropout(dropout)

    def forward(self, input, adj):
        denom = torch.sum(adj, dim=-1, keepdim=True) + 1
        Ax = adj.matmul(input)
        AxW = self.weight(Ax)
        AxW = AxW / denom
        gAxW = F.relu(AxW)
        # if dataset is not laptops else gcn_inputs = self.gcn_drop(gAxW)
        # output = self.gcn_drop(gAxW) if l < self.layers - 1 else gAxW
        return self.gcn_drop(gAxW)

class SimpleGraphConvolutionLayer(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(SimpleGraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=-1, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.Tensor(in_features, out_features))
        self.a = nn.Parameter(torch.Tensor(2 * out_features, 1))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)  # (B, seq_len, dim)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])  # (B, seq_len, 1)
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])  # (B, seq_len, 1)
        # Wh1 + Wh2.T 是N*N矩阵，第i行第j列是Wh1[i]+Wh2[j]
        # 那么Wh1 + Wh2.T的第i行第j列刚好就是文中的a^T*[Whi||Whj]
        # 代表着节点i对节点j的attention
        e = self.leakyrelu(Wh1 + Wh2.transpose(1, 2))  # (N, N)，计算所有节点的注意力分数
        padding = -9e15 * torch.ones_like(e)  # (N, N),生成负无穷的矩阵，做Mask
        attention = torch.where(adj > 0, e, padding)  # (N, N)，注意力分布矩阵，若有邻接则需要注意力系数，否则负无穷
        attention = F.softmax(attention, dim=1)  # (N, N)，注意力矩阵归一化
        # attention矩阵第i行第j列代表node_i对node_j的注意力
        # 对注意力权重也做dropout（如果经过mask之后，attention矩阵也许是高度稀疏的，这样做还有必要吗？）
        attention = F.dropout(attention, self.dropout, training=self.training)  # (N, N)
        h = torch.matmul(attention, Wh)  # (B, seq_len, dim)
        if self.concat:
            return F.elu(h)
        else:
            return h

class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.

        Args:
            d_model (int): the size of input for the first-layer of the FFN.
            d_ff (int): the hidden layer size of the second-layer
                                of the FNN.
            dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        Layer definition.

        Args:
            input: [ batch_size, input_len, model_dim ]


        Returns:
            output: [ batch_size, input_len, model_dim ]
        """
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x

class SRDGCNLayer(nn.Module):
    """
    syntactic Relative Distance GCN
    """

    def __init__(self, args, in_features, out_features, dropout=0.2):
        super(SRDGCNLayer, self).__init__()
        self.args = args
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Linear(in_features, out_features)
        self.gcn_drop = nn.Dropout(dropout)

    def forward(self, input, adj, dep_dist_adj, key_padding_mask, aspect_mask):
        """句法依存距离权重"""
        # Ax = dep_dist_adj.matmul(input)
        # AxW = self.weight(Ax)
        # gAxW = F.relu(AxW)
        """句子词间距离权重"""
        # aspect_ids_mask = torch.nonzero(torch.eq(aspect_mask, 1))
        # batch_size = input.size(0)
        # aspect_lst = []
        # temp = 0
        # for i, v in aspect_ids_mask.numpy():
        #     if i == temp:
        #         aspect_lst.append([v, v])
        #         temp = i + 1
        #     else:
        #         aspect_lst[i] = [aspect_lst[i][0], v]
        # aspect_double_idx = torch.tensor(aspect_lst, dtype=torch.int64)  # aspect ids[left_index, right_index]
        # aspect_len = aspect_mask.sum(dim=-1)  # aspect len
        # input_len = (~key_padding_mask).sum(-1)  # seq len
        # # seq distance weight
        # input = self.position_weight(input, aspect_double_idx, input_len, aspect_len)
        """不带任何权重的GCN"""
        denom = torch.sum(adj, dim=-1, keepdim=True) + 1
        Ax = adj.matmul(input)
        AxW = self.weight(Ax)
        AxW = AxW / denom
        gAxW = F.relu(AxW)
        return self.gcn_drop(gAxW)

# class RelationAttentionGATLayer(nn.Module):
#     """
#     dependent relation GAT
#     """
#     def __init__(self, args):
#         # in_dim: the dimension fo query vector
#         super(RelationAttentionGATLayer, self).__init__()
#         self.args = args
#         self.self_attn = MultiHeadAttention(args.attention_heads, args.rnn_hidden * 2,
#                                                 dropout=args.att_dropout)
#         self.relation_attn = MultiHeadAttention(args.attention_heads, args.rnn_hidden * 2,
#                                                 relation_aware=True,
#                                                 dropout=args.att_dropout)
#         self.feed_forward = PositionwiseFeedForward(args.rnn_hidden * 2, args.rnn_hidden * 2, args.layer_dropout)
#         self.layer_norm = nn.LayerNorm(args.rnn_hidden * 2, eps=1e-6)
#         self.attn_dropout = nn.Dropout(args.att_dropout)
#         self.gcn_dropout = nn.Dropout(args.gcn_dropout)
#         self.linear_tok_value = nn.Linear(args.rnn_hidden * 2, args.rnn_hidden * 2 // args.attention_heads)
#         self.linear_rel_value = nn.Linear(args.rnn_hidden * 2, args.rnn_hidden * 2 // args.attention_heads)
#
#     def position_weight(self, x, aspect_double_idx, text_len, aspect_len):
#         batch_size = x.shape[0]
#         seq_len = x.shape[1]
#         aspect_double_idx = aspect_double_idx.cpu().numpy()  # (τ+1, τ+m)
#         text_len = text_len.cpu().numpy()
#         aspect_len = aspect_len.cpu().numpy()
#         weight = [[] for i in range(batch_size)]
#         for i in range(batch_size):  # 计算qi的权重矩阵
#             context_len = text_len[i] - aspect_len[i]
#             for j in range(aspect_double_idx[i,0]):
#                 weight[i].append(1-(aspect_double_idx[i,0]-j)/context_len)
#             for j in range(aspect_double_idx[i,0], aspect_double_idx[i,1]+1):
#                 weight[i].append(0)
#             for j in range(aspect_double_idx[i,1]+1, text_len[i]):
#                 weight[i].append(1-(j-aspect_double_idx[i,1])/context_len)
#             for j in range(text_len[i], seq_len):
#                 weight[i].append(0)
#         weight = torch.tensor(weight, dtype=torch.float).unsqueeze(2).to(self.opt.device)
#         return weight*x
#
#     def forward(self, inputs, dep_relation_embs, dep_adj, src_mask, aspect_mask):
#         src_mask = ~src_mask.unsqueeze(-2)
#         inputs_norm = self.layer_norm(inputs)
#         adj_mask = dep_adj.eq(0) if dep_adj is not None else None
#         adj_mask = adj_mask.unsqueeze(1)
#         # scores
#         self_attn_score = self.self_attn(inputs_norm, inputs_norm, mask=src_mask)
#         relation_attn_score = self.relation_attn(inputs_norm, dep_relation_embs, inputs_norm, mask=src_mask).transpose(1, 2)
#         attn_score = self_attn_score + relation_attn_score
#         attn_score = attn_score.masked_fill(adj_mask, -1e18)
#         attn = F.softmax(attn_score, dim=-1)
#         attn = self.attn_dropout(attn)
#         # score * value
#         inputs_norm = self.linear_tok_value(inputs_norm).unsqueeze(1)  # tokens' value
#         self_context = torch.matmul(attn, inputs_norm)  # self-attn context
#         dep_relation_embs = self.linear_rel_value(dep_relation_embs)  # relation's value
#         relation_context = torch.matmul(attn.transpose(1, 2), dep_relation_embs)  # relation-attn context
#         context = self_context + relation_context.transpose(1, 2)
#         context = F.relu(context)
#         context = torch.cat([i.squeeze(1) for i in torch.split(context, 1, dim=1)], dim=-1)
#         out = self.gcn_dropout(context) + inputs
#         out = self.feed_forward(out)
#         return out  # ([N, L])