import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


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
        self.weight = nn.Parameter(torch.Tensor(hidden_dim + embed_dim))

    def forward(self, h, aspect):
        hx = self.w_h(h)
        vx = self.w_v(aspect)
        hv = torch.tanh(torch.cat((hx, vx), dim=-1))
        ax = torch.unsqueeze(F.softmax(torch.matmul(hv, self.weight), dim=-1), dim=1)
        rx = torch.squeeze(torch.bmm(ax, h), dim=1)
        hn = h[:, -1, :]
        hs = torch.tanh(self.w_p(rx) + self.w_x(hn))
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
            self.weight = nn.Parameter(torch.Tensor(hidden_dim * 2))
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
        e = self.leakyrelu(Wh1 + Wh2.transpose(1, 2))  # (N, N)，
        padding = -9e15 * torch.ones_like(e)  # (N, N),
        attention = torch.where(adj > 0, e, padding)  # (N, N)，
        attention = F.softmax(attention, dim=1)  # (N, N)
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
        """syntactic dependency distance weights"""
        # Ax = dep_dist_adj.matmul(input)
        # AxW = self.weight(Ax)
        # gAxW = F.relu(AxW)
        """inter-word distance weights"""
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
        """traditional GCN"""
        denom = torch.sum(adj, dim=-1, keepdim=True) + 1
        Ax = adj.matmul(input)
        AxW = self.weight(Ax)
        AxW = AxW / denom
        gAxW = F.relu(AxW)
        return self.gcn_drop(gAxW)


def rnn_zero_state(batch_size, hidden_dim, num_layers, device, bidirectional=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    return h0.to(device), c0.to(device)


def sequence_mask(lengths, max_len=None):
    """
    create a mask from sequence length `[batch_size, 1, seq_len]`
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    src_mask = torch.arange(0, max_len, device=lengths.device).type_as(lengths).unsqueeze(0).expand(
        batch_size, max_len
    ) >= (lengths.unsqueeze(1))
    return (~src_mask).to(torch.int)


def select(matrix, top_num):
    batch = matrix.size(0)
    # top_k_matrix = top_k_matrix.reshape(batch, -1)
    maxk, indices = torch.topk(matrix, top_num, dim=-1)
    # the weights of position in K is set 1 and others 0
    x = torch.zeros_like(matrix)
    top_matrix = torch.ones_like(matrix)
    top_k_matrix = x.scatter_(dim=-1, index=indices, src=top_matrix)
    top_k_matrix = top_matrix
    # selfloop
    for i in range(batch):
        top_k_matrix[i].fill_diagonal_(1)
    return top_k_matrix


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])