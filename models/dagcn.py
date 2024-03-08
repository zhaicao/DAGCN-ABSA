# coding:utf-8
import torch
import numpy as np
import copy
import math
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence

from layers.sampling import multihop_sampling
from layers.tree import head_to_adj, dep_distance_adj
from layers.layers import GraphConvolutionLayer, SimpleGraphConvolutionLayer, PositionwiseFeedForward


class DAGCN(nn.Module):
    def __init__(self, args, emb_matrix):
        super().__init__()
        in_dim = args.hidden_dim
        self.args = args
        self.enc = ContextEncoder(args, emb_matrix=emb_matrix)
        self.classifier = nn.Linear(in_dim, args.polarities_dim)

    def forward(self, inputs):
        hiddens = self.enc(inputs)
        logits = self.classifier(hiddens)
        return logits, hiddens


class ContextEncoder(nn.Module):
    def __init__(self, args, emb_matrix=None):
        super().__init__()
        self.args = args
        self.emb_matrix = emb_matrix

        # #################### Embeddings ###################
        self.tok_emb = nn.Embedding.from_pretrained(torch.tensor(emb_matrix, dtype=torch.float),
                                                    freeze=True)  # Word emb
        self.pos_emb = nn.Embedding(args.pos_size, args.pos_dim, padding_idx=0) if args.pos_dim > 0 else None  # POS emb
        self.post_emb = nn.Embedding(args.post_size, args.post_dim,
                                     padding_idx=0) if args.post_dim > 0 else None  # position emb
        self.dep_emb = nn.Embedding(args.dep_size, args.dep_dim,
                                    padding_idx=0) if args.dep_dim > 0 else None  # dependent relation emb
        self.deppost_emb = nn.Embedding(args.deppost_size, args.deppost_dim,
                                        padding_idx=0) if args.deppost_dim > 0 else None  # dependent position emb

        # #################### GNN+Attention Encoding ###################
        embeddings = (self.tok_emb, self.pos_emb, self.post_emb, self.dep_emb, self.deppost_emb)
        self.encoder = DualChannelEncoder(args, embeddings, args.hidden_dim, args.num_layers)
        self.fc = nn.Linear(args.hidden_dim * 2 * 2, args.hidden_dim * 2)

        # #################### Pooling and fusion modules ###################
        self.inp_map = torch.nn.Linear(args.hidden_dim * 2, args.hidden_dim)
        if self.args.output_merge.lower() == "gatenorm2":
            self.out_gate_map = torch.nn.Linear(args.hidden_dim * 2, args.hidden_dim)
            self.out_norm = nn.LayerNorm(args.hidden_dim)
        elif self.args.output_merge.lower() == "fc":
            self.norm_fc = nn.Linear(args.hidden_dim * 2, args.hidden_dim)
        elif self.args.output_merge.lower() == "biaffine":
            self.affine1 = nn.Parameter(torch.Tensor(args.hidden_dim, args.hidden_dim))
            self.affine2 = nn.Parameter(torch.Tensor(args.hidden_dim, args.hidden_dim))
            self.affine_dropout = nn.Dropout(args.gcn_dropout)
            self.affine_map = nn.Linear(args.hidden_dim * 2, args.hidden_dim)
        else:
            exit(0)
        if self.args.output_merge.lower() != "none":
            self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.eye_(self.inp_map.weight)
        torch.nn.init.zeros_(self.inp_map.bias)

    def forward(self, inputs):
        tok, asp, pos, head, deprel, post, aspect_mask, lengths, adj = inputs  # unpack inputs
        maxlen = max(lengths.data)
        tok = tok[:, :maxlen]
        """
        dependency, dependent relation, dependent relative distance adj
        """
        adj_lst, label_lst, aspect_samples, aspect_samples_maxlen = [], [], [], []
        for idx in range(len(lengths)):
            adj_i, label_i, adj_dict = head_to_adj(maxlen, head[idx], tok[idx], deprel[idx], lengths[idx], aspect_mask[idx],
                                         tok[idx],
                                         directed=self.args.direct,
                                         self_loop=self.args.loop)
            adj_lst.append(adj_i.reshape(1, maxlen, maxlen))
            label_lst.append(label_i.reshape(1, maxlen, maxlen))
            asp_idx = [ind for ind in range(len(aspect_mask[idx])) if aspect_mask[idx][ind] == 1]
            sample_idx, max_len = multihop_sampling(asp_idx, self.args.num_layers, adj_dict)  # sage sampling
            aspect_samples.append(sample_idx)
            aspect_samples_maxlen.append(max_len)
            # adj_dist = dep_distance_adj(adj_i, dep_post[idx], mask_ori[idx], lengths[idx], maxlen)
            # dist_lst.append(adj_dist.reshape(1, maxlen, maxlen))
            # asp_idx.append([id for id in range(len(aspect_mask[idx])) if aspect_mask[idx][id] == 1])

        dep_adj = torch.from_numpy(np.concatenate(adj_lst, axis=0)).to(self.args.device)  # [B, maxlen, maxlen]
        relation_adj = torch.from_numpy(np.concatenate(label_lst, axis=0)).to(self.args.device)  # [B, maxlen, maxlen]
        # 依存相对位置 k-hops [B, maxlen, maxlen]
        # deppost_adj = dep_post[:, :maxlen, :maxlen].add(torch.eye(maxlen)).to(self.args.device)
        # 依存相对位置距离
        # dep_dist_adj = torch.from_numpy(np.concatenate(dist_lst, axis=0)).to(self.args.device)
        # dep_dist_adj = F.softmax(dep_dist_adj, dim=-1)

        # GNN Encoding
        syn_out, sem_out = self.encoder(dep_adj=dep_adj, rel_adj=relation_adj, inputs=inputs, lengths=lengths,
                                           aspect_samples=aspect_samples, aspect_samples_maxlen=aspect_samples_maxlen,
                                           position_adj=None)

        # ###########pooling and fusion #################
        asp_wn = aspect_mask[:, :maxlen].sum(dim=1).unsqueeze(-1)  # num of aspects
        mask = aspect_mask[:, :maxlen].unsqueeze(-1).repeat(1, 1, self.args.hidden_dim)  # mask for h

        if self.args.output_merge.lower() == "gatenorm2":  # gate
            sem_out = (sem_out * mask).sum(dim=1) / asp_wn  # aspect-aware attention
            syn_out = syn_out.squeeze(1)
            gate = self.out_norm(torch.sigmoid(
                self.out_gate_map(torch.cat([syn_out, sem_out], dim=-1))
            ))  # gatenorm2 merge
            outputs = sem_out * gate + (1 - gate) * sem_out
        elif self.args.output_merge.lower() == "fc":  # fully connect
            sem_out = (sem_out * mask).sum(dim=1) / asp_wn  # aspect-aware attention
            syn_out = syn_out.squeeze(1)
            outputs = self.norm_fc(torch.cat([syn_out, sem_out], dim=-1))
        elif self.args.output_merge.lower() == "biaffine":  # Biaffine
            A1 = torch.softmax(torch.bmm(torch.matmul(sem_out, self.affine1), torch.transpose(syn_out, 1, 2)),
                               dim=-1)
            A2 = torch.softmax(torch.bmm(torch.matmul(syn_out, self.affine2), torch.transpose(sem_out, 1, 2)),
                               dim=-1)
            gAxW_dep, gAxW_ag = torch.bmm(A1, syn_out), torch.bmm(A2, sem_out)
            outputs_dep = self.affine_dropout(gAxW_dep)
            outputs_ag = self.affine_dropout(gAxW_ag)
            outputs_dep = (outputs_dep * mask).sum(dim=1) / asp_wn
            # outputs_ag = (outputs_ag * mask).sum(dim=1) / asp_wn
            outputs_ag = outputs_ag.squeeze(1)  # GraphSAGE
            outputs = F.relu(self.affine_map(torch.cat((outputs_dep, outputs_ag), dim=-1)))
        else:
            exit(0)
        return outputs


class DualChannelEncoder(nn.Module):
    """
    Aspect-aware dependency Encoding
    Aspect-aware attention Encoding
    """
    def __init__(self, args, embeddings, mem_dim, num_layers):
        super(DualChannelEncoder, self).__init__()
        self.args = args
        self.layers = num_layers
        self.mem_dim = mem_dim
        self.in_dim = args.embed_dim + args.post_dim + args.pos_dim
        self.tok_emb, self.pos_emb, self.post_emb, self.dep_emb, self.deppost_emb = embeddings
        # Sentence Encoder
        rnn_input_dim = self.in_dim
        self.Sent_encoder = nn.LSTM(rnn_input_dim, args.rnn_hidden, args.rnn_layers, batch_first=True, \
                                    dropout=args.rnn_dropout, bidirectional=args.bidirect)
        if args.bidirect:
            self.in_dim = args.rnn_hidden * 2
        else:
            self.in_dim = args.rnn_hidden
        # dropout
        self.rnn_drop = nn.Dropout(args.rnn_dropout)
        self.in_drop = nn.Dropout(args.input_dropout)
        # Selective attention
        self.sa_gcn = SelectiveAttentionGCNLayer(args)
        # SAGE
        self.sage = SAGELayer(args, self.in_dim, self.in_dim, args.num_layers).to(args.device)
        # self.gcn2 = SRDGCNLayer(args, self.in_dim, self.in_dim, self.args.gcn_dropout).to(args.device)
        # self.gat_deprel1 = SRDGCNLayer(args, self.in_dim, self.in_dim, self.args.gcn_dropout).to(args.device)
        # self.gat_deprel2 = SRDGCNLayer(args, self.in_dim, self.in_dim, self.args.gcn_dropout).to(args.device)
        # self.gcn1 = SimpleGraphConvolutionLayer(args.hidden_dim * 2, args.hidden_dim * 2)
        # self.gcn2 = SimpleGraphConvolutionLayer(args.hidden_dim * 2, args.hidden_dim * 2)
        self.deprel_dense = nn.Linear(args.rnn_hidden * 2 * 2, args.rnn_hidden * 2)
        self.dep_att = MultiHeadAttention(args.attention_heads, args.rnn_hidden * 2)
        # output mapping
        self.Wsyn = nn.Linear(args.rnn_hidden * 2, args.rnn_hidden)
        self.Wsem = nn.Linear(args.rnn_hidden * 2, args.rnn_hidden)

    def encode_with_rnn(self, rnn_inputs, seq_lens, batch_size):
        h0, c0 = rnn_zero_state(batch_size, self.args.rnn_hidden, self.args.rnn_layers, self.args.device,
                                self.args.bidirect)
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True, enforce_sorted=False)
        rnn_outputs, (ht, ct) = self.Sent_encoder(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs, (ht, ct)

    def forward(self, dep_adj, rel_adj, inputs, lengths, aspect_samples, aspect_samples_maxlen, position_adj=None):
        tok, asp, pos, head, deprel, post, aspect_mask, seq_len, _ = inputs
        maxlen = max(lengths.data)
        key_padding_mask = sequence_mask(lengths) if lengths is not None else None  # padding mask  # [B, seq_len]
        aspect_mask = aspect_mask[:, :maxlen]
        # embedding(tok, pos, post)
        word_embs = self.tok_emb(tok)
        embs = [word_embs]
        if self.args.pos_dim > 0:
            embs += [self.pos_emb(pos)]
        if self.args.post_dim > 0:
            embs += [self.post_emb(post)]
        embs = torch.cat(embs, dim=2)
        embs = self.in_drop(embs)

        # BiLSTM encoding
        self.Sent_encoder.flatten_parameters()  # improving efficiency of storage and computing
        rnn_output, _ = self.encode_with_rnn(embs, seq_len, tok.size()[0])
        rnn_output = self.rnn_drop(rnn_output)  # [B, seq_len, H]
        # rel_adj_embs = self.dep_emb(rel_adj) if rel_adj is not None else None  # relation embedding [B, seq, seq, dim]
        input = rnn_output  # Hidden from BiLSTM (B, seq_len, H)
        # Syntactic GCN
        """
        traditional GCN
        """
        # syn_output = F.relu(self.gcn1(input, dep_adj))
        # syn_output = F.relu(self.gcn2(syn_output, dep_adj))
        """
        added distance weight
        """
        syn_output = self.sage(input, dep_adj, aspect_samples, aspect_samples_maxlen, key_padding_mask, aspect_mask)
        # syn_output = self.gcn1(input, dep_adj, key_padding_mask, aspect_mask)
        # syn_output = self.gcn2(syn_output, dep_adj, key_padding_mask, aspect_mask)
        # syn_output = self.gat_deprel1(input, rel_adj_embs, dep_adj, key_padding_mask, aspect_mask)
        # syn_output = self.gat_deprel2(syn_output, rel_adj_embs, dep_adj, key_padding_mask, aspect_mask)
        # Semantic GAT
        sem_output = self.sa_gcn(input, dep_adj, key_padding_mask, aspect_mask)

        syn_output = F.relu(self.Wsyn(syn_output))
        sem_output = F.relu(self.Wsem(sem_output))

        return syn_output, sem_output

"""Start SAGE"""

class SAGELayer(nn.Module):
    """
    GraphSAGE
    """

    def __init__(self, args, in_dim, hidden_dim, num_layers):
        super(SAGELayer, self).__init__()
        self.args = args
        self.input_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_neighbors_list = args.num_layers
        self.gcn = nn.ModuleList()  # at least two layers
        self.gcn.append(SageGCN(args, in_dim, hidden_dim,
                                aggr_neighbor_method=args.aggr_neighbor_method,
                                aggr_hidden_method=args.aggr_hidden_method))
        for index in range(0, num_layers - 2):  # gcn more than 2
            self.gcn.append(SageGCN(hidden_dim, hidden_dim,
                                    aggr_neighbor_method=args.aggr_neighbor_method,
                                    aggr_hidden_method=args.aggr_hidden_method))
        self.gcn.append(SageGCN(args, hidden_dim, hidden_dim, activation=None))

    def forward(self, input, dep_adj, aspect_samples, aspect_samples_maxlen, key_padding_mask, aspect_mask):
        """GCN"""
        # denom = torch.sum(dep_adj, dim=-1, keepdim=True) + 1
        # Ax = dep_adj.matmul(input)
        # AxW = self.weight(Ax)
        # AxW = AxW / denom
        # gAxW = F.relu(AxW)
        # return self.gcn_drop(gAxW)
        """SAGE"""
        hidden_tensor = torch.empty(input.size(0), 1, self.args.rnn_hidden * 2)
        # src_nodes and neighbor_nodes embedding
        for i, batch in enumerate(aspect_samples):  # batch
            emb_layers = []
            for layer in batch:
                emb_nodes = torch.empty(len(layer), self.args.rnn_hidden * 2)
                for j, node in enumerate(layer):
                    if node == -1:
                        emb_nodes[j] = torch.zeros(self.args.rnn_hidden * 2, dtype=torch.float32)  # padding index set zero tensor
                    else:
                        emb_nodes[j] = input[i][node]
                emb_layers.append(emb_nodes)

            hidden = emb_layers
            for l in range(self.num_layers):
                next_hidden = []
                gcn = self.gcn[l]
                for hop in range(self.num_layers - l):
                    src_node_features = hidden[hop]  # source nodes
                    src_node_num = len(src_node_features)
                    # neighbor nodes feature
                    neighbor_node_features = hidden[hop + 1].view((src_node_num, aspect_samples_maxlen[i], -1))
                    # neighbor nodes indices
                    neighbor_node_idx = torch.from_numpy(batch[hop + 1]).view((src_node_num, aspect_samples_maxlen[i]))
                    h = gcn(src_node_features, neighbor_node_features, neighbor_node_idx)  # aggregation
                    next_hidden.append(h)
                hidden = next_hidden
            if len(hidden[0]) != 1:  # words of a aspect more than 1
                hidden[0] = hidden[0].mean(dim=0, keepdim=True)  # mean pooling on aspects
                # hidden[0] = hidden[0].max(dim=0, keepdim=True)  # max pooling on aspects
            hidden_tensor[i] = hidden[0]
        return hidden_tensor

class SageGCN(nn.Module):
    def __init__(self, args, input_dim, hidden_dim,
                 activation=F.leaky_relu,
                 normalize=True,
                 aggr_neighbor_method="mean",
                 aggr_hidden_method="sum"):
        """SageGCN
        Args:
            input_dim: input feature dimension
            hidden_dim: hidden or output dimension
                when aggr_hidden_method=sum, output dim is hidden_dim
                when aggr_hidden_method=concat, output dim is hidden_dim*2
            activation: activition function
            aggr_neighbor_method: the method for aggregating neighbor nodes,["mean", "sum", "max", "lstm"]
            aggr_hidden_method: the method for updating the source nodes,["sum", "concat"]
        """
        super(SageGCN, self).__init__()
        assert aggr_neighbor_method in ["mean", "sum", "pooling", "lstm"]
        assert aggr_hidden_method in ["sum", "concat"]
        self.args = args
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.aggr_neighbor_method = aggr_neighbor_method
        self.aggr_hidden_method = aggr_hidden_method
        self.aggr_neighbor_dropout = args.aggr_neighbor_dropout
        self.activation = activation
        self.normalize = normalize
        self.aggregator = NeighborAggregator(args, input_dim, hidden_dim,
                                             aggr_method=aggr_neighbor_method,
                                             dropout=args.aggr_neighbor_dropout)
        self.weight = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.hidden_map = nn.Linear(hidden_dim * 2, hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)

    def forward(self, src_node_features, neighbor_node_features, neighbor_node_idx):
        seq_len = (~neighbor_node_idx.eq(-1)).sum(dim=-1)  # seq_len
        if 0 in seq_len:  # padding nodes
            unPadding_node = torch.where(~seq_len.eq(0))  # unPadding nodes feature(not 0 in seq_len)
            avail_neighbor_feature = neighbor_node_features[unPadding_node]
            avail_src_node_features = src_node_features[unPadding_node]
            avail_neighbor_seq_len = seq_len[~seq_len.eq(0)]  # 获取seq非0项的seq_len
            neighbor_hidden = self.aggregator(avail_neighbor_feature, avail_neighbor_seq_len)
            self_hidden = torch.matmul(avail_src_node_features, self.weight)
        else:
            neighbor_hidden = self.aggregator(neighbor_node_features, seq_len)
            self_hidden = torch.matmul(src_node_features, self.weight)

        if self.aggr_hidden_method == "sum":
            hidden = self_hidden + neighbor_hidden
        elif self.aggr_hidden_method == "concat":
            hidden = torch.cat([self_hidden, neighbor_hidden], dim=-1)
            hidden = self.hidden_map(hidden)
        else:
            raise ValueError("Expected sum or concat, got {}".format(self.aggr_hidden))
        if self.activation:  # default relu
            hidden = self.activation(hidden)
        # if self.normalize:
        #     hidden = F.normalize(hidden)
        # aligning hidden tensor
        if hidden.size(0) != src_node_features.size(0):
            insert_indices = torch.where(seq_len.eq(0))[0]
            for idx in insert_indices:
                hidden = torch.cat((hidden[:idx], torch.zeros(1, hidden.size(-1)), hidden[idx:]), dim=0)
        return hidden

    def extra_repr(self):
        output_dim = self.hidden_dim if self.aggr_hidden_method == "sum" else self.hidden_dim * 2
        return 'in_features={}, out_features={}, aggr_hidden_method={}'.format(
            self.input_dim, output_dim, self.aggr_hidden_method)

class NeighborAggregator(nn.Module):
    def __init__(self, args, input_dim, output_dim,
                 use_bias=True, aggr_method="mean", dropout=0.1):
        """Aggregating neighbor nodes
        Args:
            input_dim: input feature dimension
            hidden_dim: hidden or output dimension
            use_bias: using bias (default: {False})
            aggr_method: the method for aggregating neighbor nodes (default: {mean})
        """
        super(NeighborAggregator, self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.rnn_layers = 1
        self.rnn_bidirect = False
        self.aggr_method = aggr_method
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.lstm = nn.LSTM(input_dim, output_dim, num_layers=self.rnn_layers, batch_first=True, dropout=dropout)
        if aggr_method == 'pooling':
            self.pooling_fc = nn.Linear(input_dim, output_dim)
            self.poolingRelu = nn.ReLU()
            self.poolingDropout = nn.Dropout(dropout)
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def encode_with_rnn(self, rnn_inputs, seq_lens, batch_size, encoding='lstm'):
        h0, c0 = rnn_zero_state(batch_size, self.output_dim, self.rnn_layers, self.args.device,
                                self.rnn_bidirect)
        ht, ct = h0, c0
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True, enforce_sorted=False)
        if encoding == 'lstm':
            rnn_outputs, (ht, ct) = self.lstm(rnn_inputs, (h0, c0))
        else:
            outputs = self.pooling_fc(rnn_inputs.data)
            rnn_outputs = PackedSequence(self.poolingDropout(self.poolingRelu(outputs)),
                                         rnn_inputs.batch_sizes)
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs, (ht, ct)

    def forward(self, neighbor_feature, seq_len):
        if self.aggr_method == "mean":
            neighbor_num = seq_len.unsqueeze(-1)  # num of neighbors
            aggr_neighbor = neighbor_feature.sum(dim=1) / neighbor_num
        elif self.aggr_method == "sum":
            aggr_neighbor = neighbor_feature.sum(dim=1)
        elif self.aggr_method == "pooling":
            """
            Max-pooling 
            Max-pooling has the same effect as the Mean-pooling
            """
            neighbor_pooling, (_, _) = self.encode_with_rnn(neighbor_feature, seq_len, len(neighbor_feature), encoding='pooling')
            neighbor_max_list = [neighbor_pooling[i, :length, :].max(dim=0).values.unsqueeze(0) for i, length in enumerate(seq_len)]
            aggr_neighbor = torch.cat(neighbor_max_list, dim=0)
            # mean-pooling
            # neighbor_num = seq_len.unsqueeze(-1)  # num of neighbors
            # aggr_neighbor = neighbor_pooling.sum(dim=1) / neighbor_num
        elif self.aggr_method == "lstm":
            self.lstm.flatten_parameters()
            _, (aggr_neighbor, _) = self.encode_with_rnn(neighbor_feature, seq_len, len(neighbor_feature))
            aggr_neighbor = aggr_neighbor.squeeze(0)
        else:
            raise ValueError("Unknown aggr type, expected sum, pooling, or mean, but got {}"
                             .format(self.aggr_method))

        neighbor_hidden = torch.matmul(aggr_neighbor, self.weight)
        if self.use_bias:
            neighbor_hidden += self.bias

        return neighbor_hidden

    def extra_repr(self):
        return 'in_features={}, out_features={}, aggr_method={}'.format(
            self.input_dim, self.output_dim, self.aggr_method)

"""End SAGE"""

class SelectiveAttentionGCNLayer(nn.Module):
    """
    SA-GCN(Adj depends on the attention score)
    """

    def __init__(self, args):
        super(SelectiveAttentionGCNLayer, self).__init__()
        self.args = args
        self.gcn_dim = (args.rnn_hidden * 2) // args.attention_heads
        self.gat_gate = nn.LSTM(args.rnn_hidden * 2, args.rnn_hidden, 1, batch_first=True,
                                bidirectional=args.bidirect, dropout=args.rnn_dropout)
        self.gcn = SimpleGraphConvolutionLayer(self.gcn_dim, self.gcn_dim, args.gcn_dropout)
        self.attn = MultiHeadAttention(args.attention_heads, args.rnn_hidden * 2, aspect_aware=True,
                                       dropout=args.att_dropout)
        self.dense = nn.Linear(args.rnn_hidden * 2, self.gcn_dim)
        self.gat_w = nn.Parameter(torch.Tensor(self.gcn_dim, self.gcn_dim))
        self.gat_dropout = nn.Dropout(args.gcn_dropout)
        self.leakyrelu = nn.LeakyReLU(args.gat_alpha)
        self.layerNorm = nn.LayerNorm(args.rnn_hidden * 2)
        self.bias = nn.Parameter(torch.FloatTensor(args.num_layers, self.gcn_dim))

    def forward(self, input, adj, src_mask, aspect_mask):
        src_mask = ~src_mask.unsqueeze(-2)
        # aspect feature
        aspect_mask = aspect_mask.unsqueeze(-1).repeat(1, 1, self.args.hidden_dim * 2)  # mask for aspect
        aspect = (input * aspect_mask).mean(dim=1)  # aspects average pooling
        # aspect_GAT
        for i in range(self.args.attn_layers):
            # att_adj = self.selectTopK(input, aspect, src_mask, i + 1)  # Top-k selection
            att_adj = self.attn(input, input, aspect, src_mask)  # aspect aware
            input = input.unsqueeze(1).repeat(1, self.args.attention_heads, 1, 1)  # (B, att_heads, seq_len, hidden_dim)
            input = self.dense(input)  # FC:hidden_dim->hidden_dim/heads (B, att_heads, seq_len, hidden_dim/heads)
            # input = F.relu(self.gcn(input, att_adj))  # GCN + attention
            # input = torch.matmul(input, self.gat_w)
            # denom = torch.sum(adj, dim=-1, keepdim=True) + 1
            # input = torch.matmul(att_adj, input) / denom + self.bias[i]
            input = torch.matmul(att_adj, input) + self.bias[i]
            input = F.relu(input)
            input = torch.cat([i.squeeze(1) for i in torch.split(input, 1, dim=1)], dim=-1)  # cat multi_heads
            # input = F.selu(self.gat_w(input))
            # input = self.gat_dropout(input)

        # input, (ht, ct) = self.gat_gate(input, (ht, ct))  # RNN as gate
        return input

    def selectTopK(self, input, aspect, src_mask, layers):
        """
        select top-k scores
        """
        attn = self.attn(input, input, aspect, src_mask)
        if layers != 1:  #  better performance when K more than 3
            probability = F.softmax(attn.sum(dim=(-2, -1)), dim=0)
            max_idx = torch.argmax(probability, dim=1)
            attn_sum = torch.stack([attn[i][max_idx[i]] for i in range(len(max_idx))], dim=0)
        else:
            attn_sum = torch.sum(attn, dim=1)
        att_tensor = select(attn_sum, self.args.top_k).unsqueeze(1).repeat(1, self.args.attention_heads, 1, 1) * attn
        return F.softmax(att_tensor, dim=-1)

def rnn_zero_state(batch_size, hidden_dim, num_layers, device, bidirectional=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    return h0.to(device), c0.to(device)


def sequence_mask(lengths, max_len=None):
    """
    create a boolean mask from sequence length `[batch_size, 1, seq_len]`
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return torch.arange(0, max_len, device=lengths.device).type_as(lengths).unsqueeze(0).expand(
        batch_size, max_len
    ) >= (lengths.unsqueeze(1))

def select(matrix, top_num):
    batch = matrix.size(0)
    # top_k_matrix = top_k_matrix.reshape(batch, -1)
    maxk, indices = torch.topk(matrix, top_num, dim=-1)  # 最后一维选择Topk
    # the weights of position in K is set 1 and others 0
    x = torch.zeros_like(matrix)
    eye = torch.ones_like(matrix)
    top_k_matrix = x.scatter_(dim=-1, index=indices, src=eye)
    # selfloop
    for i in range(batch):
        top_k_matrix[i].fill_diagonal_(1)

    return top_k_matrix

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, aspect_aware=False, relation_aware=False, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
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