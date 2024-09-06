import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn
from layers.layers import rnn_zero_state


class SyntacticGCN(nn.Module):
    """
    SynGCN
    """
    def __init__(self, args, input_dim, hidden_dim, num_layers):
        super(SyntacticGCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_neighbors_list = num_layers
        self.device = args.device
        self.gcn = nn.ModuleList()
        self.gcn.append(SynGCNLayer(input_dim=input_dim, hidden_dim=hidden_dim, device=args.device,
                                    agg_hidden_method=args.agg_hidden_method,
                                    agg_dropout=args.agg_hidden_dropout,
                                    agg_neighbor_method=args.agg_neighbor_method,
                                    agg_neighbor_dropout=args.agg_neighbor_dropout))
        for index in range(0, num_layers - 1):  # gcn more than 1
            self.gcn.append(SynGCNLayer(input_dim=input_dim, hidden_dim=hidden_dim, device=args.device,
                                        agg_hidden_method=args.agg_hidden_method,
                                        agg_dropout=args.agg_hidden_dropout,
                                        agg_neighbor_method=args.agg_neighbor_method,
                                        agg_neighbor_dropout=args.agg_neighbor_dropout))

    def forward(self, input, dep_adj, adj_dict_list, padding_mask, aspect_mask):
        """Traditional GCN"""
        # denom = torch.sum(dep_adj, dim=-1, keepdim=True) + 1
        # Ax = dep_adj.matmul(input)
        # AxW = self.weight(Ax)
        # AxW = AxW / denom
        # gAxW = F.relu(AxW)
        # return self.gcn_drop(gAxW)
        """SnyGCN"""
        src_nodes = [[[idx for idx, item in enumerate(row.tolist()) if item == 1] for row in aspect_mask]]  # source nodes idx (layer, B, idx)
        src_mask_list, neigh_mask_list = [], []

        # build mask matrix
        for l in range(self.num_layers):
            # the max number of source nodes
            src_node_max_num = max([len(nodes) for nodes in src_nodes[l]])
            src_mask = torch.zeros(input.shape[0], src_node_max_num, input.shape[1], input.shape[-1]).\
                to(self.device)  # (B, src_node_num, maxlen, emb_dim)
            neigh_mask = torch.zeros_like(src_mask).to(self.device)  # (B, src_node_num, maxlen, emb_dim)
            neigh_nodes = []
            for batch in range(input.shape[0]):
                src_nodes_idx = src_nodes[l][batch]
                neigh_nodes_idx = [adj_dict_list[batch].get(idx, []) for idx in src_nodes_idx]
                neigh_nodes.append(list(set([node for neigh_nodes in neigh_nodes_idx for node in
                                             neigh_nodes])))  # flatten source nodes and remove duplicates
                for i, src_idx in enumerate(src_nodes_idx):
                    src_mask[batch, i, src_idx] = 1
                    for j, neigh_idx in enumerate(neigh_nodes_idx):
                        neigh_mask[batch, j, neigh_idx] = 1
            src_nodes.append(neigh_nodes)
            src_mask_list.append(src_mask)
            neigh_mask_list.append(neigh_mask)

        hidden_tensor = input.clone()
        # Aggregation
        for l in range(self.num_layers - 1, -1, -1):
            src_mask = src_mask_list[l]
            neigh_mask = neigh_mask_list[l]
            h = hidden_tensor.unsqueeze(1).expand(hidden_tensor.shape[0], src_mask.shape[1], hidden_tensor.shape[1],
                                                  hidden_tensor.shape[-1])  # (B, src_node_num, maxlen, emb_dim)
            src_node_features = h * src_mask
            neigh_node_features = h * neigh_mask
            # aggregation
            arr_hidden = self.gcn[l](src_node_features, neigh_node_features, src_nodes[l], adj_dict_list)
            # replace the original word embedding by the row index
            output_indicator = 0
            for batch, nodes_lst in enumerate(src_nodes[l]):
                length = len(nodes_lst)
                hidden_tensor[batch][nodes_lst] = arr_hidden[output_indicator: output_indicator + length]
                output_indicator += length

        aspect_feature = hidden_tensor
        return aspect_feature

class SynGCNLayer(nn.Module):
    """
    SynGCNLayer
    """
    def __init__(self, input_dim, hidden_dim, device,
                 activation=F.leaky_relu,
                 normalize=False,
                 agg_neighbor_method="mean",
                 agg_hidden_method="sum",
                 agg_neighbor_dropout=0.1,
                 agg_dropout=0.1):
        super(SynGCNLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        assert agg_neighbor_method in ["mean", "sum", "pooling", "lstm"]
        assert agg_hidden_method in ["sum", "concat"]
        self.agg_hidden_method = agg_hidden_method
        self.activation = activation
        self.normalize = normalize
        self.dropout = nn.Dropout(agg_dropout)
        self.aggregator = NeighborAggregator(input_dim, hidden_dim, device,
                                             agg_neighbor_method,
                                             agg_neighbor_dropout)
        self.weight = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.hidden_map = nn.Linear(hidden_dim * 2, hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)

    def forward(self, src_node_features, neigh_node_features, src_nodes, adj_dict_list):
        def filter_nonsource_nodes(input):
            """
            filter out rows that are all zeros(non source nodes)
            :param input: tensor(B, src_node_num, emb_dim)
            :return:
            """
            # reshape
            hidden = input.view(input.shape[0] * input.shape[1], input.shape[-1])
            # get non-all-zero row indices
            nonzero_indices = torch.nonzero(torch.sum(hidden != 0, dim=1)).squeeze()
            return hidden[nonzero_indices]
        neighbor_hidden = self.aggregator(neigh_node_features, src_nodes, adj_dict_list)
        if self.agg_hidden_method == "sum":
            hidden = src_node_features.sum(dim=2) + neighbor_hidden
            hidden = filter_nonsource_nodes(hidden)
        elif self.agg_hidden_method == "concat":
            # concat(src_feature, neighbor_feature)
            hidden = torch.cat([src_node_features.sum(dim=2), neighbor_hidden], dim=-1)
            hidden = self.hidden_map(filter_nonsource_nodes(hidden))
            hidden = self.activation(hidden)
            if self.normalize:
                hidden = F.normalize(hidden)
            self.dropout(hidden)
        else:
            raise ValueError("Expected sum or concat, got {}".format(self.agg_hidden_method))
        return hidden

class NeighborAggregator(nn.Module):
    def __init__(self, input_dim, output_dim, device,
                 agg_neighbor_method="mean",
                 agg_neighbor_dropout=0.1,
                 use_bias=True):
        """
        Aggregating neighbor nodes
        """
        super(NeighborAggregator, self).__init__()
        self.rnn_bidirect = False
        self.agg_neighbor_method = agg_neighbor_method
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.use_bias = use_bias
        self.rnn_layers = 1
        self.device = device
        self.output_dim = output_dim
        if agg_neighbor_method == 'lstm':
            self.lstm = nn.LSTM(input_dim, output_dim, num_layers=self.rnn_layers, batch_first=True, dropout=agg_neighbor_dropout)
        if agg_neighbor_method == 'pooling':
            self.pooling_fc = nn.Linear(input_dim, output_dim)
            self.poolingRelu = nn.ReLU()
            self.poolingDropout = nn.Dropout(agg_neighbor_dropout)
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def encode_with_rnn(self, rnn_inputs, seq_lens, batch_size):
        h0, c0 = rnn_zero_state(batch_size, self.output_dim, self.rnn_layers, self.device,
                                self.rnn_bidirect)
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True, enforce_sorted=False)
        rnn_outputs, (ht, ct) = self.lstm(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs, (ht, ct)

    def forward(self, neighbor_features, src_nodes, adj_dict_list):
        neighbor_num = torch.sum(~torch.all(neighbor_features == 0, dim=-1), dim=-1)
        if self.agg_neighbor_method == "mean":
            agg_neighbor_hidden = neighbor_features.sum(dim=2) / neighbor_num.unsqueeze(2).clamp(min=1)
        elif self.agg_neighbor_method == "sum":
            agg_neighbor_hidden = neighbor_features.sum(dim=2)
        elif self.agg_neighbor_method == "pooling":
            """
            Max-pooling 
            Max-pooling has the same effect as the Mean-pooling
            """
            (batch_size, src_node_num, maxlen, emb_dim) = neighbor_features.shape
            neigh_features_seq = -9e15 * torch.ones(batch_size * src_node_num, maxlen,
                                             emb_dim).to(self.device)
            neighbor_hidden = neighbor_features.view(batch_size * src_node_num * maxlen,
                                                     emb_dim)
            # Filter out padding neighbor nodes(features with all zero)
            nonzero_indices = torch.nonzero(torch.sum(neighbor_hidden != 0, dim=1)).squeeze()
            neighbor_hidden = neighbor_hidden[nonzero_indices]

            # linear
            pooling_outputs = self.pooling_fc(neighbor_hidden)
            pooling_outputs = self.poolingDropout(self.poolingRelu(pooling_outputs))
            # pooling_outputs = self.poolingRelu(pooling_outputs)
            output_indicator = 0
            for batch, nodes_list in enumerate(src_nodes):
                neighbor_len = [len(adj_dict_list[batch].get(node_idx, [])) for node_idx in nodes_list]
                for i, node_idx in enumerate(nodes_list):

                    neigh_features_seq[(batch * src_node_num) + i][0: neighbor_len[i]] = \
                        pooling_outputs[output_indicator: output_indicator + neighbor_len[i]]
                    output_indicator += neighbor_len[i]

            # max-pooling
            neigh_features_seq = neigh_features_seq.max(dim=1, keepdim=True)[0].squeeze().view(batch_size, src_node_num, -1)
            agg_neighbor_hidden = torch.where(neigh_features_seq == -9e15, torch.tensor(0), neigh_features_seq)
        elif self.agg_neighbor_method == "lstm":
            (batch_size, src_node_num, maxlen, emb_dim) = neighbor_features.shape
            neigh_features_seq = torch.zeros(batch_size * src_node_num, maxlen,  # (B * src_node_num, maxlen, emb_dim)
                                             emb_dim).to(self.device)
            neighbor_hidden = neighbor_features.view(batch_size * src_node_num * maxlen,  # (B * src_node_num * maxlen, emb_dim)
                                                     emb_dim)
            # Filter out padding neighbor nodes(features with all zero)
            nonzero_indices = torch.nonzero(~(neighbor_hidden == 0).all(dim=1), as_tuple=False).squeeze()
            # nonzero_indices = torch.nonzero(torch.sum(neighbor_hidden != 0, dim=1)).squeeze()
            neighbor_hidden = neighbor_hidden[nonzero_indices]
            output_indicator = 0
            neigh_features_len = [0] * batch_size * src_node_num

            for batch, nodes_list in enumerate(src_nodes):
                neighbor_len = [len(adj_dict_list[batch].get(node_idx, [])) for node_idx in nodes_list]
                for i, node_idx in enumerate(nodes_list):
                    # shuffle nodes' sequence
                    shuffled_indices = torch.randperm(neighbor_len[i]).to(self.device)
                    neigh_features_seq[(batch * src_node_num) + i][0: neighbor_len[i]] = \
                        neighbor_hidden[output_indicator: output_indicator + neighbor_len[i]][shuffled_indices]
                    output_indicator += neighbor_len[i]
                neigh_features_len[(batch * src_node_num): len(nodes_list)] = neighbor_len
            # Filter out padding source nodes(features with all zero)
            indicator = {index: value for index, value in enumerate(neigh_features_len) if value != 0}
            neigh_features_seq_filter = neigh_features_seq[list(indicator.keys())]
            # LSTM
            self.lstm.flatten_parameters()
            _, (agg_neighbor, _) = self.encode_with_rnn(neigh_features_seq_filter, list(indicator.values()), len(indicator))
            agg_neighbor = agg_neighbor.squeeze(0)
            # neighbor hidden (B, src_node_num, emb_dim)
            agg_neighbor_hidden = torch.zeros(batch_size, src_node_num, emb_dim).to(self.device)
            # final hidden
            replace_indicator = 0
            for batch, src_nodes_list in enumerate(src_nodes):
                length = len(src_nodes_list)
                seq = [i for i in range(length)]
                agg_neighbor_hidden[batch][seq] = agg_neighbor[replace_indicator: replace_indicator + length]
                replace_indicator += length
        else:
            raise ValueError("Unknown agg type, expected sum, pooling, or mean, but got {}"
                             .format(self.agg_neighbor_method))
        # neighbor_hidden = torch.matmul(agg_neighbor, self.weight)
        # if use_bias:
        #     neighbor_hidden += self.bias
        return agg_neighbor_hidden  # (B, src_node_num, emb_dim)