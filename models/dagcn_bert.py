# coding:utf-8
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from layers.tree import head_to_adj
from layers.syntacticgcn import SyntacticGCN
from layers.semanticgcn import SemanticGCN


class DAGCNBert(nn.Module):
    def __init__(self, bert, args):
        super().__init__()
        hidden_dim = args.bert_dim // 2
        self.args = args
        self.enc = ContextEncoder(bert, args)
        self.classifier = nn.Linear(hidden_dim, args.polarities_dim)

    def forward(self, inputs):
        hiddens = self.enc(inputs)
        logits = self.classifier(hiddens)
        return logits, hiddens


class ContextEncoder(nn.Module):
    def __init__(self, bert, args):
        super().__init__()
        self.bert = bert
        self.args = args
        self.bert_dim = args.bert_dim
        self.hidden_dim = self.bert_dim // 2

        # #################### RNN + GNN Encoding ###################
        self.encoder = DualChannelEncoder(bert, args)
        self.fc = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        # #################### pooling and fusion modules ###################
        self.inp_map = torch.nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        if self.args.output_merge.lower() == "gatenorm":
            self.out_gate_map = torch.nn.Linear(self.hidden_dim * 2, self.hidden_dim)
            self.out_norm = nn.LayerNorm(self.hidden_dim)
        elif self.args.output_merge.lower() == "fc":
            self.norm_fc = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        elif self.args.output_merge.lower() == "biaffine":
            self.affine1 = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
            self.affine2 = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
            self.affine_dropout = nn.Dropout(args.biaffine_dropout)
            self.affine_map = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        else:
            exit(0)
        if self.args.output_merge.lower() != "none":
            self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.eye_(self.inp_map.weight)
        torch.nn.init.zeros_(self.inp_map.bias)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids, \
        attention_mask, asp_start, asp_end, \
        src_mask, aspect_mask,\
        text, head, lengths, deprel = inputs

        maxlen = self.args.max_length
        text = text[:, :maxlen]
        """
        dependency adj and dict
        """
        adj_lst, label_lst, adj_dict_list, asp_idx_list = [], [], [], []
        for idx in range(len(lengths)):
            adj_i, label_i, adj_dict = head_to_adj(maxlen, head[idx], text[idx], deprel[idx], lengths[idx],
                                                   aspect_mask[idx],
                                                   directed=self.args.direct,
                                                   self_loop=self.args.loop)
            adj_lst.append(adj_i.reshape(1, maxlen, maxlen))
            label_lst.append(label_i.reshape(1, maxlen, maxlen))
            adj_dict_list.append(adj_dict)
            asp_idx = [ind for ind in range(len(aspect_mask[idx])) if aspect_mask[idx][ind] == 1]
            asp_idx_list.append(asp_idx)

        dep_adj = torch.from_numpy(np.concatenate(adj_lst, axis=0)).to(self.args.device)  # [B, maxlen, maxlen]

        # GNN Encoding
        syn_out, sem_out, pooled_output = self.encoder(inputs=inputs,
                                                       dep_adj=dep_adj,
                                                       adj_dict_list=adj_dict_list)

        # ###########pooling and fusion #################
        asp_wn = aspect_mask[:, :maxlen].sum(dim=1).unsqueeze(-1)  # aspect count
        mask = aspect_mask[:, :maxlen].unsqueeze(-1).repeat(1, 1, self.hidden_dim)  # mask for h

        if self.args.output_merge.lower() == "gatenorm2":  # gate
            syn_out = (syn_out * mask).sum(dim=1) / asp_wn
            sem_out = (sem_out * mask).sum(dim=1) / asp_wn
            gate = self.out_norm(torch.sigmoid(
                self.out_gate_map(torch.cat([syn_out, sem_out], dim=-1))
            ))  # gatenorm2 merge
            outputs = syn_out * gate + (1 - gate) * sem_out
        elif self.args.output_merge.lower() == "fc":  # fully connect
            syn_out = (syn_out * mask).sum(dim=1) / asp_wn
            sem_out = (sem_out * mask).sum(dim=1) / asp_wn
            outputs = self.norm_fc(torch.cat([syn_out, sem_out], dim=-1))
        elif self.args.output_merge.lower() == "biaffine":  # Biaffine
            A1 = torch.softmax(torch.bmm(torch.matmul(syn_out, self.affine1), torch.transpose(sem_out, 1, 2)),
                               dim=-1)
            A2 = torch.softmax(torch.bmm(torch.matmul(sem_out, self.affine2), torch.transpose(syn_out, 1, 2)),
                               dim=-1)
            gAxW_dep, gAxW_ag = torch.bmm(A1, sem_out), torch.bmm(A2, syn_out)
            outputs_dep = self.affine_dropout(gAxW_dep)
            outputs_ag = self.affine_dropout(gAxW_ag)
            outputs_dep = (outputs_dep * mask).sum(dim=1) / asp_wn
            outputs_ag = (outputs_ag * mask).sum(dim=1) / asp_wn
            outputs = F.relu(self.affine_map(torch.cat((outputs_dep, outputs_ag), dim=-1)))
        else:
            exit(0)
        return outputs


class DualChannelEncoder(nn.Module):
    """
    Syntactic GCN
    Semantic GCN
    """

    def __init__(self, bert, args):
        super(DualChannelEncoder, self).__init__()
        self.bert = bert
        self.args = args
        self.syn_layers = args.syn_layers
        self.sem_layers = args.sem_layers
        self.input_dim = args.bert_dim
        self.hidden_dim = args.bert_dim // 2
        self.layernorm = nn.LayerNorm(args.bert_dim)
        # dropout
        self.bert_drop = nn.Dropout(args.bert_dropout)
        self.pooled_drop = nn.Dropout(args.bert_dropout)
        # SynGCN
        self.syn_gcn = SyntacticGCN(args, self.input_dim, self.input_dim, self.syn_layers).to(args.device)
        # SemGCN
        self.sem_gcn = SemanticGCN(args, self.input_dim, self.sem_layers).to(args.device)

        # output mapping
        self.Wsyn = nn.Linear(self.input_dim, self.hidden_dim)
        self.Wsem = nn.Linear(self.input_dim, self.hidden_dim)

    def forward(self, dep_adj, adj_dict_list, inputs):
        text_bert_indices, bert_segments_ids, \
        attention_mask, asp_start, asp_end, \
        src_mask, aspect_mask, \
        text, head, lengths, deprel = inputs

        # BERT encoding
        sequence_output, pooled_output = self.bert(text_bert_indices, attention_mask=attention_mask,
                                token_type_ids=bert_segments_ids, return_dict=False)
        sequence_output = self.layernorm(sequence_output)
        gcn_inputs = self.bert_drop(sequence_output)
        pooled_output = self.pooled_drop(pooled_output)
        # H from BERT (B, seq_len, dim)
        # get the sentence hidden and pad it to the same length
        inputs = (gcn_inputs[:, 1:, :] * src_mask[:, :-1].unsqueeze(-1).repeat(1, 1, self.input_dim))
        temp_tensor = torch.zeros(gcn_inputs.shape[0], 1, gcn_inputs.shape[-1]).to(self.args.device)
        inputs = torch.cat((inputs, temp_tensor), dim=1)
        # Syntactic GCN
        syn_output = self.syn_gcn(inputs, dep_adj, adj_dict_list, src_mask, aspect_mask)
        # Sematic GCN
        sem_output = self.sem_gcn(inputs, dep_adj, src_mask, aspect_mask)
        syn_output = F.relu(self.Wsyn(syn_output))
        sem_output = F.relu(self.Wsem(sem_output))

        return syn_output, sem_output, pooled_output