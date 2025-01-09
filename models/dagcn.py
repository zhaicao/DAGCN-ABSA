# coding:utf-8
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from layers.tree import head_to_adj

from layers.syntacticgcn import SyntacticGCN
from layers.semanticgcn import SemanticGCN
from layers.layers import rnn_zero_state, sequence_mask


class DAGCN(nn.Module):
    def __init__(self, args, emb_matrix):
        super().__init__()
        self.hidden_dim = args.hidden_dim
        self.args = args
        self.enc = ContextEncoder(args, emb_matrix=emb_matrix)
        self.classifier = nn.Linear(self.hidden_dim, args.polarities_dim)

    def forward(self, inputs):
        hiddens = self.enc(inputs)
        logits = self.classifier(hiddens)
        return logits, hiddens


class ContextEncoder(nn.Module):
    def __init__(self, args, emb_matrix=None):
        super().__init__()
        self.args = args
        self.emb_matrix = emb_matrix
        self.hidden_dim = args.hidden_dim

        # #################### Embeddings ###################
        self.tok_emb = nn.Embedding.from_pretrained(torch.tensor(emb_matrix, dtype=torch.float), freeze=True)  # Word emb
        self.pos_emb = nn.Embedding(self.args.pos_size, self.args.pos_dim, padding_idx=0) if self.args.pos_dim > 0 else None  # POS emb
        self.post_emb = nn.Embedding(self.args.post_size, self.args.post_dim,
                                     padding_idx=0) if self.args.post_dim > 0 else None  # position emb
        self.dep_emb = nn.Embedding(self.args.dep_size, self.args.dep_dim,
                                    padding_idx=0) if self.args.dep_dim > 0 else None  # dependent relation emb

        # #################### Syn+Sem Encoding ###################
        embeddings = (self.tok_emb, self.pos_emb, self.post_emb, self.dep_emb)
        self.encoder = DualChannelEncoder(self.args, embeddings)

        # #################### Pooling and fusion modules ###################
        self.inp_map = torch.nn.Linear(args.hidden_dim * 2, args.hidden_dim)
        if self.args.output_merge.lower() == "gatenorm":
            self.out_gate_map = torch.nn.Linear(args.hidden_dim * 2, args.hidden_dim)
            self.out_norm = nn.LayerNorm(args.hidden_dim)
        elif self.args.output_merge.lower() == "fc":
            self.norm_fc = nn.Linear(args.hidden_dim * 2, args.hidden_dim)
        elif self.args.output_merge.lower() == "biaffine":
            self.affine1 = nn.Parameter(torch.Tensor(args.hidden_dim, args.hidden_dim))
            self.affine2 = nn.Parameter(torch.Tensor(args.hidden_dim, args.hidden_dim))
            self.affine_dropout = nn.Dropout(args.biaffine_dropout)
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
        dependency adj and dict
        """
        adj_lst, label_lst, adj_dict_list, asp_idx_list = [], [], [], []
        for idx in range(len(lengths)):
            adj_i, label_i, adj_dict = head_to_adj(maxlen, head[idx], tok[idx], deprel[idx], lengths[idx], aspect_mask[idx],
                                         directed=self.args.direct,
                                         self_loop=self.args.loop)
            adj_lst.append(adj_i.reshape(1, maxlen, maxlen))
            label_lst.append(label_i.reshape(1, maxlen, maxlen))
            adj_dict_list.append(adj_dict)
            asp_idx = [ind for ind in range(len(aspect_mask[idx])) if aspect_mask[idx][ind] == 1]
            asp_idx_list.append(asp_idx)

        dep_adj = torch.from_numpy(np.concatenate(adj_lst, axis=0)).to(self.args.device)  # [B, maxlen, maxlen]

        # GCN Encoding
        syn_out, sem_out = self.encoder(inputs=inputs,
                                        dep_adj=dep_adj,
                                        adj_dict_list=adj_dict_list,
                                        lengths=lengths)

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
            outputs_dep = (outputs_dep * mask).sum(dim=1) / asp_wn  # mean pooling
            outputs_ag = (outputs_ag * mask).sum(dim=1) / asp_wn
            outputs_ag = outputs_ag.squeeze(1)
            outputs = F.relu(self.affine_map(torch.cat((outputs_dep, outputs_ag), dim=-1)))
        else:
            exit(0)
        return outputs


class DualChannelEncoder(nn.Module):
    """
    Syntactic GCN
    Semantic GCN
    """
    def __init__(self, args, embeddings):
        super(DualChannelEncoder, self).__init__()
        self.args = args
        self.syn_layers = args.syn_layers
        self.sem_layers = args.sem_layers
        self.input_dim = args.embed_dim + args.post_dim + args.pos_dim
        self.hiddden_dim = args.hidden_dim
        self.tok_emb, self.pos_emb, self.post_emb, self.dep_emb = embeddings

        self.rnn_layers = args.rnn_layers
        # Sentence Encoder
        self.sent_encoder = nn.LSTM(self.input_dim, self.hiddden_dim, args.rnn_layers, batch_first=True, \
                                    dropout=args.rnn_dropout, bidirectional=args.bidirect)
        if args.bidirect:
            self.input_dim = self.hiddden_dim * 2
        else:
            self.input_dim = self.hiddden_dim
        # dropout
        self.rnn_drop = nn.Dropout(args.rnn_dropout)
        self.in_drop = nn.Dropout(args.input_dropout)
        # SynGCN
        self.syn_gcn = SyntacticGCN(args, self.input_dim, self.input_dim, self.syn_layers).to(args.device)
        # SemGCN
        self.sem_gcn = SemanticGCN(args, self.input_dim, self.sem_layers).to(args.device)

        # self.gcn1 = SimpleGraphConvolutionLayer(args.hidden_dim * 2, args.hidden_dim * 2)
        # self.gcn2 = SimpleGraphConvolutionLayer(args.hidden_dim * 2, args.hidden_dim * 2)

        # output mapping
        self.Wsyn = nn.Linear(args.hidden_dim * 2, args.hidden_dim)
        self.Wsem = nn.Linear(args.hidden_dim * 2, args.hidden_dim)

    def encode_with_rnn(self, rnn_inputs, seq_lens, batch_size):
        h0, c0 = rnn_zero_state(batch_size, self.hiddden_dim, self.rnn_layers, self.args.device,
                                self.args.bidirect)
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True, enforce_sorted=False)
        rnn_outputs, (ht, ct) = self.sent_encoder(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs, (ht, ct)

    def forward(self, dep_adj, adj_dict_list, inputs, lengths):
        tok, asp, pos, head, deprel, post, aspect_mask, seq_len, _ = inputs
        maxlen = max(lengths.data)
        src_mask = sequence_mask(lengths) if lengths is not None else None  # padding mask  # [B, seq_len]
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
        self.sent_encoder.flatten_parameters()  # improving efficiency of storage and computing
        rnn_output, _ = self.encode_with_rnn(embs, seq_len, tok.size()[0])
        rnn_output = self.rnn_drop(rnn_output)  # [B, seq_len, dim]
        input = rnn_output  # Hidden from BiLSTM (B, seq_len, dim)
        """
        traditional GCN
        """
        # syn_output = F.relu(self.gcn1(input, dep_adj))
        # syn_output = F.relu(self.gcn2(syn_output, dep_adj))
        """
        added distance weight
        """
        # syn_output = self.gcn1(input, dep_adj, key_padding_mask, aspect_mask)
        # syn_output = self.gcn2(syn_output, dep_adj, key_padding_mask, aspect_mask)
        # Syntactic GCN
        syn_output = self.syn_gcn(input, dep_adj, adj_dict_list, src_mask, aspect_mask)
        # Semantic GCN
        sem_output = self.sem_gcn(input, dep_adj, src_mask, aspect_mask)

        syn_output = F.relu(self.Wsyn(syn_output))
        sem_output = F.relu(self.Wsem(sem_output))

        return syn_output, sem_output