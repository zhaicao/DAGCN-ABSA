"""
Basic operations on trees.
"""
import torch
import numpy as np
from collections import defaultdict
import torch.nn.functional as F

def head_to_adj(sent_len, head, tokens, label, len_, mask, directed=False, self_loop=True):
    """
    Convert a sequence of head indexes into a 0/1 matirx and label matrix.
    """
    adj_matrix = np.zeros((sent_len, sent_len), dtype=np.float32)
    label_matrix = np.zeros((sent_len, sent_len), dtype=np.int64)

    assert not isinstance(head, list)
    tokens = tokens[:len_].tolist()
    head = head[:len_].tolist()
    label = label[:len_].tolist()
    asp_idxs = [idx for idx in range(len(mask)) if mask[idx] == 1]
    for idx, head in enumerate(head):
        if idx in asp_idxs:
            for k in asp_idxs:
                adj_matrix[idx][k] = 1
                label_matrix[idx][k] = 2
        if head != 0:
            adj_matrix[idx, head - 1] = 1
            label_matrix[idx, head - 1] = label[idx]
        else:
            if self_loop:
                adj_matrix[idx, idx] = 1
                label_matrix[idx, idx] = 42  # self loop
                continue
        if not directed:
            adj_matrix[head - 1, idx] = 1
            label_matrix[head - 1, idx] = label[idx]
        if self_loop:
            adj_matrix[idx, idx] = 1
            label_matrix[idx, idx] = 42
    adj_dict = matrix_to_dict(adj_matrix)  # matrix2list
    return adj_matrix, label_matrix, adj_dict

def matrix_to_dict(adj_matrix):
    """
    Converting adjacency matrix to adjacency list
    returnType: defaultdict(set)
    """
    num_nodes = len(adj_matrix)
    adj_dict = {}

    for i in range(num_nodes):
        neighbors = []
        for j in range(num_nodes):
            if adj_matrix[i][j] == 1 and i != j:  # discard self-loop
                neighbors.append(j)
        if neighbors:  # None
            adj_dict[i] = neighbors
    return adj_dict


def dep_distance_adj(adj, dep_post_adj, aspect_mask, len_, maxlen):
    """
    The syntactic dependency distance weight
    """
    weight = np.zeros((maxlen, maxlen), dtype=np.float32)
    dep_post_adj = dep_post_adj[:len_, :len_].add(torch.eye(len_)).numpy()
    max_distance = dep_post_adj.max().item()  # max distance
    aspect_mask = aspect_mask[:len_].tolist()
    for i in range(len_):
        row_aspect = (aspect_mask[i] == 1)  # check whether this row belongs an aspect
        for j in range(len_):
            col_aspect = (aspect_mask[j] == 1)  # this col belongs an aspect
            # there is the dependency in row
            if row_aspect and adj[i][j] == 1:
                weight[i][j] = 1 - dep_post_adj[i][j] / (max_distance + 1)
            # there is the dependency in col
            if col_aspect and adj[i][j] == 1:
                weight[i][j] = 1 - dep_post_adj[i][j] / (max_distance + 1)
    adj = adj + weight  # A = A * (L + 1)
    padding = -9e15 * np.ones_like(adj)
    adj = np.where(adj > 0, adj, padding)
    return adj  # srd = A * (L + 1)
