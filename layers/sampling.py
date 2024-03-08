# -*- coding: utf-8 -*-
"""
"""
import numpy as np
import random


def sampling(src_nodes, maxlen, neighbor_table):
    """根据源节点采样指定数量的邻居节点，注意使用的是有放回的采样；
    某个节点的邻居节点数量少于采样数量时，采样结果出现重复的节点

    Arguments:
        src_nodes {list, ndarray} -- 源节点列表
        sample_num {int} -- 需要采样的节点数
        neighbor_table {dict} -- 节点到其邻居节点的映射表

    Returns:
        np.ndarray -- 采样结果构成的列表
    """
    results = []
    mask = []
    for sid in src_nodes:
        # 从节点的邻居加入全部的
        # res = np.random.choice(neighbor_table[sid], size=(sample_num,))
        # if sid != -1:
        #     sample_rand = neighbor_table[sid]
        #     random.shuffle(sample_rand)  # shuffle samples
        #     res = sample_rand + [-1] * (maxlen - len(neighbor_table[sid]))
        #     mask_idx = [1] * len(neighbor_table[sid]) + [0] * (maxlen - len(neighbor_table[sid]))
        #     results.append(res)
        #     mask.append(mask_idx)
        # if len(neighbor_table[sid]) >= maxlen:
        #     res = neighbor_table[sid]
        #     # res = np.random.choice(neighbor_table[sid], size=(maxlen,), replace=False)
        # else:
        #     res = neighbor_table[sid] + [-1] * (maxlen - len(neighbor_table[sid]))
        if sid in neighbor_table.keys():
            if sid != -1:
                sample_rand = neighbor_table[sid]
                random.shuffle(sample_rand)  # shuffle nodes
                res = sample_rand + [-1] * (maxlen - len(neighbor_table[sid]))
            else:
                res = [-1] * maxlen
        else:
            res = [-1] * maxlen
        results.append(res)
    return np.asarray(results).flatten(), np.asarray(mask).flatten()


def multihop_sampling(src_nodes, num_layers, neighbor_table):
    """根据源节点进行多阶采样

    Arguments:
        src_nodes {list, np.ndarray} -- 源节点id
        sample_nums {list of int} -- 每一阶需要采样的个数
        neighbor_table {dict} -- 节点到其邻居节点的映射

    Returns:
        [list of ndarray] -- 每一阶采样的结果
    """
    sampling_result = [np.array(src_nodes)]
    # print("sampling result = ", sampling_result)
    # print("sample_nums = ", sample_nums)
    max_len = np.max([len(row) for row in neighbor_table.values()])  # 最大邻居个数

    for k in range(num_layers):
        # print("sampling_result[k] = ", sampling_result[k])
        hopk_result, mask = sampling(sampling_result[k], max_len, neighbor_table)
        sampling_result.append(hopk_result)
    return sampling_result, max_len
