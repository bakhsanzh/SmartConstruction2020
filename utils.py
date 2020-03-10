from copy import deepcopy
import numpy as np
import torch

DEVICE = torch.device('cpu')
VERY_SMALL_NUMBER = 1e-10
VERY_LARGE_NUMBER = 1e10

TABLE_COL_NAMES = [
    'env_id',
    'soil',
    'worker',
    'active',
    'assigned',
    'accessible',
    'has_ug',
    'x',
    'y',
    'z',
    'row',
    'col',
    'process_time',
    'dt',
    'target',
    'visited_worker_id',
]

NODE_COL_NAMES = [
    'env_id',
    'g_id',
    'soil',
    'worker',
    'active',
    'target',
]

n_cols_table = len(TABLE_COL_NAMES)
TC = dict((TABLE_COL_NAMES[i], i) for i in range(n_cols_table))


def get_cols(target_col_names: list,
             col_map: dict):
    """
    :param col_map: ec, nc, tc
    :param target_col_names: list which contains names of columns
    :return: list of integers (indices of columns)
    """
    index_list = []
    for name in target_col_names:
        index_list.append(col_map[name])
    return index_list
