from functools import partial
import dgl
import torch

device = torch.device('cpu')


def filter_edges_by_type(edges, etype_id):
    return edges.data['e_type'] == etype_id


def get_filtered_edges_by_type(graph: dgl.DGLGraph, etype_id: int):
    filter_func = partial(filter_edges_by_type, etype_id=etype_id)
    edge_ids = graph.filter_edges(filter_func)
    return edge_ids


def filter_nodes_by_type(nodes, ntype_id):
    return nodes.data['n_type'] == ntype_id


def get_filtered_nodes_by_type(graph: dgl.DGLGraph, ntype_id: int):
    filter_func = partial(filter_nodes_by_type, ntype_id=ntype_id)
    node_ids = graph.filter_nodes(filter_func)
    return node_ids


def filter_action_edges(edges):
    return edges.data['action_space'] == 1


def get_action_edges(graph: dgl.DGLGraph):
    edge_ids = graph.filter_edges(filter_action_edges)
    return edge_ids


def g2e_map(graph: dgl.DGLGraph, nn_action: int):
    env_action = int(graph.edges[nn_action].data['g2e'][0].item())
    return env_action


def dn(tensor):
    return tensor.detach().cpu().numpy()