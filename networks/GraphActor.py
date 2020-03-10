
import torch
import torch.nn as nn
from networks.MLP import MultiLayerPerceptron as MLP
from graph_utils import get_action_edges


class GraphActor(nn.Module):
    def __init__(self,
                 node_embed_dim: int,
                 use_ef_init: bool,
                 ef_init_dim: int,):
        super(GraphActor, self).__init__()
        self.node_embed_dim = node_embed_dim
        self.use_ef_init = use_ef_init
        self.ef_init_dim = ef_init_dim
        self.actor_input_dim = self.node_embed_dim * 2 + self.use_ef_init * self.ef_init_dim
        self.actor = MLP(input_dimension=self.actor_input_dim,
                         output_dimension=1,
                         num_neurons=[],
                         out_activation='ReLU',)

    def actor_func(self, edges):
        if self.use_ef_init:
            actor_input = [edges.src['node_feature'], edges.dst['node_feature'], edges.data['ef_init']]
        else:
            actor_input = [edges.src['node_feature'], edges.dst['node_feature']]
        actor_input = torch.cat(actor_input, dim=1)
        logits = self.actor(actor_input)  # shape [n_actions x 1]
        action_probs = logits.softmax(0)  # shape [n_actions x 1]
        return {'action_probs': action_probs}

    def forward(self, graph, node_feature):
        graph.ndata['node_feature'] = node_feature
        action_edges = get_action_edges(graph)
        graph.apply_edges(func=self.actor_func, edges=action_edges)
        action_probs = graph.edges[action_edges].data['action_probs']  # shape [n_actions x 1]
        action_probs = action_probs.squeeze(dim=1)  # shape [n_actions]

        _ = graph.ndata.pop('node_feature')
        _ = graph.edata.pop('action_probs')

        return action_probs
