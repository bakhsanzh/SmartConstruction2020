from utils import DEVICE, dn
import torch
import torch.nn as nn
from graph_utils import *
from networks.MLP import MultiLayerPerceptron as MLP
from networks.RelationalGN import RelationalGN
from networks.GraphActor import GraphActor
from networks.ConfigBase import ConfigBase
from torch.distributions import Categorical
from environment import Environment
import numpy as np


class GACConfig(ConfigBase):
    def __init__(self, name='gac', gac_conf=None, rgn_conf=None):
        super(GACConfig, self).__init__(name=name, gac=gac_conf, rgn=rgn_conf)

        self.gac = {
            'multihop_rgn': True,
            'node_embed_dim': 32,
            'nf_init_dim': 9,
            # 'accessible', 'inaccessible', 'assigned', 'unassigned', 'has_ug', 'not_has_ug', 'dt', 'nb_min', 'nb_max'
            'ef_init_dim': 1,
        }

        self.rgn = {
            'is_linear_encoders': True,
            'nf_init_dim': self.gac['nf_init_dim'],
            'ef_init_dim': self.gac['ef_init_dim'],
            'output_dim': self.gac['node_embed_dim'],
            'num_hidden_layers': 1,
            'hidden_dim': 64,
            'num_edge_types': 4,
            'num_node_types': 2,
            'use_multi_node_types': True,
            'use_ef_init': False,
            'use_dst_features': False,
            'use_nf_concat': True,
            'num_neurons': [],
            'spectral_norm': False
        }

        self.graph_actor = {
            'node_embed_dim': self.gac['node_embed_dim'],
            'ef_init_dim': self.gac['ef_init_dim'],
            'use_ef_init': True,
        }


class GraphActorCritic(nn.Module):
    def __init__(self, conf):
        super(GraphActorCritic, self).__init__()
        self.conf = conf  # type: GACConfig
        self.node_embed_dim = self.conf.gac['node_embed_dim']
        self.multihop_rgn = self.conf.gac['multihop_rgn']
        self.rgn = RelationalGN(**self.conf.rgn).to(DEVICE)
        self.actor = GraphActor(**self.conf.graph_actor).to(DEVICE)
        self.critic = MLP(self.node_embed_dim, 1).to(DEVICE)
        self.rollout_memory = []

    def forward(self, graph: dgl.DGLGraph):
        graph = self.rgn(graph=graph, node_feature=graph.ndata['nf_init'])  # Relational Graph Network
        node_embed = graph.ndata.pop('node_feature')
        # temp = node_embed.cpu().detach().numpy()

        # critic
        state_value = self.critic(node_embed)
        state_value = state_value.mean(dim=0)

        # actor (action probs)
        action_probabilities = self.actor(graph=graph, node_feature=node_embed)  # shape [n_actions]

        # sample action for exploration
        edge_action_distribution = Categorical(action_probabilities)
        nn_action = edge_action_distribution.sample()  # tensor(x)  # index of the edge chosen for action

        # logprob of the nn_action
        logprob = edge_action_distribution.log_prob(nn_action)  # tensor(x)

        return nn_action, logprob, state_value

    def optimal(self, graph: dgl.DGLGraph):
        graph = self.rgn(graph=graph, node_feature=graph.ndata['nf_init'])
        node_embed = graph.ndata.pop('node_feature')

        action_probabilities = self.actor(graph=graph, node_feature=node_embed)  # shape [n_actions]
        argmax_nn_action = torch.argmax(action_probabilities).item()

        # action_edge_ids = get_action_edges(graph)
        # assigned_edge_id = action_edge_ids[argmax_nn_action]
        # return assigned_edge_id
        return argmax_nn_action, dn(action_probabilities)

    def evaluate(self,
                 rollout_graphs: list,
                 rollout_nn_actions: list,
                 ):
        batch_graph = dgl.batch(rollout_graphs)
        batch_graph.set_n_initializer(dgl.init.zero_initializer)

        batch_graph = self.rgn(graph=batch_graph, node_feature=batch_graph.ndata['nf_init'])
        g_batch = dgl.unbatch(batch_graph)

        logprobs = []
        state_values = []
        entropies = []

        for i, g in enumerate(g_batch):
            node_embed = g.ndata.pop('node_feature')
            # critic
            state_value = self.critic(node_embed)  # [n_nodes x 1]
            state_value = state_value.mean(dim=0)  # [1]

            # actor
            action_probabilities = self.actor(graph=g, node_feature=node_embed)  # shape [n_actions]

            edge_action_distribution = torch.distributions.Categorical(action_probabilities)

            # get old action from the rollout
            old_nn_action = rollout_nn_actions[i]

            # get log probability of the old action given current policy distribution
            logprob = edge_action_distribution.log_prob(old_nn_action)

            # get entropy of current policy
            entropy = edge_action_distribution.entropy()

            state_values.append(state_value)
            logprobs.append(logprob)
            entropies.append(entropy)

        logprobs = torch.stack(logprobs, dim=0)
        state_values = torch.stack(state_values, dim=0).squeeze(dim=1)
        entropies = torch.stack(entropies, dim=0)

        return logprobs, state_values, entropies

    def clear_memory(self):
        self.rollout_memory = []
