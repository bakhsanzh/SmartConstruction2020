import torch.nn as nn
from torch.distributions import Categorical
from networks.MLP import MultiLayerPerceptron as MLP
from networks.RelationalGraphNetworks.RelationalGraphLayers import MultiLayerRGN, SingleLayerRGN
# from networks.SingleLayerRGN import SingleLayerRGN as SingleLayerRGN
from networks.GraphActor import GraphActor
from graph_utils import *
from networks.ConfigBase import ConfigBase


class GACConfig(ConfigBase):
    def __init__(self, name='gac', gac_conf=None, rgn_conf=None):
        super(GACConfig, self).__init__(name=name, gac=gac_conf, rgn=rgn_conf)

        self.gac = {
            'multihop_rgn': True,
            'node_embed_dim': 16,
            'nf_init_dim': 5,  # depo, not depo, assigned, unassigned, to_depot_norm
            'ef_init_dim': 3,  # dt_norm, to depot, non to depot
        }

        self.rgn = {
            'is_linear_encoders': True,
            'nf_init_dim': self.gac['nf_init_dim'],
            'ef_init_dim': self.gac['ef_init_dim'],
            'output_dim': self.gac['node_embed_dim'],
            'num_hidden_layers': 2,
            'hidden_dim': 16,
            'num_edge_types': 4,
            'num_node_types': 2,
            'use_multi_node_types': True,
            'use_ef_init': True,
            'use_dst_features': False,
            'use_nf_concat': True,
            'num_neurons': [32],
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
        if self.multihop_rgn:
            self.rgn = MultiLayerRGN(**self.conf.rgn)
        else:
            self.rgn = SingleLayerRGN(**self.conf.rgn)
        self.actor = GraphActor(**self.conf.graph_actor)

        self.critic = MLP(self.node_embed_dim, 1)

        self.edge_actions = []
        self.states = []  # dgl graphs
        self.logprobs = []  # scalar values
        self.state_values = []
        self.rewards = []

    def forward(self, graph: dgl.DGLGraph):
        graph = self.rgn(graph=graph, node_feature=graph.ndata['nf_init'])
        node_embed = graph.ndata.pop('node_feature')
        temp = node_embed.cpu().detach().numpy()

        # critic
        state_value = self.critic(node_embed)
        state_value = state_value.mean(dim=0)

        # actor (action probs)
        action_probabilities = self.actor(graph=graph, node_feature=node_embed)  # shape [n_actions]

        # sample action for exploration
        edge_action_distribution = torch.distributions.Categorical(action_probabilities)
        nn_action = edge_action_distribution.sample()  # tensor(x)  # index of the edge chosen for action

        # logprob of the nn_action
        logprob = edge_action_distribution.log_prob(nn_action)  # tensor(x)

        # get action
        action_edge_ids = get_action_edges(graph)
        assigned_edge_id = action_edge_ids[nn_action]

        # ROLLOUT MEMORY
        memory_bundle = {
            'graph': graph,
            'nn_action': nn_action,
            'logprob': logprob,
            'state_value': state_value
        }
        self.states.append(graph)
        self.edge_actions.append(nn_action)
        self.logprobs.append(logprob)
        self.state_values.append(state_value)

        return assigned_edge_id.item()  # return edge id

    def optimal(self, graph: dgl.DGLGraph):
        graph = self.rgn(graph=graph, node_feature=graph.ndata['nf_init'])
        node_embed = graph.ndata.pop('node_feature')
        temp = node_embed.cpu().detach().numpy()
        action_probabilities = self.actor(graph=graph, node_feature=node_embed)  # shape [n_actions]
        argmax_edge_action = torch.argmax(action_probabilities).item()
        action_edge_ids = get_action_edges(graph)
        assigned_edge_id = action_edge_ids[argmax_edge_action]

        return assigned_edge_id

    def evaluate(self,
                 state_memory: list,
                 edge_action_memory: list,
                 ):
        batch_graph = dgl.batch(state_memory)
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
            old_edge_action = edge_action_memory[i]

            # get log probability of the old action given current policy distribution
            logprob = edge_action_distribution.log_prob(old_edge_action)

            # get entropy of current policy
            entropy = edge_action_distribution.entropy()

            state_values.append(state_value)
            logprobs.append(logprob)
            entropies.append(entropy)

        logprobs = torch.stack(logprobs, dim=0)
        state_values = torch.stack(state_values, dim=0).squeeze(dim=1)
        entropies = torch.stack(entropies, dim=0)

        self.logprobs.append(logprobs)
        self.state_values.append(state_values)
        entropy_mean = entropies.mean()

        return entropy_mean

    def clear_memory(self):
        del self.edge_actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.state_values[:]
        del self.rewards[:]
