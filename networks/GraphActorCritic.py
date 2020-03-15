from utils import distribute_assignments, filter_trivials, deepcopy
import torch
import torch.nn as nn
from graph_utils import *
from networks.MLP import MultiLayerPerceptron as MLP
from networks.RelationalGN import RelationalGN
# from networks.SingleLayerRGN import SingleLayerRGN as SingleLayerRGN
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
            'num_neurons': [32],
            'spectral_norm': False
        }

        self.graph_actor = {
            'node_embed_dim': self.gac['node_embed_dim'],
            'ef_init_dim': self.gac['ef_init_dim'],
            'use_ef_init': True,
        }


class PolicyNet(nn.Module):
    def __init__(self, conf=None):
        super(PolicyNet, self).__init__()
        self.conf = conf
        self.graph_network = RelationalGN(**conf.graph_network)
        self.actor = GraphActor(**conf.actor)

    def forward(self, graph: dgl.DGLGraph):
        graph = self.graph_network(graph=graph, node_feature=graph.ndata['nf_init'])  # Relational Graph Network
        node_feature = graph.ndata.pop('node_feature')

        # actor (action probs)
        action_probabilities = self.actor(graph=graph, node_feature=node_feature)  # shape [n_actions]

        # sample action for exploration
        edge_action_distribution = Categorical(action_probabilities)
        nn_action = edge_action_distribution.sample()  # tensor(x)  # index of the edge chosen for action

        # logprob of the nn_action
        # logprob = edge_action_distribution.log_prob(nn_action)  # tensor(x)

        return nn_action


class RolloutEncoder(nn.Module):
    def __init__(self,
                 rnn_hidden_dim: int,
                 node_embed_dim: int,
                 planning_horizon: int,
                 conf=None):
        """
        :param rnn_hidden_dim:
        :param node_embed_dim:
        :param planning_horizon:
        :param conf: ConfigBase
        """
        super(RolloutEncoder, self).__init__()
        self.conf = conf
        self.planning_horizon = planning_horizon
        self.rnn_hidden_dim = rnn_hidden_dim
        self.node_embed_dim = node_embed_dim
        self.reward_dim = 1

        self.rollout_policy = PolicyNet(conf.policy_network)
        self.rnn = torch.nn.GRUCell(self.node_embed_dim + self.reward_dim, hidden_size=self.rnn_hidden_dim)

    def forward(self, env: Environment):
        """
        :param env:
        :return:
        """

        total_num_assignments = 0  # total number of assignments (aka actions) made during the episode
        round_id = 0
        rollout_buffer = []

        # --------------------------------------------------------------------------------------------------------------
        # ROLLOUT
        while total_num_assignments < self.planning_horizon:
            round_buffer, num_trivial_assignments = distribute_assignments(env=env,
                                                                           to_compute_rewards=True,
                                                                           policy=self.rollout_policy,
                                                                           mode="rollout",
                                                                           assignment_round_id=round_id)

            done, dt_trigger, trigger_worker_ids = env.update()  # update environment
            filtered_round_buffer = filter_trivials(round_buffer, rollout_buffer)

            for assign_instance in filtered_round_buffer:
                rollout_buffer.append(assign_instance)

            total_num_assignments += len(filtered_round_buffer)
            round_id += 1

            if done:
                break

        # --------------------------------------------------------------------------------------------------------------
        # ROLLOUT GRAPHS SEQUENCE RNN ENCODING
        rollout_embed = torch.zeros(size=rollout_buffer[0]['graph'].ndata['nf_'])
        for assignment_instance in rollout_buffer:
            graph, reward = assignment_instance['graph'], assignment_instance['reward']
            node_feature = graph.ndata.pop('node_feature')
            temp = torch.tensor((), dtype=torch.float32)
            reward = temp.new_full(size=(node_feature.shape[0], 1), fill_value=reward)

            rnn_input = torch.cat([node_feature, reward], dim=-1)
            rollout_embed = self.rnn(rnn_input, rollout_embed)
        # --------------------------------------------------------------------------------------------------------------

        return rollout_embed


class GraphActorCritic(nn.Module):
    def __init__(self, conf):
        super(GraphActorCritic, self).__init__()
        self.conf = conf  # type: GACConfig
        self.num_rollouts = self.conf.gac['num_rollouts']
        self.planning_horizon = self.conf.gac['planning_horizon']
        self.node_embed_dim = self.conf.gac['node_embed_dim']
        self.rollout_encoder = RolloutEncoder(conf=conf.rollout_encoder, **conf.rollout_encoder)
        self.model_free_gn = RelationalGN(**self.conf.mf_gn).to(DEVICE)
        self.multihop_rgn = self.conf.gac['multihop_rgn']
        self.multihop_rgn = RelationalGN(**self.conf.rgn).to(DEVICE)
        self.actor = GraphActor(**self.conf.graph_actor).to(DEVICE)
        self.critic = MLP(self.node_embed_dim, 1).to(DEVICE)
        self.rollout_memory = []
        self.edge_actions = []
        self.rewards = []

        """
        # if self.multihop_rgn:
        #     self.rgn = MultiLayerRGN(**self.conf.rgn).to(DEVICE)
        # else:
        #     self.rgn = SingleLayerRGN(**self.conf.rgn).to(DEVICE)"""

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

    def sim2a_forward(self,
                      graph: dgl.DGLGraph,
                      env: Environment):

        rollout_embed = self.rollout_encoder(env=deepcopy(env))
        updated_graph = self.model_free_gn(graph=graph, node_feature=graph.ndata['nf_init'])  # Relational Graph Network
        node_embed = updated_graph.ndata.pop('node_feature')

        actor_input = torch.cat([rollout_embed, node_embed], dim=-1)
        action_probabilities = self.actor(graph=graph, node_feature=actor_input)  # shape [n_actions]

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
