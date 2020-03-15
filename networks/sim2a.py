import dgl
import torch
import torch.nn as nn
from copy import deepcopy
from utils import distribute_assignments, filter_trivials
from networks.MLP import MultiLayerPerceptron as MLP
from networks.RelationalGN import RelationalGN
from networks.GraphActor import GraphActor
from torch.distributions import Categorical
from environment import Environment
from utils import dn


class PolicyNet(nn.Module):
    def __init__(self,
                 nf_init_dim: int,
                 ef_init_dim: int,
                 node_embed_dim: int,
                 ):
        super(PolicyNet, self).__init__()

        graph_network_dict = {
            'is_linear_encoders': True,
            'nf_init_dim': nf_init_dim,
            'ef_init_dim': ef_init_dim,
            'output_dim': node_embed_dim,
            'num_hidden_layers': 0,
            'hidden_dim': 32,
            'num_edge_types': 4,
            'num_node_types': 2,
            'use_multi_node_types': True,
            'use_ef_init': True,
            'use_dst_features': True,
            'use_nf_concat': True,
            'num_neurons': [32],
            'spectral_norm': False
        }
        self.graph_network = RelationalGN(**graph_network_dict)

        graph_actor_dict = {
            'node_embed_dim': node_embed_dim,
            'ef_init_dim': ef_init_dim,
            'use_ef_init': False,
        }
        self.actor = GraphActor(**graph_actor_dict)

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
                 nf_init_dim: int,
                 ef_init_dim: int,
                 node_embed_dim: int,
                 rnn_hidden_dim: int,
                 planning_horizon: int
                 ):
        """
        :param rnn_hidden_dim:
        :param node_embed_dim:
        :param planning_horizon:
        :param conf: ConfigBase
        """
        super(RolloutEncoder, self).__init__()
        self.planning_horizon = planning_horizon
        self.rnn_hidden_dim = rnn_hidden_dim
        self.node_embed_dim = node_embed_dim
        self.reward_dim = 1

        policy_net_dict = {
            "nf_init_dim": nf_init_dim,
            "ef_init_dim": ef_init_dim,
            "node_embed_dim": node_embed_dim,
        }
        self.rollout_policy = PolicyNet(**policy_net_dict)
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
        rollout_embed = torch.zeros(size=rollout_buffer[0]['graph'].ndata['nf_'])  # TODO: check embed dim here
        for assignment_instance in rollout_buffer:
            graph, reward = assignment_instance['graph'], assignment_instance['reward']
            node_feature = graph.ndata.pop('node_feature')
            temp = torch.tensor((), dtype=torch.float32)
            reward = temp.new_full(size=(node_feature.shape[0], 1), fill_value=reward)

            rnn_input = torch.cat([node_feature, reward], dim=-1)
            rollout_embed = self.rnn(rnn_input, rollout_embed)
        # --------------------------------------------------------------------------------------------------------------

        return rollout_embed


class Sim2A(nn.Module):
    def __init__(self,
                 nf_init_dim: int,
                 ef_init_dim: int,
                 node_embed_dim: int,
                 num_rollouts: int = 1,
                 rnn_hidden_dim: int = 32,
                 planning_horizon: int = 5,
                 ):
        super(Sim2A, self).__init__()
        self.num_rollouts = num_rollouts
        self.node_embed_dim = node_embed_dim
        self.nf_init_dim = nf_init_dim
        self.ef_init_dim = ef_init_dim

        graph_network_dict = {
            'is_linear_encoders': True,
            'nf_init_dim': nf_init_dim,
            'ef_init_dim': ef_init_dim,
            'output_dim': node_embed_dim,
            'num_hidden_layers': 1,
            'hidden_dim': 32,
            'num_edge_types': 4,
            'num_node_types': 2,
            'use_multi_node_types': True,
            'use_ef_init': True,
            'use_dst_features': True,
            'use_nf_concat': True,
            'num_neurons': [32],
            'spectral_norm': False
        }
        self.model_free_gn = RelationalGN(**graph_network_dict)

        rollout_encoder_dict = {
            "nf_init_dim": nf_init_dim,
            "ef_init_dim": ef_init_dim,
            "node_embed_dim": node_embed_dim,
            "rnn_hidden_dim": rnn_hidden_dim,
            "planning_horizon": planning_horizon,
        }
        self.rollout_encoder = RolloutEncoder(**rollout_encoder_dict)

        sim2a_actor_dict = {
            'node_embed_dim': node_embed_dim + rnn_hidden_dim,
            'ef_init_dim': ef_init_dim,
            'use_ef_init': False,
        }
        self.sim2a_actor = GraphActor(**sim2a_actor_dict)

        self.critic = MLP(node_embed_dim + rnn_hidden_dim, 1)
        self.memory = []

    def forward(self,
                graph: dgl.DGLGraph,
                env: Environment):
        rollout_embed = self.rollout_encoder(env=deepcopy(env))
        updated_graph = self.model_free_gn(graph=graph, node_feature=graph.ndata['nf_init'])  # Relational Graph Network
        node_embed = updated_graph.ndata.pop('node_feature')

        sim2a_input = torch.cat([rollout_embed, node_embed], dim=-1)
        action_probabilities = self.actor(graph=graph, node_feature=sim2a_input)  # shape [n_actions]

        # critic
        state_value = self.critic(sim2a_input)
        state_value = state_value.mean(dim=0)

        # sample action for exploration
        edge_action_distribution = Categorical(action_probabilities)

        nn_action = edge_action_distribution.sample()  # tensor(x)  # index of the edge chosen for action

        # logprob of the nn_action
        logprob = edge_action_distribution.log_prob(nn_action)  # tensor(x)

        return nn_action, logprob, state_value

    def evaluate(self, rollout: list, ):
        logprobs = []
        state_values = []
        entropies = []

        for i, memory_instance in enumerate(rollout):
            graph = memory_instance['graph']
            env = memory_instance['env']
            old_nn_action = memory_instance['nn_action']

            rollout_embed = self.rollout_encoder(env=env)
            updated_graph = self.model_free_gn(graph=graph,
                                               node_feature=graph.ndata['nf_init'])  # Relational Graph Network
            node_embed = updated_graph.ndata.pop('node_feature')

            sim2a_input = torch.cat([rollout_embed, node_embed], dim=-1)
            action_probabilities = self.actor(graph=graph, node_feature=sim2a_input)  # shape [n_actions]

            # critic
            state_value = self.critic(sim2a_input)
            state_value = state_value.mean(dim=0)

            # sample action for exploration
            edge_action_distribution = Categorical(action_probabilities)

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

    def optimal(self,
                graph: dgl.DGLGraph,
                env: Environment):
        rollout_embed = self.rollout_encoder(env=deepcopy(env))
        updated_graph = self.model_free_gn(graph=graph, node_feature=graph.ndata['nf_init'])  # Relational Graph Network
        node_embed = updated_graph.ndata.pop('node_feature')

        sim2a_input = torch.cat([rollout_embed, node_embed], dim=-1)
        action_probabilities = self.actor(graph=graph, node_feature=sim2a_input)  # shape [n_actions]
        argmax_nn_action = torch.argmax(action_probabilities).item()

        # action_edge_ids = get_action_edges(graph)
        # assigned_edge_id = action_edge_ids[argmax_nn_action]
        # return assigned_edge_id
        return argmax_nn_action, dn(action_probabilities)

    def clear_memory(self):
        self.memory = []
