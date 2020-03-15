import torch
from torch.nn.functional import relu
from networks.MLP import MultiLayerPerceptron as MLP
from graph_utils import *
from functools import partial


class RelationalGNBlock(torch.nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 nf_init_dim: int,
                 ef_init_dim: int,
                 num_edge_types: int,
                 is_linear_encoders: bool,
                 use_attention: bool = True,
                 is_first_layer: bool = False,
                 num_neurons=None,
                 use_ef_init: bool = True,
                 use_dst_features: bool = True,
                 spectral_norm: bool = False,
                 use_nf_concat: bool = True,
                 use_ef_concat: bool = False,
                 use_multi_node_types: bool = True,
                 num_node_types: int = 2,
                 use_noisy=False):

        super(RelationalGNBlock, self).__init__()
        if num_neurons is None:
            num_neurons = [32]
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nf_init_dim = nf_init_dim
        self.ef_init_dim = ef_init_dim
        self.num_edge_types = num_edge_types
        self.num_node_types = num_node_types
        self.is_first_layer = is_first_layer
        self.use_attention = use_attention
        self.relational_encoder = dict()
        self.is_linear_encoders = is_linear_encoders

        self.use_multi_node_types = use_multi_node_types
        self.use_ef_concat = use_ef_concat
        self.use_nf_concat = use_nf_concat

        self.use_ef_init = use_ef_init
        self.use_dst_features = use_dst_features

        if self.is_first_layer:
            assert use_nf_concat is False
            assert input_dim == nf_init_dim

        out_activation = 'ReLU'

        # if we use linear encoders, then num_neurons in MLP is empty list []
        # activation function is RELU
        if is_linear_encoders:
            num_neurons_rel_enc = num_neurons_attn_enc = num_neurons_node_upd = []
        else:
            num_neurons_rel_enc = num_neurons_node_upd = num_neurons_attn_enc = num_neurons

        self.relational_encoder_input_dim = (self.input_dim + self.use_nf_concat * self.nf_init_dim) * (
                    1 + self.use_dst_features) + self.use_ef_init * self.ef_init_dim
        self.aggr_node_features_dim = (self.input_dim + self.use_nf_concat * self.nf_init_dim) + self.output_dim * self.num_edge_types

        # -------------------------------- ATTENTION -----------------------------------------------------------------
        if self.use_attention:
            self.attention_encoder_input_dim = self.nf_init_dim * 2 + self.ef_init_dim
            self.attention_encoder = MLP(input_dimension=self.attention_encoder_input_dim,
                                         output_dimension=1,
                                         num_neurons=num_neurons_attn_enc,
                                         out_activation='Sigmoid')

        # -------------------------------- RELATIONAL EDGE ENCODERS ---------------------------------------------------
        for i in range(num_edge_types):
            relational_encoder = MLP(input_dimension=self.relational_encoder_input_dim,
                                     output_dimension=output_dim,
                                     num_neurons=num_neurons_rel_enc,
                                     out_activation=out_activation)
            self.relational_encoder['rel_encoder_{}'.format(i)] = relational_encoder
        self.relational_encoder = torch.nn.ModuleDict(self.relational_encoder)

        # -------------------------------- MULTI-NODE-TYPE NODE ENCODERS ----------------------------------------------
        if use_multi_node_types:
            self.node_updater = dict()
            for i in range(num_node_types):
                node_updater = MLP(input_dimension=self.aggr_node_features_dim,
                                   output_dimension=output_dim,
                                   num_neurons=num_neurons_node_upd,
                                   out_activation=out_activation)
                self.node_updater['node_updater_{}'.format(i)] = node_updater
            self.node_updater = torch.nn.ModuleDict(self.node_updater)

        else:
            self.node_updater = MLP(self.aggr_node_features_dim,
                                    output_dim, num_neurons, spectral_norm, use_noisy=use_noisy)

    def forward(self, graph, node_feature):
        if self.use_nf_concat:
            graph.ndata['node_feature'] = torch.cat([node_feature, graph.ndata['nf_init']], dim=1)
        else:
            graph.ndata['node_feature'] = node_feature

        message_func = partial(self.message_function)
        reduce_func = partial(self.reduce_function)
        graph.send_and_recv(graph.edges(), message_func=message_func, reduce_func=reduce_func)

        if self.use_multi_node_types:
            for node_type_id in range(self.num_node_types):
                node_ids = get_filtered_nodes_by_type(graph, node_type_id)
                node_updater = self.node_updater['node_updater_{}'.format(node_type_id)]
                apply_node_func = partial(self.apply_node_function_multi_type, updater=node_updater)
                graph.apply_nodes(apply_node_func, v=node_ids)

        else:
            for node_type_id in range(self.num_node_types):
                node_ids = get_filtered_nodes_by_type(graph, node_type_id)
                graph.apply_nodes(self.apply_node_function, v=node_ids)

        updated_node_feature = graph.ndata.pop('updated_node_feature')
        _ = graph.ndata.pop('aggregated_node_feature')
        _ = graph.ndata.pop('node_feature')
        return updated_node_feature

    def message_function(self, edges):
        src_node_features = edges.src['node_feature']  # num_edges x input_dim
        # attention = edges.data['attention']
        dst_node_features = None
        ef_init = None

        if self.use_dst_features:
            dst_node_features = edges.dst['node_feature']
        if self.use_ef_init:
            ef_init = edges.data['ef_init']

        num_edges = src_node_features.shape[0]
        edge_types = edges.data['e_type']

        device = src_node_features.device

        msg_dict = dict()
        for edge_type_id in range(self.num_edge_types):
            msg = torch.zeros(num_edges, self.output_dim, device=device)

            # relational_updater = MLP ( rel_input_dim -> output_dim)
            relational_updater = self.relational_encoder['rel_encoder_{}'.format(edge_type_id)]  # network
            curr_relation_mask = edge_types == edge_type_id  # boolean mask for specific edge_type
            curr_relation_edge_ids = torch.arange(num_edges)[curr_relation_mask]  # edge ids of current relation

            if curr_relation_mask.sum() != 0:  # if this relation is exists in the edge types
                relational_updater_input = src_node_features[curr_relation_mask]  # get current node features
                # attn_values =
                if self.use_dst_features:
                    relational_updater_input = torch.cat([relational_updater_input,
                                                          dst_node_features[curr_relation_mask]], dim=1)
                if self.use_ef_init:
                    relational_updater_input = torch.cat([relational_updater_input,
                                                          ef_init[curr_relation_mask]], dim=1)

                # relational_updater_output = relu(relational_updater(relational_updater_input))
                relational_updater_output = relational_updater(relational_updater_input)

                msg[curr_relation_edge_ids, :] = relational_updater_output
                msg_dict['msg_{}'.format(edge_type_id)] = msg
            else:
                msg_dict['msg_{}'.format(edge_type_id)] = msg  # just fill the zeros tensor as a message
        msg_dict['nf_init_src'] = edges.src['nf_init']
        msg_dict['ef_init'] = edges.data['ef_init']
        msg_dict['nf_init_dst'] = edges.dst['nf_init']
        return msg_dict

    def reduce_function(self, nodes):
        node_feature = nodes.data['node_feature']  # num_nodes x input_dim
        # msg shape = [num nodes x num incoming edges x output_dim]
        # msg[i, :, :] = from ALL incoming messages from other edges to node-i
        # ATTENTION
        if self.use_attention:
            msg_nf_init_src = nodes.mailbox['nf_init_src']
            msg_nf_init_dst = nodes.mailbox['nf_init_dst']
            msg_ef_init = nodes.mailbox['ef_init']
            attention_encoder_input = torch.cat([msg_nf_init_src, msg_nf_init_dst, msg_ef_init], dim=-1)
            attention_encoder_output = self.attention_encoder(attention_encoder_input)

        num_nodes = node_feature.shape[0]
        device = node_feature.device

        nf_end_col = self.input_dim + self.use_nf_concat * self.nf_init_dim  # a col_id where node feature ends
        aggr_node_features = torch.zeros(num_nodes, self.aggr_node_features_dim, device=device)
        aggr_node_features[:, : nf_end_col] = relu(node_feature)

        start = nf_end_col
        for etype in range(self.num_edge_types):
            msg = nodes.mailbox['msg_{}'.format(etype)]  # num_nodes x num_incoming_edges x output dim
            # msg[incoming message source id, message destination node id, :], so msg[0, :, :] means incoming message from node 0 to every other
            if self.use_attention:
                msg = attention_encoder_output * msg
            reduced_msg = msg.sum(dim=1)
            end = start + self.output_dim
            aggr_node_features[:, start:end] = reduced_msg
            start = end
        return {'aggregated_node_feature': aggr_node_features}

    def apply_node_function(self, nodes):
        aggregated_node_feature = nodes.data['aggregated_node_feature']
        out = self.node_updater(aggregated_node_feature)
        return {'updated_node_feature': out}

    def apply_node_function_multi_type(self, nodes, updater):
        aggregated_node_feature = nodes.data['aggregated_node_feature']
        out = updater(aggregated_node_feature)
        return {'updated_node_feature': out}
