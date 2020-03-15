import torch
from networks.RelationalGNBlock import RelationalGNBlock as RelGNBlock
from graph_utils import *


class RelationalGN(torch.nn.Module):
    def __init__(self,
                 num_hidden_layers,
                 nf_init_dim,
                 ef_init_dim,
                 hidden_dim,
                 output_dim,
                 num_edge_types,
                 num_node_types,
                 num_neurons,
                 spectral_norm,
                 use_multi_node_types,
                 use_ef_init,
                 use_dst_features,
                 use_nf_concat,
                 is_linear_encoders,
                 use_noisy=False):

        super(RelationalGN, self).__init__()

        layers = []
        first_layer = RelGNBlock(is_first_layer=True,
                                 use_nf_concat=False,
                                 is_linear_encoders=is_linear_encoders,
                                 input_dim=nf_init_dim,
                                 output_dim=hidden_dim,
                                 nf_init_dim=nf_init_dim,
                                 ef_init_dim=ef_init_dim,
                                 num_edge_types=num_edge_types,
                                 use_ef_init=use_ef_init,
                                 use_dst_features=use_dst_features,
                                 use_multi_node_types=use_multi_node_types,
                                 num_neurons=num_neurons,
                                 spectral_norm=spectral_norm,
                                 use_noisy=use_noisy,
                                 num_node_types=num_node_types)
        layers.append(first_layer)

        for _ in range(num_hidden_layers):
            hidden_layer = RelGNBlock(input_dim=hidden_dim,
                                      output_dim=hidden_dim,
                                      nf_init_dim=nf_init_dim,
                                      ef_init_dim=ef_init_dim,
                                      is_linear_encoders=is_linear_encoders,
                                      num_edge_types=num_edge_types,
                                      use_ef_init=use_ef_init,
                                      use_dst_features=use_dst_features,
                                      use_nf_concat=use_nf_concat,
                                      num_neurons=num_neurons,
                                      spectral_norm=spectral_norm,
                                      use_noisy=use_noisy,
                                      use_multi_node_types=use_multi_node_types,
                                      num_node_types=num_node_types)

            layers.append(hidden_layer)

        out_layer = RelGNBlock(input_dim=hidden_dim,
                               output_dim=output_dim,
                               nf_init_dim=nf_init_dim,
                               ef_init_dim=ef_init_dim,
                               is_linear_encoders=is_linear_encoders,
                               num_edge_types=num_edge_types,
                               use_ef_init=use_ef_init,
                               use_dst_features=use_dst_features,
                               use_nf_concat=use_nf_concat,
                               num_neurons=num_neurons,
                               spectral_norm=spectral_norm,
                               use_noisy=use_noisy,
                               use_multi_node_types=use_multi_node_types,
                               num_node_types=num_node_types)
        layers.append(out_layer)

        self.layers = torch.nn.ModuleList(layers)

    def forward(self, graph, node_feature):
        for layer in self.layers:
            updated_node_feature = layer(graph, node_feature)
            node_feature = updated_node_feature

        graph.ndata['node_feature'] = node_feature
        return graph
