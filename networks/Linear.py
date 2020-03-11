import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn.init as init

TORCH_ACTIVATION_LIST = ['ReLU',
                         'Sigmoid',
                         'SELU',
                         'leaky_relu',
                         'Softplus']

ACTIVATION_LIST = ['mish', 'swish', None]


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * F.sigmoid(x)


def get_nn_activation(activation: 'str'):
    if not activation in TORCH_ACTIVATION_LIST + ACTIVATION_LIST:
        raise RuntimeError("Not implemented activation function!")

    if activation in TORCH_ACTIVATION_LIST:
        act = getattr(torch.nn, activation)()

    if activation in ACTIVATION_LIST:
        if activation == 'mish':
            act = Mish()
        elif activation == 'swish':
            act = Swish()
        elif activation is None:
            act = nn.Identity()

    return act


class NoisyLinear(nn.Linear):
    # Adapted from the original source
    # https://github.com/Kaixhin/NoisyNet-A3C/blob/master/model.py

    def __init__(self, in_features, out_features, sigma_init=0.017):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=True)  # TODO: Adapt for no bias
        # µ^w and µ^b reuse self.weight and self.bias
        self.sigma_init = sigma_init
        self.sigma_weight = Parameter(torch.Tensor(out_features, in_features))  # σ^w
        self.sigma_bias = Parameter(torch.Tensor(out_features))  # σ^b
        self.register_buffer('epsilon_weight', torch.zeros(out_features, in_features))
        self.register_buffer('epsilon_bias', torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'sigma_weight'):  # Only init after all params added (otherwise super().__init__() fails)
            init.uniform_(self.weight, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
            init.uniform_(self.bias, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
            init.constant_(self.sigma_weight, self.sigma_init)
            init.constant_(self.sigma_bias, self.sigma_init)

    def forward(self, x):
        return F.linear(x, self.weight + self.sigma_weight * self.epsilon_weight,
                        self.bias + self.sigma_bias * self.epsilon_bias)

    def sample_noise(self):
        self.epsilon_weight = torch.randn(self.out_features, self.in_features)
        self.epsilon_bias = torch.randn(self.out_features)

    def remove_noise(self):
        self.epsilon_weight = torch.zeros(self.out_features, self.in_features)
        self.epsilon_bias = torch.zeros(self.out_features)


class LinearModule(nn.Module):

    def __init__(self,
                 activation: 'str',
                 norm: 'str' = None,
                 dropout_p: 'float' = 0.0,
                 weight_init: 'str' = None,
                 use_noisy: bool = False,
                 use_residual: bool = False,
                 **linear_kwargs):
        super(LinearModule, self).__init__()

        if linear_kwargs['in_features'] == linear_kwargs['out_features'] and use_residual:
            self.use_residual = True
        else:
            self.use_residual = False

        # layers
        if use_noisy:
            linear_layer = NoisyLinear(**linear_kwargs)
        else:
            linear_layer = torch.nn.Linear(**linear_kwargs)

        self.linear_layer = linear_layer
        if dropout_p > 0.0:
            self.dropout_layer = torch.nn.Dropout(dropout_p)
        else:
            self.dropout_layer = torch.nn.Identity()
        self.activation_layer = get_nn_activation(activation)

        self.weight_init = weight_init
        self.activation = activation
        self.norm = norm

        # apply weight initialization methods
        self.apply_weight_init(self.linear_layer, self.weight_init)

        if norm == 'batch':
            self.norm_layer = torch.nn.BatchNorm1d(self.linear_layer.out_features)
        elif norm == 'layer':
            self.norm_layer = torch.nn.LayerNorm(self.linear_layer.out_features)
        elif norm == 'spectral':
            self.linear_layer = torch.nn.utils.spectral_norm(self.linear_layer)
            self.norm_layer = torch.nn.Identity()
        elif norm is None:
            self.norm_layer = torch.nn.Identity()
        else:
            raise RuntimeError("Not implemented normalization function!")

    def apply_weight_init(self, tensor, weight_init=None):
        if weight_init is None:
            pass  # do not apply weight init
        elif weight_init == "normal":
            torch.nn.init.normal_(tensor.weight, std=0.3)
            torch.nn.init.constant_(tensor.bias, 0.0)
        elif weight_init == "kaiming_normal":
            if self.activation in ['sigmoid', 'tanh', 'relu', 'leaky_relu']:
                torch.nn.init.kaiming_normal_(tensor.weight, nonlinearity=self.activation)
                torch.nn.init.constant_(tensor.bias, 0.0)
            else:
                pass
        elif weight_init == "xavier":
            torch.nn.init.xavier_uniform_(tensor.weight)
            torch.nn.init.constant_(tensor.bias, 0.0)
        else:
            raise NotImplementedError("MLP initializer {} is not supported".format(weight_init))

    def forward(self, x):
        if self.use_residual:
            input_x = x

        x = self.linear_layer(x)
        x = self.norm_layer(x)
        x = self.activation_layer(x)
        x = self.dropout_layer(x)

        if self.use_residual:
            x = input_x + x
        return x
