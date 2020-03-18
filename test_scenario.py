import numpy as np
import os
from copy import deepcopy

TEST_MAP_0 = {
    'name': 'square_sink_10x10',
    'altitude_map': np.array([
        [90, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 90],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [90, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 90, 0, 0, 0, 0, 90],
    ]),
    'required_map': -1 * np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 2, 2, 2, 2, 2, 2, 1, 0],
        [0, 1, 2, 3, 3, 3, 3, 2, 1, 0],
        [0, 1, 2, 3, 3, 3, 3, 2, 1, 0],
        [0, 1, 2, 3, 3, 3, 3, 2, 1, 0],
        [0, 1, 2, 3, 3, 3, 3, 2, 1, 0],
        [0, 1, 2, 2, 2, 2, 2, 2, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]),
}

TEST_MAP_1 = {
    'name': 'wavelike_5x9',
    'altitude_map': np.array([
        [90, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2],
        [3, 3, 3, 3, 3],
        [2, 2, 2, 2, 2],
        [1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2],
        [1, 1, 1, 1, 1],
        [90, 0, 0, 0, 90],
    ]),
    'required_map': np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]),
}

TEST_MAP_2 = {
    'name': 'two_hills_10x10',
    'altitude_map': np.array([
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [1, 2, 2, 2, 2, 0, 0, 0, 0, 0],
        [1, 2, 3, 3, 2, 0, 0, 0, 0, 0],
        [2, 2, 3, 3, 2, 0, 0, 0, 0, 0],
        [1, 1, 2, 2, 2, 0, 1, 1, 0, 90],
        [90, 0, 1, 1, 1, 1, 1, 2, 1, 0],
        [0, 0, 0, 0, 0, 1, 2, 3, 1, 1],
        [0, 0, 0, 0, 0, 1, 3, 3, 2, 1],
        [0, 0, 0, 0, 0, 1, 2, 3, 2, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    ]),
    'required_map': np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]),
}

TEST_MAP_3 = {
    'name': 'test_1',
    'altitude_map': np.array([
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 3, 3, 2, 1, 1, 0, 0, 0],
        [1, 1, 3, 3, 2, 1, 1, 0, 0, 0],
        [2, 1, 3, 3, 2, 2, 1, 0, 0, 0],
        [1, 1, 2, 2, 2, 2, 1, 1, 1, 90],
        [90, 1, 1, 1, 1, 1, 1, 2, 1, 1],
        [0, 0, 0, 1, 1, 1, 2, 3, 1, 1],
        [0, 0, 0, 1, 1, 2, 3, 3, 2, 1],
        [0, 0, 0, 0, 1, 1, 2, 2, 2, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    ]),
    'required_map': np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, -1, -1, -1, 0, 0, 0, 0, 0, 0],
        [-1, -2, -2, -1, 0, 0, 0, 0, 0, 0],
        [-1, -2, -2, -1, 0, 0, 0, 0, 0, 0],
        [0, -1, -1, -1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]),
}

CONSTRUCTION_MAP = {
    'name': 'construction',
    'altitude_map': np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 90, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 90, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 1, 1, 1, 1, 2, 1, 1],
        [0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 3, 3, 2, 2],
        [0, 0, 0, 90, 1, 2, 3, 3, 3, 3, 3, 3, 2, 1, 2, 2, 2, 3, 3, 3, 2],
        [0, 0, 0, 1, 2, 2, 3, 3, 3, 3, 3, 3, 2, 1, 2, 2, 1, 1, 2, 3, 2],
        [0, 0, 0, 1, 2, 2, 3, 3, 3, 2, 2, 2, 2, 1, 2, 2, 1, 0, 0, 2, 2],
        [0, 0, 0, 1, 2, 2, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 2, 2, 3, 3, 3, 2, 2, 1, 1, 0, 0, 90, 0, 0, 0, 0, 0],
        [0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 2, 2, 3, 3, 3, 3, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 2, 3, 3, 3, 3, 3, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 90, 1, 2, 3, 3, 3, 3, 3, 3, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 2, 3, 3, 3, 3, 3, 3, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 2, 3, 3, 3, 3, 3, 3, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 90, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 2, 1, 90, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 90, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=int),
    'required_map': np.zeros([31, 21], dtype=int)
}