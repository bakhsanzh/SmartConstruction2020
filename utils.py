from copy import deepcopy
import numpy as np
import torch
from graph_utils import g2e_map
import dgl
from environment import Environment
import json
import os
from networks.PPOAgent import PPOAgent
from networks.GraphActorCritic import GraphActorCritic, RolloutEncoder, PolicyNet
from copy import deepcopy

# DEVICE = torch.device('cpu')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VERY_SMALL_NUMBER = 1e-10
VERY_LARGE_NUMBER = 1e10

TABLE_COL_NAMES = [
    'env_id',
    'soil',
    'worker',
    'active',
    'assigned',
    'accessible',
    'has_ug',
    'x',
    'y',
    'z',
    'row',
    'col',
    'process_time',
    'under_task_id',
    'dt',
    'nb_min',
    'nb_max',
    'target',
    'visited_worker_id',
]

NODE_COL_NAMES = [
    'env_id',
    'g_id',
    'soil',
    'worker',
    'active',
    'target',
]

n_cols_table = len(TABLE_COL_NAMES)
TC = dict((TABLE_COL_NAMES[i], i) for i in range(n_cols_table))


def get_cols(target_col_names: list,
             col_map: dict):
    """
    :param col_map: ec, nc, tc
    :param target_col_names: list which contains names of columns
    :return: list of integers (indices of columns)
    """
    index_list = []
    for name in target_col_names:
        index_list.append(col_map[name])
    return index_list


def save_checkpoint(
        filename,
        folder_path,
        num_epochs: int,
        agent: PPOAgent,
        kpi_dict: dict,
        env_snapshot=None,
):
    info_dict = {
        'event': filename,
        'num_epochs': num_epochs,
        'kpi_dict': kpi_dict,
    }
    state_fname = filename + '.pth'
    state_fname = os.path.join(folder_path, state_fname)

    info_fname = filename + '.txt'
    info_fname = os.path.join(folder_path, info_fname)

    state_dict = {
        'event': filename,
        'num_epochs': num_epochs,
        'model_state_dict': agent.policy.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'kpi_dict': kpi_dict,
    }
    torch.save(state_dict, state_fname)
    with open(info_fname, 'w', encoding='utf-8') as f:
        json_dump = json.dumps(info_dict, sort_keys=True, indent=4)
        f.write(json_dump)

    if env_snapshot is not None:
        env_snapshot_fname = filename + '.npy'
        env_snapshot_fname = os.path.join(folder_path, env_snapshot_fname)
        np.save(env_snapshot_fname, env_snapshot, allow_pickle=True)


def load_model(
        agent: PPOAgent,
        state_data_path):
    state_dict = torch.load(state_data_path)
    model_state = state_dict['model_state_dict']
    optimizer_state = state_dict['optimizer_state_dict']

    agent.policy_old.cuda(DEVICE).load_state_dict(model_state)
    agent.policy.cuda(DEVICE).load_state_dict(model_state)
    agent.optimizer.load_state_dict(optimizer_state)

    num_epochs = state_dict['num_epochs']
    kpi_dict = state_dict['kpi_dict']

    return agent, num_epochs, kpi_dict


def distribute_assignments(env: Environment,
                           to_compute_rewards: bool,
                           policy,
                           mode: str,
                           assignment_round_id: int):
    """
    :param env: instance of Environment class
        env is required to env.observe, env.assign_worker
    :param to_compute_rewards: bool
        if True, env.assign_worker will compute the Reward of the assignment
    :param policy: torch.nn.Module
        policy network that processes the observation from environment and returns the nn_action
    :param mode: string
        possible mode = ["train", "test", "rollout"]
        "train": uses policy forward() function
        "test": uses policy optimal() function
        "rollout": uses policy's rollout() function
    :param assignment_round_id: int
        id of the assignment round.
    :return:
    """
    round_buffer = []
    num_trivial_assignments = 0
    unassigned_workers = env.get_unassigned_worker_ids()

    for worker_id in unassigned_workers:  # TODO: shuffle workers.
        graph, trivial, action_space_env_ids = env.observe(worker_id, fix_num_nodes=True)
        worker_coords = deepcopy(env.get_coords(worker_id))
        memory_instance = dict(
            graph=None,  # state + action space
            nn_action=None,  # action
            reward=None,  # reward
            logprob=None,
            state_value=None,
            env_action=None,
            worker_id=worker_id,
            is_trivial=trivial,
            last_assignment=False,
        )

        if not trivial:  # non trivial action, requires an agent making a decision
            if mode == 'train':
                assert policy is GraphActorCritic
                nn_action, logprob, state_value = policy(graph)
                memory_instance['graph'] = graph
                memory_instance['nn_action'] = nn_action
                memory_instance['logprob'] = logprob
                memory_instance['state_value'] = state_value
                memory_instance['worker_id'] = worker_id
            elif mode == 'test':
                nn_action, action_probs = policy.optimal(graph)

            elif mode == 'sim2a-test':
                nn_action, action_probs = policy.sim2a_optimal(graph)

            elif mode == 'rollout':
                assert policy is PolicyNet
                nn_action = policy(graph=graph, env=env)  # policy: RolloutEncoder class
                memory_instance['graph'] = graph
                memory_instance['nn_action'] = nn_action
                memory_instance['worker_id'] = worker_id

            elif mode == 'sim2a-train':
                assert policy is GraphActorCritic
                nn_action, logprob, state_value = policy.sim2a(graph=graph, env=env)  # policy: RolloutEncoder class
                memory_instance['graph'] = graph
                memory_instance['nn_action'] = nn_action
                memory_instance['logprob'] = logprob
                memory_instance['state_value'] = state_value
                memory_instance['worker_id'] = worker_id

            env_action = g2e_map(graph, nn_action)
            reward = env.assign_worker(worker_id, env_action)
            memory_instance['env_action'] = env_action

        elif trivial:  # trivial action
            if len(action_space_env_ids) == 1:  # select the only available action
                env_action = action_space_env_ids[0]
                reward = env.assign_worker(worker_id, env_action)
                num_trivial_assignments += 1
            elif len(action_space_env_ids) == 0:  # no action is available for a worker. skip
                env_action = None
                continue

        if to_compute_rewards:
            memory_instance['reward'] = reward

        memory_instance['env_action'] = env_action
        round_buffer.append(memory_instance)

    return round_buffer, num_trivial_assignments


def filter_trivials(round_buffer, large_buffer):
    """
    :param round_buffer: list
        Contains assignment instances for current round
    :param large_buffer: list
        Contains assignment instances that happened before (can be imaginary rollout instances
    or episode instances)
    :return: list
        Contains filtered assignment instances, i.e. does not contain trivial assignments
        GOAL: Eliminate trivial assignment instances and retroactively append trivial assignment rewards to previous
        real assignment's reward:
        if instance is_trivial:
            last real instance reward += trivial instance reward
    """
    filtered_round_buffer = []
    for i, assign_inst in enumerate(round_buffer):
        if assign_inst['is_trivial']:
            if len(large_buffer) == 0:
                continue

            if i == 0 and assign_inst['is_trivial']:
                prev_assign_inst = large_buffer[-1]
            else:
                prev_assign_inst = round_buffer[i - 1]

            prev_assign_inst['reward'] += assign_inst['reward']  # add trivial actions reward to prev action's reward
        else:
            filtered_round_buffer.append(assign_inst)

    if len(filtered_round_buffer) > 0:
        filtered_round_buffer[-1]['last_assignment'] = True
    return filtered_round_buffer
