from environment import Environment
from utils import *
from graph_utils import *
from networks.PPOAgent import PPOAgent, GAC
import torch
import numpy as np
import os

np.set_printoptions(suppress=True)


def distribute_assignments(env: Environment,
                           to_compute_rewards: bool,
                           policy: GAC,
                           mode: str,
                           assignment_round_id: int):
    round_buffer = []
    num_trivial_assignments = 0
    unassigned_workers = env.get_unassigned_worker_ids()

    for worker_id in unassigned_workers:  # TODO: shuffle workers.
        graph, trivial, action_space_env_ids = env.observe(worker_id)
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
                nn_action, logprob, state_value = policy(graph)
                memory_instance['graph'] = graph
                memory_instance['nn_action'] = nn_action
                memory_instance['logprob'] = logprob
                memory_instance['state_value'] = state_value
                memory_instance['worker_id'] = worker_id

            elif mode == 'test' or mode == 'test_construction':
                nn_action, action_probs = policy.optimal(graph)
            env_action = g2e_map(graph, nn_action)
            reward = env.assign_worker(worker_id, env_action)
            memory_instance['env_action'] = env_action

        elif trivial:  # trivial action
            action_probs = 'None'
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

        if DEBUG and mode == 'test':
            env_action_coords = None
            if env_action is not None:
                env_action_coords = env.get_coords(env_action)
            print('ASSIGNMENT ROUND:', str(assignment_round_id).zfill(2))
            print('assigned ', worker_id, '--->', env_action)
            print('coords:', worker_coords, '--->', env_action_coords)
            print('action_space:', action_space_env_ids)
            print('action_probs:', action_probs)
            print('reward:', reward)
            print('--------------------')
    if DEBUG and mode == 'test':
        print('----------------------------------------------------------------------------------------------------')
        print('----------------------------------------------------------------------------------------------------')

    return round_buffer, num_trivial_assignments


def filter_trivials(round_buffer, large_buffer):
    """
    :param round_buffer: list containing assignment instances for current round
    :param large_buffer: list containing assignment instances that happened before (can be imaginary rollout instances
    or episode instances)
    :return: list of filtered assignment instances, i.e. does not contain trivial assignments

    GOAL: Eliminate trivial assignment instances and retroactively append trivial assignment rewards to previous real
    assignment's reward.
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


def run_episode(
        episode_mode: str,
        max_episode_length: int,
        to_update: bool,
        agent: PPOAgent,
        to_export: bool = False,
):
    env = Environment(env_mode=episode_mode)
    if episode_mode == 'test':
        print('---')
    done = False
    total_episode_reward = 0
    total_num_assignments = 0  # total number of assignments (aka actions) made during the episode
    total_num_trivial_assignments = 0  # number of trivial assignments
    round_id = 0
    episode_buffer = []
    unfiltered_episode_buffer = []
    while not done and env.env_ticks < max_episode_length:
        round_buffer, num_trivial_assignments = distribute_assignments(env=env,
                                                                       to_compute_rewards=True,
                                                                       policy=agent.policy_old,
                                                                       mode=episode_mode,
                                                                       assignment_round_id=round_id)

        done, dt_trigger, trigger_worker_ids = env.update()  # update environment
        unfiltered_episode_buffer.append(round_buffer)
        filtered_round_buffer = filter_trivials(round_buffer, episode_buffer)

        for assign_instance in filtered_round_buffer:
            episode_buffer.append(assign_instance)

        round_reward = np.sum([el['reward'] for el in filtered_round_buffer])
        total_episode_reward += round_reward
        total_num_assignments += len(filtered_round_buffer)
        total_num_trivial_assignments += num_trivial_assignments
        round_id += 1

        if done:
            break

    if episode_mode == 'train':
        for el in episode_buffer:
            agent.policy_old.rollout_memory.append(el)
            agent.policy_old.rewards.append(el['reward'])

    # ------------------------- UPDATE PPO-AGENT ----------------------------------------------------------------------
    if to_update and episode_mode == 'train':
        agent.update()

    env_snapshot = None
    if to_export:
        env_snapshot = env.export()

    return total_episode_reward, env.env_ticks, env_snapshot


def save_checkpoint(num_epochs: int,
                    agent: PPOAgent,
                    best_dict: dict,
                    env_snapshot=None,
                    is_construction: bool = False):
    directory = "./saved_models/"
    # state_filename = "state_dict.pth"
    # env_snapshot_filename = "env_snapshot.npy"
    # if is_construction:
    # directory = "./saved_models/construction/"
    state_filename = "epochs_{}_state_dict.pth".format(num_epochs)
    env_snapshot_filename = "epochs_{:03.0f}_reward_{:03.0f}_ticks_{:03.0f}_reversed.npy".format(num_epochs,
                                                                                                 (best_dict[
                                                                                                     'episode_reward']),
                                                                                                 (best_dict[
                                                                                                     'env_ticks']))
    state_filename = os.path.join(directory, state_filename)
    env_snapshot_filename = os.path.join(directory, env_snapshot_filename)

    state = {
        'num_epochs': num_epochs,
        'model_state_dict': agent.policy.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'best_dict': best_dict,
    }

    torch.save(state, state_filename)
    if env_snapshot is not None:
        np.save(env_snapshot_filename, env_snapshot, allow_pickle=True)


def main(to_load_model: bool = False):
    train_interval = 1
    update_interval = 1
    test_interval = 3
    lr = 7e-3
    max_episode_length = 10000
    num_epochs = 0
    construction_test_interval = 200
    agent = PPOAgent(lr=lr)
    if not to_load_model:
        best_dict = {
            'episode_reward': -VERY_LARGE_NUMBER,
            'env_ticks': VERY_LARGE_NUMBER,
        }
    elif to_load_model:
        main_path = "C:/Users/Sanzhar/PycharmProjects/SmartConstruction2020/saved_models/"
        state_dict_name = "epochs_957_state_dict.pth"
        state_dict = torch.load(os.path.join(main_path, state_dict_name))
        model_state_dict = state_dict['model_state_dict']
        optimizer_state_dict = state_dict['optimizer_state_dict']
        agent.policy_old.load_state_dict(model_state_dict)
        agent.policy.load_state_dict(model_state_dict)
        agent.optimizer.load_state_dict(optimizer_state_dict)
        num_epochs = state_dict['num_epochs']
        best_dict = state_dict['best_dict']
        print('---')

    for episode_id in range(1, 10000):
        if episode_id % train_interval == 0:
            to_update = False
            if episode_id % update_interval == 0:
                to_update = True
            episode_reward, env_ticks, _ = run_episode(episode_mode='train',
                                                       to_update=to_update,
                                                       agent=agent,
                                                       max_episode_length=max_episode_length,
                                                       to_export=False)
            log_message = 'Episode: [{:03.0f}], Reward [{:03.0f}], Env ticks: [{:03.0f}]'
            log_message = log_message.format(episode_id, episode_reward, env_ticks)
            print(log_message)
            if to_update:
                num_epochs += 1

        if episode_id % test_interval == 0:
            episode_reward, env_ticks, env_snapshot = run_episode(episode_mode='test',
                                                                  to_update=False,
                                                                  agent=agent,
                                                                  max_episode_length=max_episode_length,
                                                                  to_export=True)
            print('----------------------------------------------------------------------------------')
            log_message = 'Epoch: [{:03.0f}], Reward [{:03.0f}], Env ticks: [{:03.0f}]'
            log_message = log_message.format(num_epochs, episode_reward, env_ticks)
            print(log_message)
            print('----------------------------------------------------------------------------------')
            is_best = episode_reward > best_dict['episode_reward']
            if is_best:
                print("=> Saving a new best")
                # log_message = 'Epoch: [{:03d}], Reward [{:03d}], Env ticks: [{:03d}]'
                # log_message = log_message.format(num_epochs, episode_reward, env_ticks)
                # print(log_message)
                # print('----------------------------------------------------------------------------------')

                # update best dict
                best_dict['episode_reward'] = episode_reward
                best_dict['env_ticks'] = env_ticks

                save_checkpoint(num_epochs=num_epochs,
                                agent=agent,
                                best_dict=best_dict,
                                env_snapshot=env_snapshot)


def test_construction(state_dict_path):
    agent = PPOAgent(lr=7e-4)
    main_path = "C:/Users/Sanzhar/PycharmProjects/SmartConstruction2020/saved_models/"
    state_dict_name = "epochs_171_state_dict.pth"
    num_epochs = 171
    state_dict = torch.load(os.path.join(main_path, state_dict_name))
    model_state_dict = state_dict['model_state_dict']
    optimizer_state_dict = state_dict['optimizer_state_dict']
    agent.policy_old.load_state_dict(model_state_dict)
    agent.policy.load_state_dict(model_state_dict)
    agent.optimizer.load_state_dict(optimizer_state_dict)

    episode_reward, env_ticks, env_snapshot = run_episode(episode_mode='test_construction',
                                                          to_update=False,
                                                          agent=agent,
                                                          max_episode_length=100000,
                                                          to_export=True)

    log_message = 'CONSTRUCTION [{:03.0f}], Reward [{:03.0f}], Env ticks: [{:03.0f}]'
    log_message = log_message.format(num_epochs, episode_reward, env_ticks)
    print(log_message)
    best_dict = dict()
    best_dict['episode_reward'] = episode_reward
    best_dict['env_ticks'] = env_ticks

    save_checkpoint(num_epochs=171, agent=agent, best_dict=best_dict, env_snapshot=env_snapshot, is_construction=True)


if __name__ == '__main__':
    DEBUG = False
    # main(to_load_model=True)
    test_construction(None)
