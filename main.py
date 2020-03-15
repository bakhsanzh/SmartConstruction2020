from environment import Environment
from utils import *
from graph_utils import *
from networks.PPOAgent import PPOAgent, GraphActorCritic
import torch
import numpy as np
import os
import json
from test_maps import TEST_MAP_0, TEST_MAP_1, TEST_MAP_2

np.set_printoptions(suppress=True)

# GLOBALS
TEST_MAPS = [TEST_MAP_0, TEST_MAP_1, TEST_MAP_2]
CHECKPOINTS_FOLDER_DIR = os.path.join(os.getcwd(), "checkpoints")



def run_episode(
        episode_mode: str,
        max_episode_length: int,
        to_update: bool,
        agent: PPOAgent,
        to_export: bool = False,
        test_data: dict = None,
):
    """
    :param episode_mode: "train", "test", "rollout"
    :param max_episode_length:
    :param to_update:
    :param agent:
    :param to_export:
    :param test_data:
    :return:
    """
    env = Environment(env_mode=episode_mode, test_data_dict=test_data)
    if episode_mode == 'test':
        pass
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


def train_model(model_state_path=None):
    train_interval = 1
    update_interval = 1
    test_interval = 3
    model_save_interval = 100

    lr = 7e-4
    max_episode_length = 10000
    num_epochs = 0
    agent = PPOAgent(lr=lr)
    if model_state_path is not None:
        agent, num_epochs, kpi_dict = load_model(agent=agent, state_data_path=model_state_path)
    else:
        kpi_dict = dict(
            (test_map['name'], dict(reward=-VERY_LARGE_NUMBER, env_ticks=VERY_LARGE_NUMBER)) for test_map in
            TEST_MAPS)

    for episode_id in range(1, 100000):
        if episode_id % train_interval == 0:
            to_update = False
            if episode_id % update_interval == 0:
                to_update = True
            episode_reward, env_ticks, _ = run_episode(episode_mode='train',
                                                       to_update=to_update,
                                                       agent=agent,
                                                       max_episode_length=max_episode_length,
                                                       to_export=False)
            # log_message = 'Episode: [{:03.0f}], Reward [{:03.0f}], Env ticks: [{:03.0f}]'
            # log_message = log_message.format(episode_id, episode_reward, env_ticks)
            # print(log_message)
            if to_update:
                num_epochs += 1

        if episode_id % test_interval == 0:
            for test_map in TEST_MAPS:
                episode_reward, env_ticks, env_snapshot = run_episode(episode_mode='test',
                                                                      to_update=False,
                                                                      agent=agent,
                                                                      max_episode_length=max_episode_length,
                                                                      to_export=True,
                                                                      test_data=deepcopy(test_map),
                                                                      )
                print('----------------------------------------------------------------------------------')
                log_message = 'Epoch: [{:03.0f}], Test Map: [{}], Reward [{:03.0f}], Env ticks: [{:03.0f}]'
                log_message = log_message.format(num_epochs, test_map['name'], episode_reward, env_ticks)
                print(log_message)

                is_best = episode_reward > kpi_dict[test_map['name']]['reward']
                if is_best:
                    print("===> Saving a new best")
                    # update best dict
                    kpi_dict[test_map['name']]['reward'] = episode_reward
                    kpi_dict[test_map['name']]['env_ticks'] = env_ticks

                    save_checkpoint(
                        filename=test_map['name'],
                        folder_path=os.path.join(CHECKPOINTS_FOLDER_DIR, test_map['name']),
                        num_epochs=num_epochs,
                        agent=agent,
                        kpi_dict=kpi_dict,
                        env_snapshot=env_snapshot)

        if episode_id % model_save_interval == 0:
            print('----------------------------------------------------------------------------------')
            save_checkpoint(filename='Epoch_{:04.0f}'.format(num_epochs),
                            num_epochs=num_epochs,
                            folder_path=os.path.join(CHECKPOINTS_FOLDER_DIR, "latest"),
                            agent=agent,
                            kpi_dict=kpi_dict)
            print('Latest model save event')
            print('----------------------------------------------------------------------------------')


if __name__ == '__main__':
    for el in TEST_MAPS:
        test_map_dir = os.path.join(CHECKPOINTS_FOLDER_DIR, el['name'])
        if not os.path.exists(test_map_dir):
            os.makedirs(test_map_dir)
    state_path = os.path.join(CHECKPOINTS_FOLDER_DIR, "latest", "Epoch_2500.pth")
    train_model(model_state_path=None)
