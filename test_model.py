from utils import *
import numpy as np
from networks.PPOAgent import PPOAgent
from main import run_episode


def save_snapshot(data, filename, directory):
    fname = filename + '.npy'
    fname = os.path.join(directory, fname)
    np.save(file=fname, arr=data, allow_pickle=True)


def test_model(model_state_path, test_map_dict):
    agent = PPOAgent(lr=7e-4)
    agent, num_epochs, kpi_dict = load_model(agent=agent, state_data_path=model_state_path)
    episode_reward, env_ticks, env_snapshot = run_episode(episode_mode='test',
                                                          to_update=False,
                                                          agent=agent,
                                                          max_episode_length=int(1e5),
                                                          to_export=True,
                                                          test_data=deepcopy(test_map_dict),
                                                          )
    log_message = 'Epoch: [{:03.0f}], Test Map: [{}], Reward [{:03.0f}], Env ticks: [{:03.0f}]'
    log_message = log_message.format(num_epochs, test_map_dict['name'], episode_reward, env_ticks)
    print(log_message)

    return episode_reward, env_ticks, env_snapshot


def main():
    current_dir = os.getcwd()
    state_dict_path = os.path.join(current_dir, "checkpoints", "latest", "Epoch_1100.pth")
    from test_scenario import TEST_MAP_3
    target_map = TEST_MAP_3

    save_dir = os.path.join(current_dir, "test_cases")
    epi_reward, env_ticks, env_snapshot = test_model(model_state_path=state_dict_path, test_map_dict=target_map)
    save_snapshot(data=env_snapshot, filename=target_map['name'], directory=save_dir)

    from kivy_animator import run_kivy_app
    kivy_kwargs = dict(
        data_path=os.path.join(save_dir, target_map['name'] + '.npy'),
        to_export=False,
        to_animate=False,)
    run_kivy_app(kivy_kwargs)


if __name__ == '__main__':
    main()
