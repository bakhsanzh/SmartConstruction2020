from new_environment import Environment
from utils import *


def distribute_assignments(env: Environment,
                           to_compute_rewards: bool,
                           policy,
                           mode: str):
    assignments = []
    num_trivial_assignments = 0
    unassigned_workers = env.get_unassigned_worker_ids()

    for worker_id in unassigned_workers:  # TODO: shuffle workers.
        # graph, action_space_cardinality = env.observe(worker_id) # TODO: env.observe
        action_space_cardinality = 10
        graph = None
        nn_action = None
        # Schedule-Net assignment
        # assert action_space_cardinality >= 1
        is_trivial = False
        available_task_ids = env.get_available_task_ids()
        action_space_cardinality = len(available_task_ids)

        if action_space_cardinality > 1:  # non trivial action, requires an agent making a decision

            nn_action, logprob, state_value = policy(graph)
            env_action = np.random.choice(available_task_ids)

            # if mode == "train":
            # nn_action = policy(graph)  # type: int # nn_action is a edge id of an action in the graph.
            # elif mode == "rollout":
            #     nn_action = policy(graph)
            # elif mode == "mf":
            #     nn_action = policy.model_free(graph)
            # elif mode == "test":
            #     nn_action = policy.optimal(graph)
            # env_action = g2e_map(graph, nn_action)  # type: int # map nn_action back to env_action
            distance = env.assign_worker(worker_id, env_action)  # type: float # obtain reward from env.
        else:
            if action_space_cardinality == 1:
                is_trivial = True
                # nn_action = graph.edata['action'].nonzero().item()  # get the only available action
                # env_action = g2e_map(graph, nn_action)
                env_action = available_task_ids[0]
                distance = env.assign_worker(worker_id, env_action)  # type: float # obtain reward from env.
                num_trivial_assignments += 1

        reward = None
        if to_compute_rewards:
            reward = -distance

        assignment = dict(worker_id=worker_id,
                          graph=graph,
                          env_action=env_action,
                          nn_action=nn_action,
                          is_trivial=is_trivial,
                          is_transitory=False,
                          reward=reward)

        assignments.append(assignment)

        bundle = {
            'graph': None,
            ''

        }

    assignments[-1]['is_transitory'] = True  # mark last assignment as transitory

    return assignments, num_trivial_assignments


def distribute_rewards(current_assignments, policy_rewards_memory: list):

    for i, assign_instance in enumerate(current_assignments):
        if assign_instance['is_trivial']:
            if len(policy_rewards_memory) == 0:
                raise ValueError('Error with trivial first state')
            else:
                policy_rewards_memory[-1] += assign_instance['reward']
                current_assignments[-1]['reward'] += assign_instance['reward']
        else:
            .append(assign_instance)
    return out


def main():
    max_episode_length = 30
    episode_mode = 'train'

    env = Environment(env_mode='train')
    done = False
    total_episode_reward = 0
    total_num_assignments = 0  # total number of assignments (aka actions) made during the episode
    total_num_trivial_assignments = 0  # number of trivial assignments

    while not done and env.env_ticks < max_episode_length:
        assignments, num_trivial_assignments = distribute_assignments(env=env,
                                                                      to_compute_rewards=True,
                                                                      policy=None,
                                                                      mode=episode_mode)

        done, dt_trigger, trigger_worker_ids = env.update()
        round_reward = np.sum([assignment['reward'] for assignment in assignments])

        total_episode_reward += round_reward
        total_num_assignments += len(assignments)
        total_num_trivial_assignments += num_trivial_assignments

        if done:
            break

    print('episoded completed, num ticks: ', env.env_ticks)

    for tick, d_map in env.demand_history:
        print('t:', tick)
        print(d_map)
        print('------------')
    return total_episode_reward
if __name__ == '__main__':
    main()
