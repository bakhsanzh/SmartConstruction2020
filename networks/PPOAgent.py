import torch
import torch.nn as nn
from graph_utils import *
from networks.GraphActorCritic import GraphActorCritic as GAC
from networks.GraphActorCritic import GACConfig


class PPOAgent:
    def __init__(self,
                 lr: float,
                 betas: tuple = (0.9, 0.999),
                 gamma: float = 0.99,
                 k_epochs: int = 4,
                 eps_clip: float = 0.2,
                 ):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        gac_conf = GACConfig()
        self.policy = GAC(gac_conf)  # type: GAC
        self.policy_old = GAC(gac_conf)  # type: GAC

        # optimizer and loss
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        # self.optimizer = Ranger(self.policy.parameters(), lr=lr)
        self.MseLoss = nn.MSELoss()

    def update(self):

        rollout_rewards = []
        rollout_graphs = []
        rollout_nn_actions = []
        rollout_logprobs = []

        for memory_instance in self.policy_old.rollout_memory:
            rollout_rewards.append(memory_instance['reward'])
            rollout_graphs.append(memory_instance['graph'])
            rollout_nn_actions.append(memory_instance['nn_action'])
            rollout_logprobs.append(memory_instance['logprob'])

        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward in reversed(rollout_rewards):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(DEVICE)  # shape [rollout size]
        rewards = (rewards - rewards.mean())
        rewards = rewards / (rewards.std() + 1e-5)

        rollout_logprobs = torch.stack(rollout_logprobs).to(DEVICE).detach()  # size [rollout_size]

        # Optimize policy for k_epochs:
        for _ in range(self.k_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, entropies = self.policy.evaluate(rollout_graphs=rollout_graphs, rollout_nn_actions=rollout_nn_actions)  # size [] = scalar value
            dist_entropy = entropies.mean()

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - rollout_logprobs)  # size [batch_size]

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()  # size [13] = [batch_size]

            surr1 = ratios * advantages  # size [13] = [batch_size]
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages  # size [13] = [batch_size]
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.clear_memory()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
