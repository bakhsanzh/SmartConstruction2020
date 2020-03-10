from copy import deepcopy
import dgl
import torch
from networks.ConfigBase import ConfigBase
import torch.nn as nn
from temp.networks.GraphActorCritic import GraphActorCritic as GAC
from temp.networks.GraphActorCritic import GACConfig
from temp.graph_utils import *
from ranger import Ranger  # this is from ranger.py
from ranger import RangerVA  # this is from ranger913A.py
from ranger import RangerQH  # this is from rangerqh.py


class PPOAgent:
    def __init__(self,
                 lr: float,
                 betas: tuple = (0.9, 0.999),
                 gamma: float = 0.99,
                 k_epochs: int = 5,
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
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward in reversed(self.policy_old.rewards):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)  # shape [rollout size]
        rewards = (rewards - rewards.mean())
        rewards = rewards / (rewards.std() + 1e-5)

        states_memory = self.policy_old.states
        edge_actions_memory = self.policy_old.edge_actions

        old_logprobs = torch.stack(self.policy_old.logprobs).to(device).detach()  # size [rollout_size]

        # Optimize policy for K epochs:
        for _ in range(self.k_epochs):
            # Evaluating old actions and values :
            dist_entropy = self.policy.evaluate(state_memory=states_memory,
                                                edge_action_memory=edge_actions_memory)  # size [] = scalar value

            # Finding the ratio (pi_theta / pi_theta__old):
            logprobs = self.policy.logprobs[0].to(device)  # size [batch_size]
            ratios = torch.exp(logprobs - old_logprobs.detach())  # size [batch_size]

            # Finding Surrogate Loss:
            state_values = self.policy.state_values[0].to(device)  # size [13] = [batch_size]
            advantages = rewards - state_values.detach()  # size [13] = [batch_size]

            surr1 = ratios * advantages  # size [13] = [batch_size]
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages  # size [13] = [batch_size]
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            mean_loss = loss.mean()
            mean_loss.backward()

            self.optimizer.step()
            self.policy.clear_memory()

        self.policy_old.clear_memory()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
