import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import numpy as np

class ShowGraph:
    def __init__(self, log_dir='runs/exp1'):
        self.writer = SummaryWriter(log_dir=log_dir)

    def log(self, reward_per_episode, epsilon):
        i = len(reward_per_episode) - 1
        mean_reward = np.mean(reward_per_episode[max(0, i - 99): i + 1])
        self.writer.add_scalar("Reward/Mean_100_Episodes", mean_reward, i)
        self.writer.add_scalar("Epsilon", epsilon, i)

    def close(self):
        self.writer.close()