import abc

import numpy as np
import pygame
# hyperparameters
from utilities.util import prepro, discount_rewards


class AgentBase(abc.ABC):

    def __init__(self, game, speed, network):
        self.game = game
        self.speed = speed
        self.network = network
        self.age = 1
        self.rewards = []
        self.running_reward = None

    @abc.abstractmethod
    def sample_vector(self, frame, prev_reward):
        raise NotImplementedError

    def reset(self):
        rws = sum((r for r in self.rewards if r is not None))

        if self.running_reward is None:
            self.running_reward = rws
        else:
            self.running_reward *= 0.9
            self.running_reward += (rws * 0.1)

    def update(self, reward):
        self.age += 1
        self.reset()


class ManualAgent(AgentBase):

    def sample_vector(self, frame, prev_reward):
        self.rewards.append(prev_reward)
        action = pygame.key.get_pressed()
        vector = np.array([0, 0])
        if action[pygame.K_UP]:
            vector[1] -= 1
        if action[pygame.K_DOWN]:
            vector[1] += 1
        if action[pygame.K_LEFT]:
            vector[0] -= 1
        if action[pygame.K_RIGHT]:
            vector[0] += 1
        return vector * self.speed


class CleverAgent(AgentBase):

    def __init__(self, game, speed, network):
        super().__init__(game, speed, None)
        self.network = network
        self.Xs = []
        self.raw_Ys = []
        self.args = []
        self.rewards = []

    def reset(self):
        super().reset()
        self.Xs = []
        self.args = []
        self.raw_Ys = []
        self.rewards = []

    def sample_vector(self, frame, prev_reward):
        self.rewards.append(prev_reward)
        X = prepro(frame)
        probs = self.network.predict(X[None, :])[0]
        arg, direction, label = self.game.sample_action(probs)
        self.Xs.append(X)
        self.raw_Ys.append(probs)
        self.args.append(arg)
        return np.array(direction) * self.speed

    def update(self, reward):
        Xs = np.vstack(self.Xs)
        Ys = np.vstack(self.raw_Ys)
        drwds = discount_rewards(np.array(self.rewards[1:] + [reward]))
        drwds -= drwds.mean()
        drwds /= drwds.std()

        self.network.epoch(Xs, Ys, bsize=100, discount_rwds=drwds)
        super().update(reward)


class SpazzAgent(AgentBase):

    def sample_vector(self, frame, prev_reward):
        self.rewards.append(prev_reward)
        dvec = np.random.uniform(-self.speed, self.speed, size=2)
        return dvec.astype(int)
