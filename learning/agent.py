import abc

import numpy as np
import pygame

from utilities import discount_rewards
from .optimizer import Adam, cross_entropy


class AgentBase(abc.ABC):

    def __init__(self, game, speed, network):
        self.game = game
        self.speed = speed
        self.network = network
        self.age = 1
        self.rewards = []

    @abc.abstractmethod
    def sample_vector(self, frame, prev_reward):
        raise NotImplementedError

    def accumulate(self, rewards):
        pass

    def update(self):
        self.age += 1


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
        self.Ys = []
        self.args = []
        self.rewards = []
        self.gradients = np.zeros_like(network.get_gradients())
        self.optimizer = Adam()
        self.recurrent = False

    def reset(self):
        self.Xs = []
        self.args = []
        self.Ys = []
        self.rewards = []

    def sample_vector(self, state, prev_reward):
        self.rewards.append(prev_reward)
        probs = self.network.prediction(state[None, :])[0]
        arg, direction, label = self.game.sample_action(probs)
        self.Xs.append(state)
        self.Ys.append(label)
        return np.array(direction) * self.speed

    def accumulate(self, reward):
        print("ANN gradient accumulation... Policy Cost:", end=" ")
        Xs = np.vstack(self.Xs)
        Ys = np.vstack(self.Ys)
        drwds = discount_rewards(np.array(self.rewards[1:] + [reward]))
        m = Xs.shape[0]

        preds = self.network.prediction(Xs)
        net_cost = cross_entropy(preds, Ys) / m
        print("{:.4f}".format(net_cost * drwds.mean()))
        delta = (preds - Ys) * drwds / m
        self.network.backpropagation(delta)
        self.gradients += self.network.get_gradients()
        self.reset()

    def update(self):
        net = self.network
        update = self.optimizer.optimize(
            W=net.get_weights(),
            gW=net.get_gradients(),
        )
        net.set_weights(update)
        self.gradients = np.zeros_like(self.gradients)


class SpazzAgent(AgentBase):

    def sample_vector(self, frame, prev_reward):
        self.rewards.append(prev_reward)
        dvec = np.random.uniform(-self.speed, self.speed, size=2)
        return dvec.astype(int)
