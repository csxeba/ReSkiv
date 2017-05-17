import abc

import numpy as np
import pygame

from utilities import discount_rewards
from .optimizer import Adam, cross_entropy


class AgentBase(abc.ABC):

    type = ""

    def __init__(self, game, speed, network):
        self.game = game
        self.speed = speed
        self.network = network
        self.age = 1
        self.rewards = []

    @abc.abstractmethod
    def sample_vector(self, state, prev_reward):
        raise NotImplementedError

    def accumulate(self, rewards):
        pass

    def update(self):
        self.age += 1


class ManualAgent(AgentBase):

    type = "manual"

    def sample_vector(self, state, prev_reward):
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

    type = "clever"

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
        print("{: .4f}".format(net_cost * drwds.mean()))
        delta = (preds - Ys) * drwds / m
        self.network.backpropagation(delta)
        self.gradients += self.network.get_gradients()
        self.reset()

    def update(self):
        print("ANN gradient update!")
        net = self.network
        update = self.optimizer.optimize(
            W=net.get_weights(),
            gW=net.get_gradients(),
        )
        net.set_weights(update)
        self.gradients = np.zeros_like(self.gradients)


class SpazzAgent(AgentBase):

    type = "spazz"

    def sample_vector(self, state, prev_reward):
        self.rewards.append(prev_reward)
        dvec = np.random.uniform(-self.speed, self.speed, size=2)
        return dvec.astype(int)


class MathAgent(AgentBase):

    type = "math"

    def __init__(self, game, speed, network):
        super().__init__(game, speed, network)

        self.prevstate = None
        self.prevmove = None
        self.scared = True
        self.velocity = np.array([0., 0.])
        self.velocity_backup = np.array([0., 0.])

    def sample_vector(self, state, prev_reward):
        self.velocity *= 0.9
        cstate = self.game.statistics()

        mypos = cstate[:2]
        sqpos = cstate[2:4]

        denemy = cstate[4] * self.game.meandist

        if denemy < self.game.meandist / 4.:
            if denemy > self.prevstate[-1]:
                self.velocity = -self.velocity_backup

                self.scared = True
            else:
                self.velocity = np.array([0., 0.])
        else:
            if not self.scared:
                self.velocity += np.sign(sqpos - mypos)
            else:
                self.velocity = self.velocity_backup
            self.velocity_backup = np.clip(self.velocity, -self.speed, self.speed)
            self.scared = False

        self.prevstate = cstate
        return np.clip(self.velocity, -self.speed, self.speed)
