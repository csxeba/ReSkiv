import abc

import numpy as np
from scipy.ndimage import distance_transform_edt as dte
from matplotlib import pyplot as plt
import pygame


from utilities import (
    discount_rewards, prepro_recurrent, prepro_convolutional
)
from .optimization import Adam, cross_entropy


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
        super().__init__(game, speed, network)
        self.Xs = []
        self.Ys = []
        self.rewards = []
        self.gradients = np.zeros_like(network.get_gradients())
        self.optimizer = Adam()
        self.recurrent = False

    def reset(self):
        self.Xs = []
        self.Ys = []
        self.rewards = []

    def sample_vector(self, state, prev_reward):
        X = prepro_recurrent(state) if self.recurrent else state[None, :]
        self.rewards.append(prev_reward)
        probs = self.network.prediction(X)[0]
        direction, label = self.game.sample_action(probs)
        self.Xs.append(X)
        self.Ys.append(label)
        return np.array(direction) * self.speed

    def accumulate(self, reward):
        print("ANN gradient accumulation... Policy Cost:", end=" ")
        drwds = discount_rewards(np.array(self.rewards[1:] + [reward]))
        nz = np.argwhere(drwds).ravel()
        m = nz.size
        if m == 0:
            print("No valid lessons this round!")
            self.reset()
            return

        Xs = np.concatenate(self.Xs, axis=0)[nz]
        Ys = np.vstack(self.Ys)[nz]

        preds = self.network.prediction(Xs)

        net_cost = cross_entropy(preds, Ys) / m
        print("{: .4f}".format(net_cost * drwds.mean()))
        delta = (preds - Ys) * drwds[:, None] / m
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


class KerasAgent(AgentBase):

    type = "keras"

    def __init__(self, game, speed, network):
        super().__init__(game, speed, network)
        self.Xs = []
        self.Ys = []
        self.rewards = []
        self.recurrent = False
        self.convolutional = False

    def sample_vector(self, state, prev_reward):
        if self.convolutional:
            state = prepro_convolutional(state, 4)
        if self.recurrent:
            state = prepro_recurrent(state)
        if not self.convolutional or self.recurrent:
            state = state[None, :]
        self.rewards.append(prev_reward)
        self.Xs.append(state)
        probs = self.network.predict(state)[0]
        direction, label = self.game.sample_action(probs)
        self.Ys.append(label)
        return np.array(direction) * self.speed

    def reset(self):
        self.Xs = []
        self.Ys = []
        self.rewards = []

    def accumulate(self, reward):
        drwds = discount_rewards(np.array(self.rewards[1:] + [reward]))  # type: np.ndarray
        nz = np.argwhere(drwds != 0.).ravel()
        m = nz.size
        if m == 0:
            self.reset()
            return
        drwds = drwds[nz]
        X = np.concatenate(self.Xs, axis=0)[nz]
        Y = np.vstack(self.Ys)[nz]
        self.network.fit(X, Y, epochs=1, batch_size=300, sample_weight=drwds, shuffle=False)
        self.reset()


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

        self.prevstate = np.random.randn(5,)
        self.velocity = np.zeros((2,))
        self.memory = np.zeros((2,))

    def sample_vector1(self, state, prev_reward):
        self.velocity *= 0.8
        cstate = self.game.statistics()

        mypos = cstate[:2]  # relative position
        sqpos = cstate[2:4]  # relative position
        cdenemy = cstate[4]  # relative (0-1) distance

        square_vector = np.sign(sqpos - mypos)
        square_vector += square_vector * (1. - cdenemy)
        backpedal_vector = (mypos - self.prevstate[:2]) * cdenemy

        dangerous = (self.game.meandist / (self.game.maxdist*4.))

        net_movement = square_vector + backpedal_vector

        closing_enemy_modulator = 1
        if cdenemy < dangerous:
            closing_enemy_modulator = -1

        self.velocity += net_movement * closing_enemy_modulator
        self.velocity = np.clip(self.velocity, -self.speed, self.speed)

        self.prevstate = cstate
        return self.velocity

    def sample_vector(self, state, prev_reward):
        ds = 2
        state = self.game.statistics()
        peaks = np.ones(self.game.size // ds)
        valleys = np.ones_like(peaks)
        for e in self.game.enemies:
            peaks[tuple(e.coords // ds)] = 0.
        valleys[tuple(self.game.square.coords // ds)] = 0.

        scared = state[-1] / self.game.maxdist

        peaks = dte(peaks)
        peaks = 1. - peaks / self.game.maxdist
        valleys = dte(valleys)
        valleys = valleys / self.game.maxdist
        hills = peaks + valleys

        pc = tuple(self.game.player.coords // ds)

        grad = np.gradient(hills)
        grad = np.array([grad[0][pc], grad[1][pc]])*300

        decay_v = 0.8
        decay_m = 0.9

        # self.velocity = decay_v * self.velocity + (1. - decay_v) * grad
        # self.memory = decay_m * self.memory + (1. - decay_m) * grad**2.

        self.velocity *= decay_v
        self.velocity += grad

        danger = self.game.meandist / self.game.maxdist
        if state[-1] > danger / 2.:
            self.velocity += np.sign(state[:2] - state[2:4])

        self.velocity = np.clip(self.velocity, -self.speed, self.speed)

        return -grad
