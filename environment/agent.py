import abc

import numpy as np
import pygame

from utilities import (
    discount_rewards, prepro_recurrent, prepro_convolutional, normalize,
    proximity_gradient_sream, time_gradient_stream, calculate_gradient
)
from learning.experience import Experience
from learning.optimization import Adam, cross_entropy


class AgentBase(abc.ABC):

    type = ""

    def __init__(self, game, speed, network, scale):
        self.game = game
        self.speed = speed
        self.network = network
        self.scale = scale
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


class RecordAgent(ManualAgent):

    def __init__(self, game, speed, network, scale):
        super().__init__(game, speed, network, scale)
        self.outchain = ""

    def sample_vector(self, state, prev_reward):
        dvec = super().sample_vector(state, prev_reward)
        grad1, grad2 = calculate_gradient(self.game, 2*self.scale, 2)
        data = list(self.game.statistics()[:2]) + list(grad1) + list(grad2)
        self.outchain += ",".join(str(d) for d in data)
        self.outchain += ";" + ",".join(str(d) for d in np.sign(dvec))
        self.outchain += "\n"
        return dvec

    def accumulate(self, rewards):
        with open("supervised.data", "a") as handle:
            handle.write(self.outchain)
        self.outchain = ""


class OnlineAgent(RecordAgent):

    def __init__(self, game, speed, network, scale):
        super().__init__(game, speed, network, scale)
        self.gradients = np.zeros_like(network.get_gradients())
        self.optimizer = Adam()
        self.Xs = []
        self.Ys = []
        self.ngrads = 0

    def sample_vector(self, state, prev_reward):
        dvec = super().sample_vector(state, prev_reward)
        direction = dvec // self.speed
        self.Xs.append(state.ravel())
        self.Xs.append(state[:, ::-1, :].ravel())
        direction[0] *= -1
        self.Ys.append(self.game.labels[self.game.actions.index(tuple(direction))])
        self.Xs.append(state[:, :, ::-1].ravel())
        direction *= -1
        self.Ys.append(self.game.labels[self.game.actions.index(tuple(direction))])
        return dvec

    def _process_data(self, X, Y):
        m = X.shape[0]
        preds = self.network.prediction(X)
        cost = cross_entropy(preds, Y) / m
        self.network.backpropagation((preds - Y) / m)
        self.gradients += self.network.get_gradients()
        self.ngrads += 1
        print("Online ANN accumulating {} lessons. Cost: {:.3f}".format(m, cost))

    def accumulate(self, rewards):
        super().accumulate(rewards)
        X, Y = np.vstack(self.Xs)[:-240], np.vstack(self.Ys)[:-240]
        self.Xs = []
        self.Ys = []
        if X.shape[0] < 1:
            return
        aY = np.copy(Y)
        aY[:, 0] *= -1
        self._process_data(X, Y)

    def update(self):
        print("Online ANN weight updating!")
        if not self.ngrads:
            return
        updates = self.optimizer.optimize(
            self.network.get_weights(), self.gradients / self.ngrads
        )
        self.network.set_weights(updates)
        self.gradients = np.zeros_like(self.gradients)
        self.ngrads = 0
        self.network.save("online.agent")


class SavedAgent(AgentBase):

    def sample_vector(self, state, prev_reward):
        X = state.ravel()[None, :]
        probs = self.network.prediction(X)[0]
        actn, label = self.game.sample_action(probs)
        return np.array(actn) * self.speed


class PolicyLearningAgent(AgentBase):

    type = "clever"

    def __init__(self, game, speed, network, scale):
        super().__init__(game, speed, network, scale)
        self.Xs = []
        self.Ys = []
        self.rewards = []
        self.gradients = np.zeros_like(network.get_gradients())
        self.ngrads = 0
        self.optimizer = Adam()
        self.recurrent = False

    def reset(self):
        self.Xs = []
        self.Ys = []
        self.rewards = []

    def sample_vector(self, state, prev_reward):
        X = state.ravel()[None, :]
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

        if self.recurrent:
            self.network.reset_lstm()
        preds = self.network.prediction(Xs)

        net_cost = cross_entropy(preds, Ys) / m
        print("{: .4f}".format(net_cost * drwds.mean()))
        delta = (preds - Ys) * drwds[:, None] / m
        self.network.backpropagation(delta)
        self.gradients += self.network.get_gradients()
        self.ngrads += 1
        self.reset()

    def update(self):
        if not self.ngrads:
            return
        print("ANN gradient update!")
        net = self.network
        update = self.optimizer.optimize(
            W=net.get_weights(),
            gW=self.gradients / self.ngrads,
        )
        net.set_weights(update)
        self.gradients = np.zeros_like(self.gradients)
        self.ngrads = 0


class QLearningAgent(AgentBase):

    type = "q"

    def __init__(self, game, speed, network, scale):
        from keras.models import Sequential
        from keras.layers import Dense
        network = Sequential([
            Dense(500, activation="tanh", input_dim=20**2 + 5),
            Dense(120, activation="tanh"),
            Dense(len(game.actions), activation="linear")
        ])
        network.compile("adam", "mse")
        super().__init__(game, speed, network, scale)
        self.S = []
        self.Qs = []
        self.R = []
        self.A = []
        self.xp = Experience()

        self.epsilon = 0.1
        self.recurrent = False

    def reset(self):
        self.S, self.A, self.Qs, self.R = [[] for _ in range(4)]

    def sample_vector(self, state, prev_reward):
        S = np.append(self.game.proximity().ravel(), self.game.statistics())
        self.S.append(S)
        self.R.append(prev_reward)
        Q = self.network.predict(S[None, ...])[0]
        self.Qs.append(Q)
        ix = (np.argmax(Q) if np.random.uniform() < self.epsilon
              else np.random.randint(0, len(Q)))
        self.A.append(ix)
        return np.array(self.game.actions[ix]) * self.speed

    def accumulate(self, rewards):
        X = np.vstack(self.S)
        X = X[:-1]
        R = discount_rewards(np.array(self.R[1:]))
        Y = np.vstack(self.Qs[:-1])
        ixs = tuple(self.A[1:])
        Y[-1, self.A[-1]] = rewards
        rwd = R + Y.max(axis=1)
        Y[:, ixs] = rwd
        self.xp.accumulate(X, Y)
        self.reset()

    def update(self):
        X, Y = self.xp.get_batch(15000)
        self.network.fit(X, Y, epochs=1, batch_size=300)


class KerasAgent(AgentBase):

    type = "keras"

    def __init__(self, game, speed, network, scale):
        super().__init__(game, speed, network, scale)
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


# noinspection PyUnusedLocal
class MathAgent(AgentBase):

    type = "math"

    def __init__(self, game, speed, network, scale):
        super().__init__(game, speed, network, scale)

        self.prevstate = np.random.randn(5,)
        self.velocity = np.zeros((2,))
        self.memory = np.zeros((2,))
        self.proxy_grads = proximity_gradient_sream(game, 4)
        self.time_grads = time_gradient_stream(game, 4)

    def prime(self):
        next(self.proxy_grads), next(self.time_grads)

    def sample_vector_direct_vectors(self, state, prev_reward):
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

    def sample_vector_grad_raw(self, state, prev_reward):
        state = self.game.statistics()
        danger = self.game.meandist / self.game.maxdist
        if state[-1] > danger / 2.:
            # descend on the square directly
            return normalize(state[2:4] - state[:2], k=self.speed)[0]
        return -normalize(calculate_gradient(self.game, 2*self.scale, 1), k=self.speed)[0]

    def sample_vector_grad_momentum(self, state, prev_reward):
        state = self.game.statistics()
        grad_vec = calculate_gradient(self.game, 2*self.scale, 1)
        square_vec = np.sign(state[2:4] - state[:2])
        epsilon_vec = np.random.uniform(-self.speed, self.speed, 2)

        decay_v = 0.8
        self.velocity *= decay_v
        danger = self.game.meandist / self.game.maxdist
        if state[-1] > danger / 5:
            # descend on the square directly if no enemies are close
            self.velocity -= square_vec
        self.velocity += 0.3*epsilon_vec + grad_vec
        self.velocity = np.clip(self.velocity, -self.speed, self.speed)

        return -self.velocity

    def sample_vector_grad_momentum2(self, state, prev_reward):
        state = self.game.statistics()
        pgrad, tgrad, square_vec, epsilon_vec = normalize(
            next(self.proxy_grads),
            next(self.time_grads),
            state[:2] - state[2:4],
            np.random.uniform(size=2)
        )

        pgcoef = 1.0
        tgcoef = 1.0
        closest_enemy_distance = 1. / state[-1]**0.2
        decay_v = 0.9
        epsilon_factor = 0.2

        delta = (
            epsilon_vec * epsilon_factor +
            tgrad*tgcoef + pgrad*pgcoef +
            square_vec * closest_enemy_distance
        )

        self.velocity *= decay_v
        self.velocity += np.nan_to_num(normalize(delta*self.speed, k=5)[0])
        self.velocity = np.clip(self.velocity, -self.speed, self.speed)

        return -self.velocity

    sample_vector = sample_vector_grad_momentum2
