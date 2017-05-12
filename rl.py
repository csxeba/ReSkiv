import numpy as np
import pygame

from environment import Game

# hyperparameters
H = 200  # number of hidden layer neurons
batch_size = 10  # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99  # discount factor for reward
decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2


def mse_derivative(A, Y):
    return Y - A


def prepro(I):
    I = I[::4, ::4, 0]  # downsample by factor of 4
    I[I != 0] = 1  # set foreground to 1
    return I.astype(np.float).ravel()


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(r.size)):
        if r[t] != 0:
            running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def policy_forward(x):
    h = x @ model["W1"] + model["b1"]
    h[h < 0.] = 0.
    dvec = h @ model["W2"] + model["b2"]
    return dvec, h


def policy_backward(ep_hstates, ep_dlogprob):
    """ backward pass. (eph is array of intermediate hidden states) """
    dW2 = ep_hstates.T @ ep_dlogprob
    db2 = ep_dlogprob.sum(axis=0)
    dh = ep_dlogprob @ model["W2"].T
    dh[ep_hstates <= 0] = 0  # backpro prelu
    dW1 = ep_inputs.T @ dh
    db1 = dh.sum(axis=0)
    return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}


env = Game()
prev_x = prepro(env.reset())
D = prev_x.size
# model initialization
model = {'W1': np.random.randn(D, H) / np.sqrt(D),
         'b1': np.random.randn(H,) * 4,
         'W2': np.random.randn(H, 2) / np.sqrt(H),
         "b2": np.random.randn(2,) * 4}

# update buffers that add up gradients over a batch
grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}
# rmsprop memory
rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}

inputs, hs, dlogps, drs = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0
cur_x = prepro(env.step(None)[0])
r = 0
while True:
    # preprocess the observation, set input to network to be difference image
    inpt = cur_x - prev_x
    prev_x = cur_x

    # forward the policy network and sample an action from the returned probability
    output, hstate = policy_forward(inpt)

    # record various intermediates (needed later for backprop)
    inputs.append(inpt)  # observation
    hs.append(hstate)  # hidden state

    # grad that encourages the action that was taken to be taken
    # (see http://cs231n.github.io/neural-networks-2/#losses if confused)
    dlogps.append(output)

    # step the environment and get new measurements
    cur_x, reward, done = env.step(output)
    cur_x = prepro(cur_x)
    reward_sum += reward

    drs.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)

    if done:  # an episode finished
        episode_number += 1

        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        ep_inputs = np.vstack(inputs)
        ep_hstates = np.vstack(hs)
        ep_dlogprob = np.vstack(dlogps)
        ep_rewards = np.vstack(drs)
        inputs, hs, dlogps, drs = [], [], [], []  # reset array memory

        # compute the discounted reward backwards through time
        discounted_epr = discount_rewards(ep_rewards)
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_epr -= discounted_epr.mean()
        discounted_epr /= discounted_epr.std()

        ep_dlogprob *= discounted_epr  # modulate the gradient with advantage (PG magic happens right here.)
        grad = policy_backward(ep_hstates, ep_dlogprob)
        for k in model:
            grad_buffer[k] += grad[k]  # accumulate grad over batch

        # perform rmsprop parameter update every batch_size episodes
        if episode_number % batch_size == 0:
            for k, v in model.items():
                g = grad_buffer[k]  # gradient
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g ** 2
                model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer

        # boring book-keeping
        running_reward = (reward_sum if running_reward is None else
                          running_reward * 0.99 + reward_sum * 0.01)
        print("\rEpisode {}, total reward: {}"
              .format(episode_number, reward_sum))
        reward_sum = 0
        observation = env.reset()

    env.clock.tick(30)
    pygame.display.flip()
