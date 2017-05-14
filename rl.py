import numpy as np
import pygame

from ann import Network, Dense, ReLU

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


def discount_rewards(rwd):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(rwd)
    running_add = 0
    for t in reversed(range(rwd.size)):
        if rwd[t] != 0:
            running_add = 0  # reset the sum, since this was a game boundary
        running_add = running_add * gamma + rwd[t]
        discounted_r[t] = running_add
    return discounted_r


env = Game(ball_type="clever", fps=60, screensize=(450, 400), escape=False)
actions = [(-1, -1), (-1, 0), (-1, 1),
           (0, -1), (0, 0), (0, 1),
           (1, -1), (1, 0), (1, 1)]
fake_labels = np.eye(len(actions))

previous_frame = prepro(env.reset())
current_frame = prepro(env.step(None)[0])
D = previous_frame.size

# model initialization
model = Network(D, [Dense(H), ReLU(), Dense(9)])

Xs, hs, fake_Ys, rwrds = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0
r = 0
frames_seen = 1
while True:
    print("\rFrame: {:>5}, Reward: {:>5}".format(frames_seen, reward_sum), end="")
    # preprocess the observation, set input to network to be difference image
    inpt = current_frame - previous_frame
    previous_frame = current_frame

    # forward the policy network and sample an action from the returned probability
    probs = model.predict(inpt[None, :]).ravel()

    # record various intermediates (needed later for backprop)
    Xs.append(inpt)  # observation
    arg_action = np.random.choice(np.arange(len(actions)), size=1, p=probs)
    action = np.array(actions[arg_action[0]])
    fake_Ys.append(fake_labels[arg_action])

    # step the environment and get new measurements
    current_frame, reward, done = env.step(action)
    current_frame = prepro(current_frame)
    reward_sum += reward

    rwrds.append(reward)  # record reward

    if done:
        episode_number += 1

        ep_inputs = np.vstack(Xs)
        ep_fake_Ys = np.vstack(fake_Ys)
        ep_rewards = np.vstack(rwrds)
        Xs, hs, fake_Ys, rwrds = [], [], [], []  # reset array memory

        model.learn_batch(ep_inputs, ep_fake_Ys, ep_rewards, learning_rate, decay_rate)

        # boring book-keeping
        running_reward = (reward_sum if running_reward is None else
                          running_reward * 0.99 + reward_sum * 0.01)
        print("\rEpisode {}, total reward: {}"
              .format(episode_number, reward_sum))
        reward_sum = 0
        observation = env.reset()
        done = False

    frames_seen += 1
    env.clock.tick(30)
    pygame.display.flip()
