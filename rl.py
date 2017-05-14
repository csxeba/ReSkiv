import os

import numpy as np
import pygame
from ann import Network, Dense, ReLU, Tanh
from environment import Game
# hyperparameters
from util import prepro

batch_size = 10  # every how many episodes to do a param update?
learning_rate = 1e-4
decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2

env = Game(ball_type="clever", fps=60, screensize=(450, 400), escape=False)


previous_frame = prepro(env.reset())
current_frame = prepro(env.step(None)[0])
D = previous_frame.size

# model initialization
saved = os.path.join(os.getcwd(), "net.pgz")
if os.path.exists(saved):
    print("-- LOADING SAVED MODEL --")
    model = Network.load(saved)
else:
    print("-- STARTING NEW MODEL --")
    model = Network(D, [Dense(200, lmbd=0.1), ReLU(),
                        Dense(100, lmbd=0.1), Tanh(),
                        Dense(9, lmbd=0.1)])

Xs, hs, fake_Ys, rwrds = [], [], [], []
running_reward = 0
reward_sum = 0
episode_number = 0
r = 0
frames_seen = 1
while True:
    print("\rFrame: {:>5}, Reward: {:>5.2f}".format(frames_seen, reward_sum), end="")
    # preprocess the observation, set input to network to be difference image
    # forward the policy network and sample an action from the returned probability
    probs = model.predict(current_frame[None, :]).ravel()

    # record various intermediates (needed later for backprop)
    Xs.append(current_frame)  # observation
    action, fake_label = env.sample_action(probs)
    fake_Ys.append(fake_label)

    # step the environment and get new measurements
    info = env.step(action)
    if info is None:
        break
    current_frame, reward, done = info
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
        running_reward = (reward_sum if running_reward is 0 else
                          running_reward * 0.99 + reward_sum * 0.01)
        print("\rEpisode {:>5}, running reward: {:>5.5f}"
              .format(episode_number, running_reward))
        reward_sum = 0
        observation = env.reset()
        done = False
        frames_seen = 0

        if episode_number % 100 == 0:
            model.save()

    frames_seen += 1
    env.clock.tick(180)
    pygame.display.flip()
print()
model.save()
print("-- END PROGRAM --")
