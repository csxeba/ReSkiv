from os.path import exists

from environment import Game, NoEnemyGame, NoSquareGame, agent
from utilities import *

# Parameters of the environment
fps = 30  # the higher, the faster the game's pace
player_speed = 7  # the higher, the faster the player
screen = "300x200"
player_size, player_color = 10, DARK_GREY
square_size, square_color = 10, LIGHT_GREY
enemy_size, enemy_color = 5, BLUE
headless = False

# The environment can be simplified.
# - "nosquare" generates no squares, only enemies and the goal is survival
# - "noenemy" generates no enemies, only the square, the goal is to reach the square
game_simplification = "noenemy"

# Set this to make everything bigger
# Be careful, setting to a non-integer
# may break everything...
GENERAL_SCALING_FACTOR = 1

# Parameters for learning
# Clever agent types are as follows:
# - "policy" is for direct policy learning (RL), learns a direct stochastic policy
# - "dqn" is for Deep Q Learning (RL), approximates Q (quality) for every action given a state
# - "online" learns a direct policy from a human playing the game (SL)
agent_type = "policy"

# Set the neural network type
# - "FC" is fully connected, vanilla ANN (no lib dependency)
# - "CNN" is a LeNet-like convolutional NN (keras & theano/tensorflow lib dependecies)
ann_type = "FC"

# State determines what input we give to the neural net
# can be either one of the following:
# - "statistics" gives coordinates and distances of entities
# - "pixels" gives pixel values
# - "proximity" gives the pixel values around the player
state = "pixels"
downsampling = True  # strongly suggested. Reduces all input dims by a factor of 4


player_speed, player_size, square_size, enemy_size = map(
    lambda x: GENERAL_SCALING_FACTOR*x,
    (player_speed, player_size, square_size, enemy_size)
)
screen = tuple(map(lambda x: int(x)*GENERAL_SCALING_FACTOR, screen.split("x")))


def dense_ann(inshape, outshape):
    from learning.ann import Network, Dense, Tanh, Softmax

    if agent_type == "online":
        inshape = (screen[0] * screen[1]) // (16 * GENERAL_SCALING_FACTOR**2)
        if exists("online.agent"):
            return Network.load("online.agent")
    layers = [Dense(neurons=300, lmbd=0.0), Tanh(),
              Dense(neurons=outshape, lmbd=0.0)]
    if agent_type != "q":
        layers.append(Softmax())

    return Network(inshape, layers=layers, optimizer="adam")


def convolutional_ann(inshape, outshape):
    from keras.models import Sequential
    from keras.layers import Dense, Conv2D, MaxPool2D, Activation, Flatten

    outact = "linear" if agent_type == "dqn" else "softmax"
    loss = "mse" if agent_type == "dqn" else "categorical_crossentropy"
    # inputs: 1x75x50
    network = Sequential([
        Conv2D(6, (3, 3), activation="relu",  # 6x72x48
               input_shape=inshape, data_format="channels_first"),
        Conv2D(8, (3, 3), data_format="channels_first"),  # 8x70x46
        MaxPool2D(),  # 8x35x23
        Activation("relu"),
        Conv2D(8, (6, 4), activation="relu", data_format="channels_first"),  # 8x30x20
        Conv2D(12, (3, 3), data_format="channels_first"),  # 8x28x18
        MaxPool2D(),  # 12x14x9
        Activation("relu"),
        Conv2D(12, (3, 4), activation="relu", data_format="channels_first"),  # 12x12x6
        Conv2D(3, (3, 3), activation="relu", data_format="channels_first"),  # 3x10x4
        Flatten(),
        Dense(300, activation="tanh"),
        Dense(120, activation="tanh"),
        Dense(outshape, activation=outact)
    ])
    network.compile("adam", loss)
    return network


def get_agent(env, get_network):
    network = get_network(env.data_shape, len(env.actions))
    actor = {
        "policy": agent.PolicyLearningAgent,
        "online": agent.OnlineAgent,
        "dqn": agent.QLearningAgent,
    }[agent_type](game=env, speed=player_speed, network=network,
                  scale=GENERAL_SCALING_FACTOR)
    return actor


def get_game():
    return {
        "nosquare": NoSquareGame,
        "noenemy": NoEnemyGame
    }.get(game_simplification, Game)(
        fps=fps, screensize=screen, state=state,
        playersize=player_size, playercolor=player_color,
        enemysize=enemy_size, enemycolor=enemy_color,
        squaresize=square_size, squarecolor=square_color,
        headless=headless
    )


def main():
    env = get_game()
    actor = get_agent(env, dense_ann if ann_type == "FC" else convolutional_ann)
    env.reset(actor)
    env.mainloop()


if __name__ == '__main__':
    main()
