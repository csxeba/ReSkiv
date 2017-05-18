#! /bin/python3

"""
Use this script to run the experiment.
What is happening here?
Basically in a Reinforcement Learning setting,
you have two things interacting with each other:
- Environment: the game itself
- Agent/Actor: an entity doing stuff in the environment

The environment produces a state at every timestep.
The agent takes the state and produces an action.
The environment takes the action and produces a reward.
And the cycle continues.

One of the ways an agent can be taught is by utilizing
Artificial Neural Networks. The network helps the agent
chose an action each timestep. These types of settings
are called Stochastic Policy Networks.
Stochastic means that the network
Policy, by the way, means the logic behind choosing an
action.
"""

from environment import Game, agent

#####################
# Parameters to set #
#####################

# Some colors in RGB (0-255)
BLUE = (0, 0, 200)
DARK_GREY = (50, 50, 50)
LIGHT_GREY = (100, 100, 100)

# Parameters of the environment
fps = 60  # the higher, the faster the game's pace
player_speed = 7  # the higher, the faster the player

# State determines what input we give to the neural net
# can be either one of the following:
# - "statistics" gives coordinates and distances of entities
# - "pixels" gives pixel values
state = "statistics"

# Because of a 4-wise downsampling, each dimension has to be divisible by 4!
screen = "500x400"
player_size, player_color = 10, DARK_GREY
square_size, square_color = 10, LIGHT_GREY
enemy_size, enemy_color = 5, BLUE

# Set this to make everything bigger
# Be careful, setting to a non-integer
# may break everything...
GENERAL_SCALING_FACTOR = 1

player_speed, player_size, square_size, enemy_size = map(
    lambda x: GENERAL_SCALING_FACTOR*x,
    (player_speed, player_size, square_size, enemy_size)
)
screen = tuple(map(lambda x: int(x)*GENERAL_SCALING_FACTOR, screen.split("x")))

# Please note two things about colors and screen size:
# If you use the pixel values to teach the network,
# the frames from the game are preprocessed.
# - first, every frame is downsampled by a factor of 4
# - second, the red and green channels are omitted
# So pay attention to only set the screen to sizes divisible by 4
# and to set the colors so, that the entities (enemies, player, square)
# can be distinguished by the blue channel (the last number)

# Can be one of [clever, forged, keras, manual, spazz, math]
# Where "clever" is the builtin ANN-driven agent,
# "forged" is the Brainforged ANN-driven agent,
# "keras" is the Keras ANN-driven agent,
# "spazz" is a random-moving agent,
# "math" is the math-driven parametric agent.
# "manual" is a controllable agent
# "recorded" is a controllable agent, whose actions are recorded.
agent_type = "math"

# Please set these if you intend to use one of the recurrent
# or convolutional layer architectures in Brainforge/Keras.
# Some recurrent layers worth mentioning:
# - LSTM: recurrence with internal memory cells
# - GRU: similar to LSTM, but a more recent architecture
# - ClockworkRNN: similar to RLayer, but faster and better
# Some notes on convolutional architectures:
# - Please use the keras_ann() function to construct convolutional
#   networks! Brainforge's ConvLayer is unstable and slow.
# - It's not recommended to use pooling layers in RL configurations.
#   Stick to multiple Conv2D layers instead!

agent_is_convolutional = False  # makes sure images are reshaped for convolutions
agent_is_recurrent = False  # makes sure data is reshaped for recurrence


#####################
# Sanity check here #
#####################

if state == "pixels" and not all(d % 4 == 0 for d in screen):
    msg = "Screen dimensions have to be divisible " + \
          "by 4 if pixels are used as state!"
    raise RuntimeError(msg)
if state == "pixels" and agent_is_recurrent:
    msg = "Pixel-mode is not compatible with recurrence!"
    raise RuntimeError(msg)
if state == "statistics" and agent_is_convolutional:
    msg = "Statistics mode is not compatible with convolutional architectures!"
    raise RuntimeError(msg)
assert not all((agent_is_recurrent, agent_is_convolutional)), \
    "Choose either a Recurrent or a Convolutional architecture!"
assert GENERAL_SCALING_FACTOR >= 1, \
    "The GENERAL_SCALING_FACTOR should be >= 1"
if agent_type == "keras" and agent_is_recurrent:
    msg = "Keras doesn't support variable-time recurrence." \
          "Select a different architecture or use Brainforge!"
    raise msg
if agent_type == "forged" and agent_is_convolutional:
    msg = "Convolution is unstable in brainforge." \
          "Please consider using Keras instead!"
    raise msg


#######################
# Some setup routines #
#######################

def build_ann(inshape, outshape):
    """
    Initialize the ANN with the builtin ANN lib
    Possible layers:
    - Dense: normal fully connected layer
    - Tanh: applies tanh nonlinearity
    - ReLU: applies ReLU nonlinearity
    The output activation is fixed to softmax, so
    be careful! ReLU can't be put immediately before
    the output layer, because it causes overflow error.
    """
    from learning.ann import Network, Dense, Tanh

    net = Network(inshape, layers=[
        Dense(neurons=200, lmbd=0.0), Tanh(),
        Dense(neurons=outshape, lmbd=0.0)
    ])
    return net


def forge_ann(inshape, outshape):
    """
    Use Brainforge to create an ANN.
    Brainforge is my ANN lib. See: https://github.com/csxeba/brainforge
    It has much more functionality than this project's "learning" submodule.
    I include a [fairly :)] stable version in case you want to experiment
    with more advanced architectures.
    (And because I couldn't get Keras to work...)
    There are some decent recurrent layer implementations like LSTM, GRU,
    ClockworkRNN. The ConvLayer is not very stable and it's also quite slow.
    
    A note on recurrent architectures:
    If you use a one of the recurrent layers, you'll need to preprocess the
    input data a bit more. This is in addition to the pixel subsampling.
    """
    from brainforge import Network
    from brainforge.layers import DenseLayer
    brain = Network(inshape, layers=(
        DenseLayer(neurons=200, activation="tanh"),
        DenseLayer(outshape, activation="softmax")
    ))

    # Attention! The implicit optimizer of the network won't be used,
    # the agent will have its own optimizer, which is used instead!
    # So don't set finalize()'s parameters, they won't be utilized.
    brain.finalize("xent")
    return brain


def keras_ann(inshape, outshape):
    from keras.models import Sequential
    from keras.layers import Conv2D, Flatten, Dense

    if agent_is_convolutional:
        inshape = (1, inshape[0] // 4, inshape[1] // 4)

    model = Sequential([
        Conv2D(8, (6, 6), data_format="channels_first",
               activation="relu", input_shape=inshape),
        Conv2D(6, (3, 3), data_format="channels_first",
               activation="relu"),
        Conv2D(4, (3, 3), data_format="channels_first",
               activation="relu"),
        Conv2D(2, (3, 3), data_format="channels_first",
               activation="relu"),
        Flatten(),
        Dense(120, activation="tanh"),
        Dense(outshape, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy")
    return model


def get_agent(env, get_network):
    """
    Three agent types are available:
    - "clever" is controlled by an Artificial Neural Network
    - "manual" can be controlled manually
    - "spazz" moves randomly around
    
    If you use a clever agent, you can set its optimizer
    explicitly. Available algorithms are:
    - SGD: Stochastic Gradient Descent
    - RMSProp: Root Mean Squared Propagation
    - Adam: Adaptive Momentum
    Check this out for details:
    http://sebastianruder.com/optimizing-gradient-descent/
    Please don't use the optimizers from brainforge, they are not
    compatible with the ones I implemented for you.
    """
    from learning import optimization
    network = get_network(env.data_shape, len(env.actions))
    actor = {
        "clever": agent.CleverAgent,
        "forged": agent.CleverAgent,
        "keras": agent.KerasAgent,
        "manual": agent.ManualAgent,
        "recorded": agent.RecordAgent,
        "spazz": agent.SpazzAgent,
        "math": agent.MathAgent
    }[agent_type](game=env, speed=player_speed, network=network,
                  scale=GENERAL_SCALING_FACTOR)
    if agent_type == "clever":
        # Set the optimizer below
        actor.recurrent = agent_is_recurrent
        actor.optimizer = optimization.Adam()
    elif agent_type == "keras":
        actor.recurrent = agent_is_recurrent
        actor.convolutional = agent_is_convolutional
    return actor


def main():
    env = Game(fps=fps, screensize=screen, state=state,
               playersize=player_size, playercolor=player_color,
               enemysize=enemy_size, enemycolor=enemy_color,
               squaresize=square_size, squarecolor=square_color)
    get_ann = {
        "clever": build_ann,
        "forged": forge_ann,
        "keras": keras_ann
    }.get(agent_type, lambda *args: None)

    # the agent name was already taken by the agent module :(
    actor = get_agent(
        env=env,
        get_network=get_ann
    )
    env.reset(actor)
    env.mainloop()


if __name__ == '__main__':
    main()
