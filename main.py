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

from environment import Game
from learning import agent

# Some colors in RGB (0-255)
BLUE = (0, 0, 200)
DARK_GREY = (50, 50, 50)
LIGHT_GREY = (100, 100, 100)


# Parameters of the environment
fps = 120  # the higher, the faster the game's pace
player_speed = 7  # the higher, the faster the player
boundaries_kill = True  # whether touching the game boundary kills the player

# state determines what input we give to the neural net
# can be either one of the following:
# - "statistics" gives coordinates and distances of entities
# - "pixels" gives pixel values
state = "statistics"

screen = "500x400"  # each dimension has to be divisible by 4!
screen = tuple(map(int, screen.split("x")))

# Please not two things about colors and screen size:
# If you use the pixel values to teach the network,
# the frames from the game are preprocessed.
# - first, every frame is downsampled by a factor of 4
# - second, the red and green channels are omitted
# So pay attention to only set the screen to sizes divisible by 4
# and to set the colors so, that the entities (enemies, player, square)
# can be distinguished by the blue channel (the last number)

agent_type = "clever"  # Can be one of [clever, manual, spazz]

# Please set this if you intend to use one of the recurrent
# layer architectures in Brainforge.
# These are the following:
# - RLayer: simple recurrent layer, receives it's previous
#   outputs as inputs
# - LSTM: recurrence with internal memory cells
# - GRU: similar to LSTM, but a more recent architecture
# - ClockworkRNN: similar to RLayer, but faster and better
agent_is_recurrent = False

if state == "pixels" and not all(d % 4 == 0 for d in screen):
    msg = "Screen dimensions have to be divisible " + \
          "by 4 if pixels are used as state!"
    raise RuntimeError()


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
    from brainforge.layers import DenseLayer, DropOut
    brain = Network(inshape, layers=(
        DenseLayer(neurons=300, activation="sigmoid"),
        DropOut(dropchance=0.5),
        DenseLayer(neurons=120, activation="sigmoid"),
        DropOut(dropchance=0.5),
        DenseLayer(outshape, activation="softmax")
    ))

    # Attention! The implicit optimizer of the network won't be used,
    # the agent will have its own optimizer, which is used instead!
    # So don't set finalize()'s parameters, they won't be utilized.
    brain.finalize("xent")
    return brain


def get_agent(environment, network=None):
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
    from learning import optimizer
    actor = {
        "clever": agent.CleverAgent,
        "manual": agent.ManualAgent,
        "spazz": agent.SpazzAgent
    }[agent_type](game=environment, speed=player_speed, network=network)
    if agent_type == "clever":
        # Set the optimizer below
        actor.recurrent = agent_is_recurrent
        actor.optimizer = optimizer.Adam()
    return actor


def main():
    env = Game(fps=fps, screensize=screen, escape_allowed=(not boundaries_kill),
               state="statistics",
               playersize=10, playercolor=DARK_GREY,
               enemysize=5, enemycolor=BLUE,
               squaresize=10, squarecolor=LIGHT_GREY)

    net = build_ann(
        inshape=env.data_shape,
        outshape=len(env.actions)
    )

    # the agent name was already taken by the agent module :(
    actor = get_agent(
        environment=env,
        network=net
    )
    env.reset(actor)
    env.mainloop()


if __name__ == '__main__':
    main()
