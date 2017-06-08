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

from os.path import exists
from environment import Game, agent

#####################
# Parameters to set #
#####################

# Some colors in RGB (0-255)
BLUE = (0, 0, 200)
DARK_GREY = (50, 50, 50)
LIGHT_GREY = (100, 100, 100)

# Parameters of the environment
fps = 210  # the higher, the faster the game's pace
player_speed = 7  # the higher, the faster the player

# State determines what input we give to the neural net
# can be either one of the following:
# - "statistics" gives coordinates and distances of entities
# - "pixels" gives pixel values
state = "pixels"

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
# "keras" is the Keras ANN-driven agent,
# "spazz" is a random-moving agent,
# "math" is the math-driven parametric agent.
# "manual" is a controllable agent
# "recorded" records your actions while you play.
# "online" learns as YOU play and also records your actions.
# "saved" prompts you to select a saved agent.
agent_type = "q"

# Please set these if you intend to use one of the recurrent
# or convolutional layer architectures in Keras.
# Some recurrent layers worth mentioning:
# - LSTM: recurrence with internal memory cells
# - GRU: similar to LSTM, but a more recent architecture
# - ClockworkRNN: similar to RLayer, but faster and better
# Some notes on convolutional architectures:
# - Please use the keras_ann() function to construct convolutional
#   networks!
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
          "Build a different architecture or use Brainforge!"
    raise msg
if agent_type == "online" and state != "pixels":
    msg = "Online agent only supports 'pixels' mode!"
    raise RuntimeError(msg)


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

    if agent_type == "online":
        inshape = (screen[0] * screen[1]) // (16 * GENERAL_SCALING_FACTOR**2)
        if exists("online.agent"):
            return Network.load("online.agent")
    net = Network(inshape, layers=[
        Dense(neurons=300, lmbd=0.0), Tanh(),
        Dense(neurons=outshape, lmbd=0.0)
    ])
    return net


def keras_ann(inshape, outshape):
    from keras.models import Sequential
    from keras.layers import Dense

    inshape = inshape[0]*inshape[1]

    outact = "linear" if agent_type == "q" else "softmax"
    cost = "mse" if agent_type == "q" else "categorical_crossentropy"

    model = Sequential([
        Dense(300, activation="relu", input_dim=inshape),
        Dense(120, activation="relu"),
        Dense(outshape, activation=outact)
    ])
    model.compile(optimizer="adam", loss=cost)
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
        "clever": agent.PolicyLearningAgent,
        "keras": agent.KerasAgent,
        "manual": agent.ManualAgent,
        "recorded": agent.RecordAgent,
        "saved": agent.SavedAgent,
        "online": agent.OnlineAgent,
        "spazz": agent.SpazzAgent,
        "q": agent.QLearningAgent,
        "math": agent.MathAgent
    }[agent_type](game=env, speed=player_speed, network=network,
                  scale=GENERAL_SCALING_FACTOR)
    if agent_type in ("clever", "forged", "online"):
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
        "keras": keras_ann,
        "online": build_ann,
        "saved": build_ann,
        "q": keras_ann
    }.get(agent_type, lambda *args: None)

    # the "agent" variable name was already taken by the "agent" module :(
    actor = get_agent(
        env=env,
        get_network=get_ann
    )
    env.reset(actor)
    env.mainloop()


if __name__ == '__main__':
    main()
