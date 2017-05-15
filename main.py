#! /bin/python3

import sys
from os import path
from argparse import ArgumentParser

from learning.agents import *
from environment import Game


PLAYERSPEED = 7
DEFAULT_SCREEN = "500x400"


def invalid_screen_size():
    print("Invalid screen size specification, sticking to default!")
    return tuple(map(int, DEFAULT_SCREEN.split("x")))


def reparse_screensize(ns):
    try:
        ns.screen = tuple(map(int, ns.screen[0].split("x")))
    except ValueError:
        ns.screen = invalid_screen_size()
    if len(ns.screen) > 2 or len(ns.screen) < 1:
        ns.screen = invalid_screen_size()
    elif len(ns.screen) == 1:
        ns.screen = (ns.screen[0], ns.screen[0])
    return ns


def build_brain(inshape, outshape, lambda_):
    from learning.ann import Network
    inshape = np.prod(inshape) // 16
    return Network.default(inshape, outshape, lmbd_global=lambda_)


def forge_brain(inshape, outshape):
    from brainforge import Network
    from brainforge.layers import DenseLayer
    inshape = np.prod(inshape) // 16
    brain = Network(inshape, layers=(
        DenseLayer(200, activation="tanh"),
        DenseLayer(60, activation="tanh"),
        DenseLayer(outshape, activation="softmax")
    ))
    brain.finalize("xent", "rmsprop")
    return brain


def forge_convnet(inshape, outshape):
    from brainforge import Network
    from brainforge.layers import (
        ConvLayer, Flatten, DenseLayer
    )
    inshape = [1] + [sh // 4 for sh in inshape]
    brain = Network(inshape, layers=[
        ConvLayer(3, 9, 9, activation="tanh"),
        ConvLayer(1, 3, 3, activation="tanh"),
        ConvLayer(1, 3, 3, activation="tanh"),
        Flatten(), DenseLayer(200, activation="tanh"),
        DenseLayer(60, activation="tanh"),
        DenseLayer(outshape, activation="softmax")
    ])
    brain.finalize("xent", "rmsprop")
    return brain


def resolve_agent_type(env, ns):
    from learning.ann import Network
    agentspec = ns.agent[0]
    if agentspec[-4:] == ".pgz":
        return CleverAgent(
            env, PLAYERSPEED, network=Network.load(agentspec)
        )
    agent = {
        "clever": CleverAgent(env, PLAYERSPEED, forge_convnet(
            inshape=ns.screen,
            outshape=len(env.actions))
        ),
        "human": ManualAgent(env, PLAYERSPEED, None),
        "spazz": SpazzAgent(env, PLAYERSPEED, None)
    }.get(agentspec, None)
    if agent is None:
        raise ValueError("Invalid agent type: " + agentspec)
    return agent


def parse_args():
    parser = ArgumentParser(
        prog="python3 " + path.split(sys.argv[0])[-1],
        description="Reimplementation of the arcade game: Eskiv"
    )
    parser.add_argument("--agent", nargs=1, default=["human"], metavar="",
                        help="`human` is controllable, " +
                             "`clever` creates a new ANN, " +
                             "`spazz` is random, " +
                             "or you can specify a path to a .pgz saved network."
                             "Defaults to `human`")
    parser.add_argument("--screen", nargs=1, default=[DEFAULT_SCREEN], metavar="",
                        help="supply screen size so: WIDTHxHEIGHT " +
                             "for a square size. +"
                             "Defaults to " + DEFAULT_SCREEN)
    parser.add_argument("--fps", nargs=1, default=[30], metavar="", type=int,
                        help="supply the FPS (Frames Per Second). " +
                             "Defaults to 30")

    ns = parser.parse_args(sys.argv[1:])
    ns = reparse_screensize(ns)

    return ns


def main():
    args = parse_args()
    env = Game(args.fps[0], args.screen, escape_allowed=False)
    agent = resolve_agent_type(env, args)
    env.reset(agent)
    env.mainloop()


if __name__ == '__main__':
    main()
