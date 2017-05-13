#! /bin/python3

import sys
from os import path
from argparse import ArgumentParser

from environment import Game


def invalid_screen_size():
    print("Invalid screen size specification, sticking to default!")
    return 600, 600


def parse_args():
    parser = ArgumentParser(
        prog="python3 " + path.split(sys.argv[0])[-1],
        description="Reimplementation of the arcade game: Eskiv"
    )
    parser.add_argument("--ball-type", nargs=1, default=["manual"], metavar="",
                        choices=("manual", "clever", "spazz"),
                        help="`manual` is controllable, " +
                             "`clever` is ANN driven, " +
                             "`spazz` is brownian (random). " +
                             "Defaults to `manual`")
    parser.add_argument("--screen", nargs=1, default=["450x400"], metavar="",
                        help="supply screen size so: WIDTHxHEIGHT or simply WIDTH " +
                        "for a square size. +"
                        "Defaults to 600x600")
    parser.add_argument("--fps", nargs=1, default=[30], metavar="", type=int,
                        help="supply the FPS (Frames Per Second). " +
                             "Defaults to 30")
    ns = parser.parse_args(sys.argv[1:])

    try:
        ns.screen = tuple(map(int, ns.screen[0].split("x")))
    except ValueError:
        ns.screen = invalid_screen_size()
    if len(ns.screen) > 2 or len(ns.screen) < 1:
        ns.screen = invalid_screen_size()
    elif len(ns.screen) == 1:
        ns.screen = (ns.screen[0], ns.screen[0])
    return ns


def main():
    args = parse_args()
    env = Game(args.ball_type[0], args.fps[0], args.screen)
    env.mainloop()


if __name__ == '__main__':
    main()
