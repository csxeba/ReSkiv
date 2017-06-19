from environment import Game
from environment.agent import MathAgent
from utilities import *

# Parameters of the environment
fps = 30  # the higher, the faster the game's pace
player_speed = 7  # the higher, the faster the player
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

env = Game(fps=fps, screensize=screen, state="statistics",
           playersize=player_size, playercolor=player_color,
           enemysize=enemy_size, enemycolor=enemy_color,
           squaresize=square_size, squarecolor=square_color,
           headless=False)
agent = MathAgent(env, player_speed, None, GENERAL_SCALING_FACTOR)
env.reset(agent)
env.mainloop()
