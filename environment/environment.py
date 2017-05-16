from environment.ball import *
from utilities.util import calc_meand

BLUE = (0, 0, 200)
DARK_GREY = (50, 50, 50)
LIGHT_GREY = (100, 100, 100)


class Game:

    def __init__(self, fps, screensize, escape_allowed=True, state="statistics",
                 playersize=10, enemysize=5, squaresize=10,
                 playercolor=DARK_GREY, enemycolor=BLUE, squarecolor=LIGHT_GREY,
                 reward_function=None):

        pygame.init()
        self.ballargs = {
            "player": (self, playercolor, playersize),
            "enemy": (self, enemycolor, enemysize),
            "square": (self, squarecolor, squaresize)
        }
        self.size = np.array(screensize, dtype=int)
        self.meandist = calc_meand(self.size)
        self.screen = pygame.display.set_mode(self.size)
        self.clock = pygame.time.Clock()
        self.escape_allowed = escape_allowed
        self.fps = fps
        self.agent = None
        self.player = None
        self.square = None
        self.enemies = None
        self.points = 0.
        self.nopoints = 0
        self.steps_taken = 0
        self.episodes = 0
        self.actions = [(-1, -1), (-1, 0), (-1, 1),
                        (0, -1), (0, 0), (0, 1),
                        (1, -1), (1, 0), (1, 1)]
        self.labels = np.eye(len(self.actions))
        self._state_str = state
        self.state = self.pixels if state == "pixels" else self.statistics
        self.data_shape = tuple(self.size) if state == "pixels" else (5,)
        self.reward_function = (_default_reward_function
                                if reward_function is None else
                                reward_function)

    def sample_action(self, probs):
        arg = np.random.choice(np.arange(len(self.actions)), size=1, p=probs)[0]
        return arg, self.actions[arg], self.labels[arg]

    def pixels(self):
        return pygame.surfarray.array3d(self.screen)

    def statistics(self):
        pcoords = self.player.coords / self.size
        scoords = self.square.coords / self.size
        enemies = min(self.player.distance(enemy) / self.meandist
                      for enemy in self.enemies)
        return np.array(list(pcoords) + list(scoords) + [enemies])

    def reset(self, agent=None):
        self.agent = agent if self.agent is None else self.agent
        if self.agent is None:
            raise RuntimeError("Please instantiate and pass one of the Agents!")
        self.player = PlayerBall(*self.ballargs["player"])
        self.square = Square(*self.ballargs["square"])
        self.enemies = [EnemyBall(*self.ballargs["enemy"])]
        self.points = 0.
        return self.step(np.array([0, 0]))[0]

    def score(self):
        return self.player.touches(self.square)

    def step(self, dvec):

        self.screen.fill((0, 0, 0))
        self.square.draw()
        self.player.move(dvec)

        reward, done = self.reward_function(self)

        for e in self.enemies:
            e.move()
        return self.state(), reward, done

    def mainloop(self, max_steps=3000):
        self.steps_taken = 0  # a step is a single frame
        self.episodes = 1  # an episode is one game
        tock = self.fps
        state = self.reset()  # state is either the pixels or the statistics
        reward = None
        reward_sum = 0.
        running_reward = None
        done = 0
        print("Episode: 1")
        while 1:
            self.steps_taken += 1
            if any([e.type == pygame.QUIT for e in pygame.event.get()]):
                break

            dvec = self.agent.sample_vector(state, reward)
            state, reward, done = self.step(dvec)

            reward_sum += reward
            if running_reward is None:
                running_reward = reward_sum
            if reward:
                print()
                self.agent.accumulate(reward)
            if self.steps_taken >= max_steps:
                print()
                self.agent.accumulate(reward)
                self.steps_taken = 0
            if done:
                self.episodes += 1
                running_reward = running_reward * 0.99 + reward_sum * 0.01
                self.reset()
                print("Episode:", self.episodes, end="")
                self.steps_taken = 0
                if self.episodes % 10 == 0:
                    print(" performing update!")
                    self.agent.update()
                else:
                    print()

            print("\rStep count: {:>5}, Running reward: {: .3f}"
                  .format(self.steps_taken, running_reward),
                  end="")

            self.clock.tick(tock)
            pygame.display.flip()

        if not done:
            self.agent.update(reward)
        self.agent.network.save()
        print("\n-- END PROGRAM --")


def _default_reward_function(environment):
    done = 0
    rwd = 0.
    if environment.score():
        environment.square = Square(*environment.ballargs["square"])
        environment.enemies.append(EnemyBall(*environment.ballargs["enemy"]))
        environment.points += 1.
        rwd = 1.
    if environment.player.dead():
        done = 1
        rwd = -0.5
    if not environment.escape_allowed:
        if environment.player.escaping():
            done = 1
            rwd = -0.5
    return rwd, done
