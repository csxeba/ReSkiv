from environment.ball import *
from utilities.util import calc_meand, downsample

BLUE = (0, 0, 200)
DARK_GREY = (50, 50, 50)
LIGHT_GREY = (100, 100, 100)


class Game:

    def __init__(self, fps, screensize, state="statistics",
                 playersize=10, enemysize=5, squaresize=10,
                 playercolor=DARK_GREY, enemycolor=BLUE, squarecolor=LIGHT_GREY,
                 reward_function=None, downsmpl=True):

        pygame.init()
        self.ballargs = {
            "player": (self, playercolor, playersize),
            "enemy": (self, enemycolor, enemysize),
            "square": (self, squarecolor, squaresize)
        }
        self.size = np.array(screensize, dtype=int)
        self.meandist = calc_meand(self.size)
        self.maxdist = np.linalg.norm(self.size)
        self.screen = pygame.display.set_mode(self.size)
        self.clock = pygame.time.Clock()
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
                        ( 0, -1), ( 0, 0), ( 0, 1),
                        (+1, -1), (+1, 0), (+1, 1)]
        self.labels = np.eye(len(self.actions))
        if state not in ("statistics", "pixels"):
            raise RuntimeError("State should be one of: [statistics, pixels]")
        self._state_str = state
        self.state = self.pixels if state == "pixels" else self.statistics
        self.data_shape = (
            tuple(self.size // (4 if downsmpl else 1))
            if state == "pixels" else (5,)
        )
        self.reward_function = (
            _default_reward_function if reward_function is None else reward_function
        )
        self.downsample = downsmpl

    def sample_action(self, prob):
        arg = np.random.choice(np.arange(len(self.actions)), size=1, p=prob)[0]
        actions = self.actions[arg]
        labels = self.labels[arg]
        return actions, labels

    def pixels(self):
        state = pygame.surfarray.array3d(self.screen)
        if self.downsample:
            state = downsample(state)
        return state

    def statistics(self):
        pcoords = self.player.coords / self.size
        scoords = self.square.coords / self.size
        enemies = min(self.player.distance(enemy) / self.maxdist
                      for enemy in self.enemies)
        return np.array(list(pcoords) + list(scoords) + [enemies])

    def reset(self, agent=None):
        self.agent = agent if self.agent is None else self.agent
        if self.agent is None:
            raise RuntimeError("Please instantiate and pass one of the Agents!")
        self.player = PlayerBall(*self.ballargs["player"]).draw()
        self.square = Square(*self.ballargs["square"]).draw()
        self.enemies = [EnemyBall(*self.ballargs["enemy"]).draw() for _ in range(1)]
        self.points = 0.
        return self.step(np.array([0, 0]))[0]

    def score(self):
        return self.player.touches(self.square)

    def step(self, dvec):

        self.screen.fill((0, 0, 0))
        # hills = pygame.pixelcopy.make_surface(steep_hills(self))
        # self.screen.blit(hills, (0, 0))

        self.square.draw()
        self.player.move(dvec)
        self.player.draw()

        reward, done = self.reward_function(self)

        for e in self.enemies:
            e.move()
            e.draw()
        return self.state(), reward, done

    def progression(self, tock):
        self.clock.tick(tock)
        pygame.display.flip()

    def check_quit(self):
        if any([e.type == pygame.QUIT for e in pygame.event.get()]):
            return True
        return False

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

            if self.check_quit():
                break

            action = self.agent.sample_vector(state, reward)
            state, reward, done = self.step(action)

            reward_sum += reward
            if running_reward is None:
                running_reward = reward_sum

            print("\rStep count: {:>5}, Current reward: {: .3f}, Current points: {:.0f}"
                  .format(self.steps_taken, reward, self.points),
                  end="")

            if self.steps_taken >= max_steps or done:
                self.episodes += 1
                print()
                self.agent.accumulate(reward)
                running_reward = running_reward * 0.99 + reward_sum * 0.01
                reward_sum = 0
                self.steps_taken = 0
                if done:
                    self.reset()
                if self.episodes % 10 == 0:
                    self.agent.update()
                print("\nEpisode: {} Running reward: {: .3f}"
                      .format(self.episodes, running_reward))

            self.progression(tock)

        if not done:
            print()
            self.agent.update()
        if self.agent.type == "clever":
            self.agent.network.save()
        print("\n-- END PROGRAM --")


class Headless(Game):

    def __init__(self, fps, screensize, state="statistics",
                 playersize=10, enemysize=5, squaresize=10,
                 playercolor=DARK_GREY, enemycolor=BLUE, squarecolor=LIGHT_GREY,
                 reward_function=None):

        self.ballargs = {
            "player": (self, playercolor, playersize),
            "enemy": (self, enemycolor, enemysize),
            "square": (self, squarecolor, squaresize)
        }
        self.size = np.array(screensize, dtype=int)
        self.meandist = calc_meand(self.size)
        self.maxdist = np.linalg.norm(self.size)
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
        if state != "statistics":
            raise RuntimeError("Headless mode only supports 'statistics' state!")
        self._state_str = state
        self.state = self.statistics
        self.data_shape = (5,)
        self.reward_function = (_default_reward_function
                                if reward_function is None else
                                reward_function)

    def pixels(self):
        raise NotImplementedError

    def reset(self, agent=None):
        self.agent = agent if self.agent is None else self.agent
        if self.agent is None:
            raise RuntimeError("Please instantiate and pass one of the Agents!")
        self.player = PlayerBall(*self.ballargs["player"])
        self.square = Square(*self.ballargs["square"])
        self.enemies = [EnemyBall(*self.ballargs["enemy"]) for _ in range(1)]
        self.points = 0.
        return self.step(np.array([0, 0]))[0]

    def progression(self, tock):
        pass

    def check_quit(self):
        pass

    def step(self, dvec):

        self.square.draw()
        self.player.move(dvec)

        reward, done = self.reward_function(self)

        for e in self.enemies:
            e.move()
        return self.state(), reward, done


def _default_reward_function(env):
    done = 0
    rwd = 0.5 - env.player.distance(env.square) / env.maxdist
    if env.score():
        env.square = Square(*env.ballargs["square"])
        env.enemies.append(EnemyBall(*env.ballargs["enemy"]))
        env.points += 5.
        rwd = 9.
    if env.player.dead():
        done = 1
        rwd = -2.
    if env.player.escaping():
        rwd = -2.
    return rwd, done
