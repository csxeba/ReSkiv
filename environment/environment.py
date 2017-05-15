from environment.classes import *
from utilities.util import calc_meand


class Game:

    def __init__(self, fps, screensize, escape_allowed=True):
        pygame.init()
        self.size = np.array(screensize, dtype=int)
        self.meandist = calc_meand(self.size)
        self.screen = pygame.display.set_mode(self.size)
        self.clock = pygame.time.Clock()
        self.agent = None
        self.escape_allowed = escape_allowed
        self.fps = fps
        self.player = None
        self.square = None
        self.enemies = None
        self.points = 0.
        self.nopoint = 1
        self.steps_taken = 0
        self.actions = [(-1, -1), (-1, 0), (-1, 1),
                        (0, -1), (0, 1),
                        (1, -1), (1, 0), (1, 1)]
        self.labels = np.eye(len(self.actions))

    def sample_action(self, probs):
        arg = np.random.choice(np.arange(len(self.actions)), size=1, p=probs)[0]
        return arg, self.actions[arg], self.labels[arg]

    def reset(self, agent=None):
        self.agent = agent if self.agent is None else self.agent
        if self.agent is None:
            raise RuntimeError("Please instantiate and pass one of the Agents!")
        self.player = PlayerBall(self)
        self.square = Square(self)
        self.enemies = [EnemyBall(self)]
        self.points = 0.
        self.steps_taken = 0
        self.nopoint = 1
        return self.step(np.array([0, 0]))[0]

    def score(self):
        return self.player.touches(self.square)

    def step(self, dvec):
        rwd = 0.
        done = 0
        self.screen.fill((0, 0, 0))
        self.square.draw()
        self.player.move(dvec)
        for e in self.enemies:
            e.move()
        if self.score():
            self.square = Square(self)
            self.enemies.append(EnemyBall(self))
            self.steps_taken = 0
            self.points += 1.
            rwd += 1.
        if self.player.dead():
            done = 1
            rwd -= 1.
        if not self.escape_allowed:
            if self.player.escaping():
                done = 1
        return pygame.surfarray.array3d(self.screen), rwd, done

    def mainloop(self):
        if any(prop is None for prop in (self.player, self.enemies, self.square)):
            self.reset()
        tock = self.fps
        frame = np.zeros(self.size.tolist() + [3])
        reward = None
        print("Age: 1")
        while 1:
            self.steps_taken += 1
            if any([e.type == pygame.QUIT for e in pygame.event.get()]):
                return
            dvec = self.agent.sample_vector(frame, reward)
            info = self.step(dvec)

            if info is None:
                break
            frame, reward, done = info
            if self.steps_taken >= 4000:
                done = 1
                reward = -1
            if done:
                print()
                self.agent.update(reward)
                self.reset()
                print("Age: {} running reward: {}"
                      .format(self.agent.age, self.agent.running_reward))

            print("\rFrame: {:>5}, Reward: {:>6.2f}"
                  .format(self.steps_taken, reward),
                  end="")

            self.clock.tick(tock)
            pygame.display.flip()
        print("\n-- END PROGRAM --")
