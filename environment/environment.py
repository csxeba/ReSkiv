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
        self.actions = [(-1, -1), (-1, 0), (-1, 1),
                        (0, -1), (0, 0), (0, 1),
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
        done = 0
        rwd = 0.
        self.screen.fill((0, 0, 0))
        self.square.draw()
        self.player.move(dvec)
        for e in self.enemies:
            e.move()
        if self.score():
            self.square = Square(self)
            self.enemies.append(EnemyBall(self))
            self.points += 1.
            rwd += 3.
        if self.player.dead():
            done = 1
            self.points -= 1
            rwd -= 1
        if not self.escape_allowed:
            if self.player.escaping():
                done = 1
        return pygame.surfarray.array3d(self.screen), rwd, done

    def mainloop(self):
        tock = self.fps
        frame = self.reset()
        reward = None
        running_reward = None
        steps_taken = 0
        print("Age: 1")
        while 1:
            steps_taken += 1
            if any([e.type == pygame.QUIT for e in pygame.event.get()]):
                return
            dvec = self.agent.sample_vector(frame, reward)
            info = self.step(dvec)

            if info is None:
                break

            frame, reward, done = info

            if running_reward is None:
                running_reward = reward
            if reward:
                running_reward += running_reward * 0.99 + reward * 0.01
                self.agent.update(reward)
                steps_taken = 0
            if steps_taken >= 3000:
                self.agent.update(-1.)
                steps_taken = 0
            if done:
                self.reset()
                print("Age:", self.agent.age)
                steps_taken = 0

            print("\rStep count: {:>5}, Points: {:.0f}"
                  .format(steps_taken, self.points),
                  end="")

            self.clock.tick(tock)
            pygame.display.flip()
        print("\n-- END PROGRAM --")
