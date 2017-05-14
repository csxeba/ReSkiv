from classes import *
from util import calc_meand


class Game:

    def __init__(self, ball_type, fps, screensize, escape=True):
        pygame.init()
        self.size = np.array(screensize, dtype=int)
        self.meandist = calc_meand(self.size)
        self.screen = pygame.display.set_mode(self.size)
        self.clock = pygame.time.Clock()
        self.playertype = {
            "manual": PlayerBall,
            "clever": CleverBall,
            "spazz": SpazzBall
        }[ball_type]
        self.escape_allowed = escape
        self.fps = fps
        self.player = None
        self.square = None
        self.enemies = None
        self.points = 0.
        self.steps_taken = 0
        self.actions = [(-1, -1), (-1, 0), (-1, 1),
                        (0, -1), (0, 0), (0, 1),
                        (1, -1), (1, 0), (1, 1)]
        self.fake_labels = np.eye(len(self.actions))

    def sample_action(self, probs):
        arg_action = np.random.choice(np.arange(len(self.actions)), size=1, p=probs)
        action = np.array(self.actions[arg_action[0]])
        label = self.fake_labels[arg_action]
        return action, label

    def reset(self):
        self.player = self.playertype(self)
        self.square = Square(self)
        self.enemies = [EnemyBall(self)]
        self.points = 0
        self.steps_taken = 0
        return self.step()[0]

    def score(self):
        return self.player.touches(self.square)

    def step(self, action=None):
        reward = 0
        done = 0
        self.steps_taken += 1
        if any([e.type == pygame.QUIT for e in pygame.event.get()]):
            return
        self.screen.fill((0, 0, 0))

        self.square.draw()
        self.player.move(action)
        for e in self.enemies:
            e.move()
        if self.score():
            self.square = Square(self)
            self.enemies.append(EnemyBall(self))
            self.steps_taken = 0
            self.points += 5
            reward += 10
        if self.player.dead():
            reward -= 10
            done = 1
        if not self.escape_allowed:
            if self.player.escaping():
                reward -= 10
                done = 1
        return pygame.surfarray.array3d(self.screen), reward, done

    def mainloop(self):
        if any(prop is None for prop in (self.player, self.enemies, self.square)):
            self.reset()
        tock = self.fps
        while 1:
            frame, reward, done = self.step()
            if done:
                print("Done! Exiting!")
                break
            self.clock.tick(tock)
            pygame.display.flip()
