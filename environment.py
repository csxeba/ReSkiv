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

    def reset(self):
        self.player = PlayerBall(self)
        self.square = Square(self)
        self.enemies = [EnemyBall(self)]
        self.points = 0
        return self.step()[0]

    def point(self):
        return self.player.touches(self.square)

    def step(self, action=None):
        if any([e.type == pygame.QUIT for e in pygame.event.get()]):
            return pygame.surfarray.array3d(self.screen), self.points, 1
        self.screen.fill((0, 0, 0))

        self.square.draw()
        self.player.move(action)
        for e in self.enemies:
            e.move()
        if self.point():
            self.square = Square(self)
            self.enemies.append(EnemyBall(self))
            self.points += 1.
        if self.player.dead():
            return pygame.surfarray.array3d(self.screen), -10., 1
        if not self.escape_allowed:
            if self.player.esaping():
                return pygame.surfarray.array3d(self.screen), -10, 1
        return pygame.surfarray.array3d(self.screen), self.points, 0

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
