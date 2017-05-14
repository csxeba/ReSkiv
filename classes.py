import numpy as np
import pygame


ENEMYSIZE = 5
PLAYERSIZE = 10
SQUARESIZE = 20  # will be divided by 2!

ENEMYCOLOR = (0, 0, 200)
PLAYERCOLOR = (50, 50, 50)
SQARECOLOR = (100, 100, 100)

PLAYERSPEED = 7
ENEMYSPEED = 5


class _EntityBase:

    def __init__(self, game, color, coords=None, size=10):
        self.game = game
        self.coords = np.array([-50, -50])
        self.color = color
        self.size = size
        if coords is not None:
            self.teleport(coords)
        self.draw()

    def draw(self):
        pygame.draw.circle(self.game.screen, self.color, self.coords, self.size)

    def adjust_coordinates(self):
        self.coords[self.coords < self.size] = self.size
        toobig = self.coords > (self.game.size - self.size)
        self.coords[toobig] = self.game.size[toobig] - self.size

    def move(self, dvec):
        self.coords += dvec.astype(int)
        self.adjust_coordinates()
        self.draw()

    def distance(self, other):
        return np.linalg.norm(self.coords - other.coords)

    def touches(self, other):
        return self.distance(other) <= (self.size + other.size)

    def escaping(self):
        r = self.size
        P = self.coords
        return np.any(P <= r) or np.any(P >= self.game.size - r)

    def teleport(self, destination=None):
        if destination is None:
            destination = (np.random.uniform(0.05, 0.95, 2) * self.game.size).astype(int)
        self.coords = destination
        self.adjust_coordinates()
        self.draw()


class EnemyBall(_EntityBase):

    def __init__(self, game):
        self._move_generator = self._automove()
        super().__init__(game, color=ENEMYCOLOR, size=ENEMYSIZE)
        self.hori = np.random.uniform(size=1)[0] < 0.5
        self.teleport()
        while self.distance(game.player) < game.meandist / 2:
            self.teleport()

    def _automove(self):
        d = np.array([ENEMYSPEED]*2)
        d[int(self.hori)] = 0
        if np.random.randn() < 0:
            d *= -1
        while 1:
            if self.escaping():
                d *= -1
            yield d

    def move(self, dvec=None):
        super().move(next(self._move_generator))


class PlayerBall(_EntityBase):

    def __init__(self, game):
        super().__init__(game, color=PLAYERCOLOR, size=PLAYERSIZE,
                         coords=game.size // 2)

    def move(self, dvec=None):
        if dvec is None:
            action = pygame.key.get_pressed()
            dvec = np.array([0, 0])
            if action[pygame.K_UP]:
                dvec[1] -= PLAYERSPEED
            if action[pygame.K_DOWN]:
                dvec[1] += PLAYERSPEED
            if action[pygame.K_LEFT]:
                dvec[0] -= PLAYERSPEED
            if action[pygame.K_RIGHT]:
                dvec[0] += PLAYERSPEED
        super().move(dvec)

    def dead(self):
        return any(self.touches(other) for other in self.game.enemies)

    def escaping(self):
        e = super().escaping()
        if e:
            print("Escaping!")
        return e


class CleverBall(PlayerBall):

    def move(self, action: np.ndarray=None):
        if action is None:
            action = np.array([0, 0])
        _EntityBase.move(self, action * PLAYERSPEED)


class SpazzBall(PlayerBall):

    def move(self, dvec=None):
        super().move(np.random.randn(2).astype(int) * 5)


class Square(_EntityBase):

    def __init__(self, game):
        super().__init__(game, color=SQARECOLOR, coords=None, size=SQUARESIZE)
        self.teleport()

    def draw(self):
        adjust = self.size // 2
        pygame.draw.rect(
            self.game.screen, self.color,
            pygame.Rect(self.coords-adjust, [self.size]*2)
        )
