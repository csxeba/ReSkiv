import numpy as np
import pygame


class _BallBase:

    def __init__(self, game, coords, color, size=10):
        self.game = game
        self.coords = coords  # type: np.ndarray
        self.color = color
        self.size = size
        self.draw()

    def draw(self):
        pygame.draw.circle(self.game.screen, self.color, self.coords, self.size)

    def move(self, dvec):
        self.coords += dvec.astype(int)
        self.coords[self.coords < 0] = 0
        toobig = self.coords > self.game.size
        self.coords[toobig] = self.game.size[toobig]
        self.draw()

    def touches(self, other):
        return np.linalg.norm(self.coords - other.coords) <= (self.size + other.size)

    def escaping(self):
        c = self.coords - self.size
        return np.any(c <= 0) or np.any(c >= self.game.size)


class EnemyBall(_BallBase):

    def __init__(self, game):
        initdata = np.random.uniform(size=3)
        super().__init__(game, color=(255, 100, 0), size=10,
                         coords=(initdata[:2] * game.size).astype(int))
        self.hori = initdata[2] < 0.5
        self._move_generator = self._automove()

    def _automove(self):
        d = np.array([2, 2])
        d[int(self.hori)] = 0
        while 1:
            if self.escaping():
                d *= -1
            yield d

    def move(self, dvec=None):
        super().move(next(self._move_generator))


class PlayerBall(_BallBase):

    def __init__(self, game):
        super().__init__(game, color=(0, 128, 255), size=20,
                         coords=game.size // 2)

    def move(self, dvec=None):
        if dvec is None:
            action = pygame.key.get_pressed()
            dvec = np.array([0, 0])
            if action[pygame.K_UP]:
                dvec[1] -= 3.
            if action[pygame.K_DOWN]:
                dvec[1] += 3.
            if action[pygame.K_LEFT]:
                dvec[0] -= 3.
            if action[pygame.K_RIGHT]:
                dvec[0] += 3.
        super().move(dvec)

    def dead(self):
        return any(self.touches(other) for other in self.game.enemies)


class SpazzBall(PlayerBall):

    def move(self, dvec=None):
        super().move(np.random.randn(2).astype(int) * 5)


class Square(_BallBase):

    def __init__(self, game):
        coords = (np.random.uniform() * game.size).astype(int)
        color = (50, 75, 50)
        super().__init__(game, coords, color, size=10)
        print("RECT @", coords)

    def draw(self):
        pygame.draw.rect(
            self.game.screen, self.color,
            pygame.Rect(self.coords, [self.size]*2)
        )
