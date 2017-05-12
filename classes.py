import abc

import numpy as np
import pygame


class BallBase(abc.ABC):

    def __init__(self, game, coords, color, size=10):
        self.game = game
        self.coords = coords  # type: np.ndarray
        self.color = color
        self.size = size
        self.draw()

    def draw(self):
        pygame.draw.circle(self.game.screen, self.color, self.coords, self.size)

    @abc.abstractmethod
    def move(self):
        raise NotImplementedError

    def touches(self, other):
        return np.linalg.norm(self.coords - other.coords) <= (self.size + other.size)


class EnemyBall(BallBase):

    def __init__(self, game):
        initdata = np.random.uniform(size=3)
        super().__init__(game, color=(255, 100, 0), size=10,
                         coords=(initdata[:2] * game.size).astype(int))
        self.hori = initdata[2] < 0.5
        self._move_coroutine = self._automove()

    def _automove(self):
        d = np.array([2, 2])
        d[int(self.hori)] = 0
        while 1:
            if np.any(self.coords <= 0) or np.any(self.coords >= self.game.size):
                d *= -1
            yield d

    def move(self):
        self.coords += next(self._move_coroutine)
        self.draw()


class PlayerBall(BallBase):

    def __init__(self, game):
        super().__init__(game, color=(0, 128, 255), size=20,
                         coords=game.size // 2)

    def move(self):
        dvec = np.array([0, 0])

        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_UP]:
            dvec[1] -= 3.
        if pressed[pygame.K_DOWN]:
            dvec[1] += 3.
        if pressed[pygame.K_LEFT]:
            dvec[0] -= 3.
        if pressed[pygame.K_RIGHT]:
            dvec[0] += 3.

        self.coords += dvec
        self.draw()
