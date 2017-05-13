import numpy as np
import pygame


EPSILON = 1


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
        self.coords[self.coords < self.size] = self.size + EPSILON
        toobig = self.coords > (self.game.size - self.size)
        self.coords[toobig] = self.game.size[toobig] - self.size - EPSILON

    def move(self, dvec):
        self.coords += dvec.astype(int)
        self.adjust_coordinates()
        self.draw()

    def distance(self, other):
        return np.linalg.norm(self.coords - other.coords)

    def touches(self, other):
        return self.distance(other) <= (self.size + other.size)

    def escaping(self):
        r = self.size + EPSILON
        P = self.coords
        return np.any(P <= r) or np.any(P >= self.game.size - r)

    def teleport(self, destination=None):
        if destination is None:
            destination = (np.random.uniform(size=2) * self.game.size * 0.95).astype(int)
        self.coords = destination
        self.adjust_coordinates()
        self.draw()


class EnemyBall(_EntityBase):

    def __init__(self, game):
        self._move_generator = self._automove()
        super().__init__(game, color=(255, 100, 0), size=5)
        self.hori = np.random.uniform(size=1)[0] < 0.5
        self.teleport()
        while self.distance(game.player) < game.meandist / 2:
            print("Adjusting enemy position to", self.coords)
            self.teleport()

    def _automove(self):
        speed = 5
        d = np.array([speed, speed])
        d[int(self.hori)] = 0
        while 1:
            if self.escaping():
                d *= -1
            yield d

    def move(self, dvec=None):
        super().move(next(self._move_generator))


class PlayerBall(_EntityBase):

    def __init__(self, game):
        super().__init__(game, color=(0, 128, 255), size=10,
                         coords=game.size // 2)

    def move(self, dvec=None):
        speed = 7
        if dvec is None:
            action = pygame.key.get_pressed()
            dvec = np.array([0, 0])
            if action[pygame.K_UP]:
                dvec[1] -= speed
            if action[pygame.K_DOWN]:
                dvec[1] += speed
            if action[pygame.K_LEFT]:
                dvec[0] -= speed
            if action[pygame.K_RIGHT]:
                dvec[0] += speed
        super().move(dvec)

    def dead(self):
        return any(self.touches(other) for other in self.game.enemies)


class CleverBall(PlayerBall):

    def move(self, dvec=None):
        if dvec is None:
            dvec = np.array([0, 0])
        _EntityBase.move(self, dvec)


class SpazzBall(PlayerBall):

    def move(self, dvec=None):
        super().move(np.random.randn(2).astype(int) * 5)


class Square(_EntityBase):

    def __init__(self, game):
        color = (50, 75, 50)
        super().__init__(game, color=color, coords=None, size=20)
        self.teleport()

    def draw(self):
        adjust = self.size // 2
        pygame.draw.rect(
            self.game.screen, self.color,
            pygame.Rect(self.coords-adjust, [self.size]*2)
        )
