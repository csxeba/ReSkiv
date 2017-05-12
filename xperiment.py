import pygame
import numpy as np

from classes import PlayerBall, EnemyBall


class Game:

    def __init__(self, width=400, height=300):
        pygame.init()
        self.size = np.array([width, height], dtype=int)
        self.screen = pygame.display.set_mode(self.size)
        self.clock = pygame.time.Clock()
        self.player = PlayerBall(self)
        self.enemies = [EnemyBall(self)]

    def dead(self):
        return any(self.player.touches(other) for other in self.enemies)

    def mainloop(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            self.screen.fill((0, 0, 0))
            self.player.move()
            for e in self.enemies:
                e.move()
            pygame.display.flip()
            self.clock.tick(60)
            if self.dead():
                running = False


if __name__ == '__main__':
    Game().mainloop()
