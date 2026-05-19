import numpy as np
import pygame
from config import (FOOD_RADIUS, FOOD_COLOR, FOOD_COUNT_MAX,
                    FOOD_RESPAWN_INTERVAL, FOOD_RESPAWN_BATCH,
                    WORLD_WIDTH, WORLD_HEIGHT)


class Food:
    def __init__(self, x=None, y=None):
        margin = FOOD_RADIUS + 10
        self.x = x if x is not None else np.random.uniform(margin, WORLD_WIDTH - margin)
        self.y = y if y is not None else np.random.uniform(margin, WORLD_HEIGHT - margin)
        self.radius = FOOD_RADIUS
        self.color = FOOD_COLOR
        self.alive = True

    def draw(self, surface):
        pygame.draw.circle(surface, self.color,
                           (int(self.x), int(self.y)), self.radius)
        # Small bright center dot
        pygame.draw.circle(surface, (180, 255, 180),
                           (int(self.x), int(self.y)), max(1, self.radius - 2))


class FoodManager:
    def __init__(self):
        self.items = []
        self._respawn_timer = 0
        self._spawn_initial()

    def _spawn_initial(self):
        for _ in range(FOOD_COUNT_MAX):
            self.items.append(Food())

    def update(self):
        # Remove eaten food
        self.items = [f for f in self.items if f.alive]

        # Periodic respawn
        self._respawn_timer += 1
        if self._respawn_timer >= FOOD_RESPAWN_INTERVAL:
            self._respawn_timer = 0
            deficit = FOOD_COUNT_MAX - len(self.items)
            batch = min(FOOD_RESPAWN_BATCH, deficit)
            for _ in range(batch):
                self.items.append(Food())

    def draw(self, surface):
        for food in self.items:
            food.draw(surface)

    def count(self):
        return len(self.items)
