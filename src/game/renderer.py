import pygame
from .constants import Cell
from ..config import CELL_PIX, MARGIN, BG_COLOR

# Colores para cada tipo de celda
COLORS = {
    Cell.EMPTY: (40, 40, 50),
    Cell.DEFENDER: (0, 200, 120),
    Cell.INVADER: (200, 60, 60),
    Cell.MISSILE: (220, 220, 40),
    Cell.BOMB: (180, 180, 230),
}


class Renderer:
    """Dibuja el entorno en una ventana de Pygame."""

    def __init__(self, rows: int, cols: int, title: str = "Space Invaders GA"):
        pygame.init()
        w = cols * CELL_PIX + (cols + 1) * MARGIN
        h = rows * CELL_PIX + (rows + 1) * MARGIN
        self.screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()

    def draw(self, grid, fps: int):
        """Dibuja el grid completo y controla los FPS."""
        self.screen.fill(BG_COLOR)
        rows, cols = grid.shape

        for r in range(rows):
            for c in range(cols):
                color = COLORS.get(grid[r, c], (90, 90, 100))
                x = c * CELL_PIX + (c + 1) * MARGIN
                y = r * CELL_PIX + (r + 1) * MARGIN
                pygame.draw.rect(
                    self.screen,
                    color,
                    (x, y, CELL_PIX, CELL_PIX),
                    border_radius=6
                )

        pygame.display.flip()
        self.clock.tick(fps)

    def handle_quit(self) -> bool:
        """Detecta si el usuario cierra la ventana."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
        return False
