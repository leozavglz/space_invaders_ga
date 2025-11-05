import random
from typing import Optional, Tuple
import numpy as np

from .constants import Cell, Action
from .entities import Defender, Invader, Missile, Bomb
from ..config import EnvConfig


class SpaceInvadersEnv:
    """
    Entorno en grid:
      - 1 defensor en la última fila (acciones: LEFT, RIGHT, SHOOT, STAY)
      - 1 invasor que baja en zig-zag (cambia de dirección y baja 1 fila en los bordes)
      - 1 misil máximo en el aire
      - Bombas aleatorias que caen del invasor
    El episodio termina al golpear al invasor, al aterrizar el invasor, al golpear al defensor,
    o al alcanzar max_steps.
    """

    def __init__(self, cfg: EnvConfig, seed: Optional[int] = None):
        self._tick = 0
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.reset()

    # ------------- ciclo de episodio -------------

    def reset(self) -> None:
        R, C = self.cfg.rows, self.cfg.cols
        self.grid = np.zeros((R, C), dtype=np.int8)

        # Defender en última fila, columna aleatoria
        self.defender = Defender(r=R - 1, c=self.rng.randrange(C))

        # Invasor en filas altas (0 o 1), col/dir aleatorias
        inv_row = self.rng.randrange(0, 2)
        self.invader = Invader(
            r=inv_row, c=self.rng.randrange(C), dir=self.rng.choice([-1, 1])
        )

        # Proyectiles
        self.missile = Missile(r=-1, c=-1, active=False)
        self.bomb = Bomb(r=-1, c=-1, active=False)

        self.steps = 0
        self.done = False
        self._place_entities()

    def reseed_episode(self) -> None:
        """Reinicia SOLO el episodio con nuevas condiciones del invasor (para evaluación repetida)."""
        self._tick = 0
        C = self.cfg.cols
        inv_row = self.rng.randrange(0, 2)
        self.invader.r = inv_row
        self.invader.c = self.rng.randrange(C)
        self.invader.dir = self.rng.choice([-1, 1])
        self.missile.active = False
        self.bomb.active = False
        self.steps = 0
        self.done = False
        self._place_entities()

    # ------------- observación -------------

    def observation(self) -> np.ndarray:
        """Devuelve una copia del grid actual (para render o agentes)."""
        return self.grid.copy()

    # ------------- utilidades internas -------------

    def _place_entities(self) -> None:
        self.grid.fill(Cell.EMPTY)
        self.grid[self.defender.r, self.defender.c] = Cell.DEFENDER
        self.grid[self.invader.r, self.invader.c] = Cell.INVADER
        if self.missile.active:
            self.grid[self.missile.r, self.missile.c] = Cell.MISSILE
        if self.bomb.active:
            self.grid[self.bomb.r, self.bomb.c] = Cell.BOMB

    def _move_invader(self) -> None:
        C = self.cfg.cols
        nxt_c = self.invader.c + self.invader.dir
        if nxt_c < 0 or nxt_c >= C:
            # en borde: baja una fila y voltea dirección
            self.invader.r += self.cfg.invader_zigzag_drop
            self.invader.dir *= -1
        else:
            self.invader.c = nxt_c

    def _maybe_drop_bomb(self) -> None:
        if self.bomb.active:
            return
        if self.rng.random() < self.cfg.bomb_prob:
            # La bomba inicia justo debajo del invasor
            start_r = self.invader.r + 1
            if start_r < self.cfg.rows:
                self.bomb = Bomb(r=start_r, c=self.invader.c, active=True)

    def _move_projectiles(self) -> None:
        # Misil sube
        if self.missile.active:
            self.missile.r -= self.cfg.missile_speed
            if self.missile.r < 0:
                self.missile.active = False

        # Bomba baja
        if self.bomb.active:
            self.bomb.r += self.cfg.bomb_speed
            if self.bomb.r >= self.cfg.rows:
                self.bomb.active = False

    def _collisions(self) -> Tuple[bool, bool, bool]:
        """Devuelve: (hit_invader, hit_defender, landed)"""
        hit_invader = False
        hit_defender = False
        landed = False

        # Misil vs Invasor
        if self.missile.active and (
            self.missile.r == self.invader.r and self.missile.c == self.invader.c
        ):
            hit_invader = True
            self.missile.active = False

        # Bomba vs Defensor
        if self.bomb.active and (
            self.bomb.r == self.defender.r and self.bomb.c == self.defender.c
        ):
            hit_defender = True
            self.bomb.active = False

        # Invasor aterriza (llega a la última fila)
        if self.invader.r >= self.cfg.rows - 1:
            landed = True

        return hit_invader, hit_defender, landed

    # ------------- paso de simulación -------------

    def step(self, action: Action):
        """
        Avanza 1 tick aplicando 'action'.
        Retorna: (done, row_aligned, horiz_dist, outcome)
          - done: bool
          - row_aligned: 1 si misil e invasor están en la MISMA fila en este tick; 0 si no
          - horiz_dist: distancia horizontal |c_misil - c_invasor| si row_aligned==1; 0 si no
          - outcome: 1=hit_invader, -1=hit_defender, -2=invader_landed, 0=ninguno
        """
        if self.done:
            return True, 0, 0, 0

        self.steps += 1

        # Acción del defensor
        if action == Action.LEFT:
            self.defender.c = max(0, self.defender.c - 1)
        elif action == Action.RIGHT:
            self.defender.c = min(self.cfg.cols - 1, self.defender.c + 1)
        elif action == Action.SHOOT:
            if not self.missile.active:
                self.missile = Missile(r=self.defender.r - 1, c=self.defender.c, active=True)
        # STAY no hace nada

        # Avanza el mundo
        self._tick += 1
        if (self._tick % max(1, self.cfg.invader_step_every)) == 0:
            self._move_invader()
        self._maybe_drop_bomb()
        self._move_projectiles()

        # Colisiones y actualización de grid
        hit_invader, hit_defender, landed = self._collisions()
        self._place_entities()

        done = (
            hit_invader
            or hit_defender
            or landed
            or (self.steps >= self.cfg.max_steps)
        )
        self.done = done

        # Métrica para fitness: distancia horizontal cuando el misil y el invasor
        # están en la misma fila (row-aligned).
        row_aligned = 1 if (self.missile.active and self.missile.r == self.invader.r) else 0
        horiz_dist = abs(self.missile.c - self.invader.c) if row_aligned else 0

        outcome = 1 if hit_invader else (-1 if hit_defender else (-2 if landed else 0))
        return done, row_aligned, horiz_dist, outcome
