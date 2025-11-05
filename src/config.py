from dataclasses import dataclass

@dataclass(frozen=True)
class EnvConfig:
    rows: int = 12
    cols: int = 11
    invader_step_every: int = 1   # mueve al invasor cada tick
    invader_zigzag_drop: int = 1  # baja 1 fila al llegar a un borde
    bomb_prob: float = 0.05       # probabilidad de soltar bomba por tick
    missile_speed: int = 1        # celdas por tick hacia arriba
    bomb_speed: int = 1           # celdas por tick hacia abajo
    max_steps: int = 400          # límite duro por episodio

@dataclass(frozen=True)
class GAConfig:
    genome_len: int = 200
    pop_size: int = 80
    generations: int = 50
    mutation_rate: float = 0.05
    tournament_k: int = 4
    crossover_rate: float = 0.9
    episodes_per_eval: int = 8

# Render
FPS_DEFAULT = 12
CELL_PIX = 40
MARGIN = 4
BG_COLOR = (12, 12, 14)
