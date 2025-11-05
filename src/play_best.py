import argparse
import os
import numpy as np
import pygame

from .config import EnvConfig, FPS_DEFAULT
from .game.environment import SpaceInvadersEnv
from .game.constants import ACTIONS
from .game.renderer import Renderer

INT2ACT = {i: a for i, a in enumerate(ACTIONS)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome", type=str, default="best_genome.npy")
    ap.add_argument("--fps", type=int, default=FPS_DEFAULT)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    print(f"[play_best] CWD = {os.getcwd()}")
    print(f"[play_best] Intentando cargar genoma: {args.genome}")

    # Carga del genoma
    if not os.path.exists(args.genome):
        print(f"[play_best] ERROR: no existe el archivo {args.genome}")
        return
    genome = np.load(args.genome)
    if genome.ndim != 1 or genome.size == 0:
        print(f"[play_best] ERROR: genoma con forma inválida: shape={genome.shape}")
        return
    print(f"[play_best] Genoma cargado. length={len(genome)} dtype={genome.dtype}")

    # Entorno + renderer
    env = SpaceInvadersEnv(EnvConfig(), seed=args.seed)
    renderer = Renderer(rows=env.cfg.rows, cols=env.cfg.cols, title="Best Genome Player")
    env.reseed_episode()

    print("[play_best] Ventana abierta. Reproduciendo… (cierra la ventana para salir)")
    t = 0
    quit_req = False
    while not quit_req:
        # Procesar eventos de cierre lo primero para macOS
        quit_req = renderer.handle_quit()

        # Acción del genoma
        action = INT2ACT[int(genome[t % len(genome)])]
        done, *_ = env.step(action)
        if done:
            env.reseed_episode()

        # Dibujar
        grid = env.observation()
        renderer.draw(grid, fps=args.fps)

        t += 1

    pygame.quit()
    print("[play_best] Salida limpia.")

if __name__ == "__main__":
    main()
