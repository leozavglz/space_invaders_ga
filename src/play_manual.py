import argparse
import pygame
from .config import EnvConfig, FPS_DEFAULT
from .game.environment import SpaceInvadersEnv
from .game.constants import Action
from .game.renderer import Renderer

def key_to_action() -> Action:
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        return Action.LEFT
    if keys[pygame.K_RIGHT]:
        return Action.RIGHT
    if keys[pygame.K_SPACE]:
        return Action.SHOOT
    return Action.STAY

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=FPS_DEFAULT)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    env = SpaceInvadersEnv(EnvConfig(), seed=args.seed)
    renderer = Renderer(rows=env.cfg.rows, cols=env.cfg.cols, title="Manual Play (Space Invaders GA)")

    quit_req = False
    while not quit_req:
        quit_req = renderer.handle_quit()

        # elegir acción por teclado
        action = key_to_action()

        # step
        done, _, _, _ = env.step(action)
        if done:
            env.reseed_episode()

        # dibujar
        grid = env.observation()
        renderer.draw(grid, fps=args.fps)

    pygame.quit()

if __name__ == "__main__":
    main()
