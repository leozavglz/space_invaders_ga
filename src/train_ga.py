import argparse
import json
from datetime import datetime
from statistics import mean, median, pstdev
import numpy as np

from .config import EnvConfig, GAConfig
from .game.environment import SpaceInvadersEnv
from .game.constants import ACTIONS
from .ai.genetic import GeneticAlgorithm

# Mapa seguro int->Action
INT2ACT = {i: a for i, a in enumerate(ACTIONS)}


def evaluate_genome(genome: np.ndarray, episodes: int, seed: int, env_cfg: EnvConfig) -> float:
    """
    Fitness = suma de distancias horizontales cuando misil e invasor están en la MISMA fila,
              + penalizaciones/bonos por aterrizaje o impacto.
    Menor = mejor.
    """
    rng = np.random.default_rng(seed)
    env = SpaceInvadersEnv(env_cfg, seed=int(rng.integers(1_000_000_000)))
    total = 0.0

    for _ in range(episodes):
        env.reseed_episode()
        acc_dist = 0.0
        bonus = 0.0

        for gene in genome:
            action = INT2ACT[int(gene)]
            done, row_aligned, horiz_dist, outcome = env.step(action)

            if row_aligned:
                acc_dist += horiz_dist

            # bonificación/penalización suave para acelerar convergencia
            if outcome == 1:      # hit invader
                bonus -= 50.0
            elif outcome == -2:   # invader landed
                bonus += 50.0

            if done:
                break

        total += (acc_dist + bonus)

    return float(total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=GAConfig.generations)
    ap.add_argument("--pop-size", type=int, default=GAConfig.pop_size)
    ap.add_argument("--episodes", type=int, default=GAConfig.episodes_per_eval)
    ap.add_argument("--genome-len", type=int, default=GAConfig.genome_len)
    ap.add_argument("--mutation-rate", type=float, default=GAConfig.mutation_rate)
    ap.add_argument("--crossover-rate", type=float, default=GAConfig.crossover_rate)
    ap.add_argument("--tournament-k", type=int, default=GAConfig.tournament_k)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--metrics-out", type=str, default="metrics.json")
    ap.add_argument("--best-npy", type=str, default="best_genome.npy")
    ap.add_argument("--best-json", type=str, default="best_genome.json")
    ap.add_argument("--topk", type=int, default=3, help="Cuántos mejores fitness guardar por generación")
    args = ap.parse_args()

    # Configs
    env_cfg = EnvConfig()
    rng = np.random.default_rng(args.seed)
    ga = GeneticAlgorithm(
        genome_len=args.genome_len,
        pop_size=args.pop_size,
        mutation_rate=args.mutation_rate,
        tournament_k=args.tournament_k,
        crossover_rate=args.crossover_rate,
        rng=rng,
    )

    def eval_fn(genome: np.ndarray) -> float:
        return evaluate_genome(
            genome,
            episodes=args.episodes,
            seed=int(rng.integers(1_000_000_000)),
            env_cfg=env_cfg,
        )

    # Inicialización y primera evaluación
    pop = ga.init_population()
    ga.evaluate(pop, eval_fn)

    # Métricas de corrida
    run_metrics = {
        "run_started_at": datetime.now().isoformat(timespec="seconds"),
        "env": {
            "rows": env_cfg.rows,
            "cols": env_cfg.cols,
            "bomb_prob": env_cfg.bomb_prob,
            "missile_speed": env_cfg.missile_speed,
            "bomb_speed": env_cfg.bomb_speed,
            "max_steps": env_cfg.max_steps,
        },
        "ga": {
            "generations": args.generations,
            "pop_size": args.pop_size,
            "genome_len": args.genome_len,
            "mutation_rate": args.mutation_rate,
            "crossover_rate": args.crossover_rate,
            "tournament_k": args.tournament_k,
            "episodes_per_eval": args.episodes,
            "seed": args.seed,
        },
        "history": []  # métricas por generación
    }

    # Tracking de mejor global
    best_now = min(pop, key=lambda x: x.fitness)
    global_best = {
        "fitness": float(best_now.fitness),
        "genome": best_now.genome.copy(),
        "generation": 0,
    }

    def snapshot_metrics(gen_idx: int):
        fits = [float(ind.fitness) for ind in pop]
        best_ind = min(pop, key=lambda x: x.fitness)

        # top-k fitness (ordenados ascendente: mejor primero)
        k = max(1, min(args.topk, len(fits)))
        topk = sorted(fits)[:k]

        data = {
            "generation": gen_idx,
            "best_fitness": float(best_ind.fitness),
            "mean_fitness": float(mean(fits)),
            "median_fitness": float(median(fits)),
            "std_fitness": float(pstdev(fits)) if len(fits) > 1 else 0.0,
            "topk_fitness": topk,
        }
        run_metrics["history"].append(data)

        print(
            f"Gen {gen_idx:3d} | "
            f"best={data['best_fitness']:.2f}  "
            f"mean={data['mean_fitness']:.2f}  "
            f"median={data['median_fitness']:.2f}  "
            f"std={data['std_fitness']:.2f}"
        )

        return best_ind

    # Gen 0
    best_ind = snapshot_metrics(0)

    # Actualiza global best si aplica
    if best_ind.fitness < global_best["fitness"]:
        global_best = {
            "fitness": float(best_ind.fitness),
            "genome": best_ind.genome.copy(),
            "generation": 0,
        }

    # Loop evolutivo
    for g in range(1, args.generations + 1):
        pop = ga.next_generation(pop)
        ga.evaluate(pop, eval_fn)
        best_ind = snapshot_metrics(g)

        # Actualiza global best si aplica
        if best_ind.fitness < global_best["fitness"]:
            global_best = {
                "fitness": float(best_ind.fitness),
                "genome": best_ind.genome.copy(),
                "generation": g,
            }

    # Guardados (mejor global)
    np.save(args.best_npy, global_best["genome"])
    with open(args.best_json, "w", encoding="utf-8") as f:
        json.dump({"genome": global_best["genome"].tolist()}, f, ensure_ascii=False, indent=2)

    # Finaliza métricas
    run_metrics["run_finished_at"] = datetime.now().isoformat(timespec="seconds")
    run_metrics["global_best"] = {
        "fitness": float(global_best["fitness"]),
        "generation": int(global_best["generation"]),
        "saved_as": {"npy": args.best_npy, "json": args.best_json},
    }
    # También guarda el mejor de la última generación para comparar
    run_metrics["last_gen_best"] = {
        "fitness": float(best_ind.fitness),
        "generation": int(args.generations),
    }

    with open(args.metrics_out, "w", encoding="utf-8") as f:
        json.dump(run_metrics, f, ensure_ascii=False, indent=2)

    print(
        f"\nSaved GLOBAL best genome (gen {global_best['generation']}) "
        f"to {args.best_npy} and {args.best_json}"
    )
    print(f"Metrics written to {args.metrics_out}")


if __name__ == "__main__":
    main()
