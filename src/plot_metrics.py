# src/plot_metrics.py
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_metrics(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_series(history):
    gens = [int(h["generation"]) for h in history]
    best = [float(h["best_fitness"]) for h in history]
    mean = [float(h["mean_fitness"]) for h in history]
    median = [float(h["median_fitness"]) for h in history]
    std = [float(h.get("std_fitness", 0.0)) for h in history]
    return gens, best, mean, median, std


def plot_curves(gens, best, mean, median, title, out_path: Path):
    plt.figure(figsize=(10, 5))
    # No definir colores explícitos (usar los default)
    plt.plot(gens, best, label="Best (↓ mejor)", linewidth=2)
    plt.plot(gens, mean, label="Mean", linewidth=1.5)
    plt.plot(gens, median, label="Median", linewidth=1.5)

    plt.title(title)
    plt.xlabel("Generación")
    plt.ylabel("Fitness (más bajo es mejor)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_std(gens, std, title, out_path: Path):
    plt.figure(figsize=(10, 4))
    plt.plot(gens, std, label="Std fitness", linewidth=1.8)
    plt.title(title)
    plt.xlabel("Generación")
    plt.ylabel("Desviación estándar")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Graficar métricas del GA desde metrics.json")
    ap.add_argument("--metrics", type=str, default="metrics.json", help="Ruta a metrics.json")
    ap.add_argument("--out-prefix", type=str, default="metrics_plot", help="Prefijo para archivos de salida PNG")
    args = ap.parse_args()

    metrics_path = Path(args.metrics)
    out_prefix = Path(args.out_prefix)

    data = load_metrics(metrics_path)
    history = data.get("history", [])
    if not history:
        raise SystemExit("No hay 'history' en el JSON de métricas.")

    gens, best, mean, median, std = extract_series(history)

    # Título enriquecido con info de GA si existe
    ga = data.get("ga", {})
    title_main = (
        f"GA Fitness — gens={ga.get('generations','?')}  "
        f"pop={ga.get('pop_size','?')}  "
        f"episodes={ga.get('episodes_per_eval','?')}  "
        f"genome_len={ga.get('genome_len','?')}"
    )
    title_std = "GA Varianza — Desviación estándar por generación"

    # Salidas
    out_main = out_prefix.with_suffix(".png")
    out_std = out_prefix.with_name(out_prefix.name + "_std").with_suffix(".png")

    # Graficar
    plot_curves(gens, best, mean, median, title_main, out_main)
    plot_std(gens, std, title_std, out_std)

    # Imprime un breve resumen en consola
    global_best = data.get("global_best", {})
    last_best = data.get("last_gen_best", {})
    print(f"Gráficas guardadas:")
    print(f" - {out_main}")
    print(f" - {out_std}")
    if global_best:
        print(
            f"Global best: fitness={global_best.get('fitness')} "
            f"en gen={global_best.get('generation')}"
        )
    if last_best:
        print(
            f"Última gen best: fitness={last_best.get('fitness')} "
            f"en gen={last_best.get('generation')}"
        )


if __name__ == "__main__":
    main()
