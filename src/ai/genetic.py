from dataclasses import dataclass
from typing import List, Tuple, Callable
import numpy as np

# Acciones: 0=STAY, 1=LEFT, 2=RIGHT, 3=SHOOT
ACTION_SPACE_SIZE = 4


@dataclass
class Individual:
    genome: np.ndarray  # vector de ints (acciones)
    fitness: float = np.inf


class GeneticAlgorithm:
    """
    Implementación simple de GA con:
    - selección por torneo
    - crossover 1 punto
    - mutación puntual
    - elitismo (conserva al mejor)
    """
    def __init__(
        self,
        genome_len: int,
        pop_size: int,
        mutation_rate: float,
        tournament_k: int,
        crossover_rate: float,
        rng: np.random.Generator,
    ):
        self.genome_len = genome_len
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.tournament_k = tournament_k
        self.crossover_rate = crossover_rate
        self.rng = rng

    # ---------- población ----------
    def init_population(self) -> List[Individual]:
        pop = []
        # distribución uniforme inicial (se puede sesgar si quieres)
        for _ in range(self.pop_size):
            genome = self.rng.integers(0, ACTION_SPACE_SIZE, size=self.genome_len, dtype=np.int8)
            pop.append(Individual(genome=genome))
        return pop

    def evaluate(self, pop: List[Individual], eval_fn: Callable[[np.ndarray], float]) -> None:
        for ind in pop:
            ind.fitness = float(eval_fn(ind.genome))

    # ---------- selección ----------
    def _tournament(self, pop: List[Individual]) -> Individual:
        k = min(self.tournament_k, len(pop))
        idxs = self.rng.choice(len(pop), size=k, replace=False)
        best = pop[idxs[0]]
        for i in idxs[1:]:
            if pop[i].fitness < best.fitness:
                best = pop[i]
        return best

    # ---------- crossover ----------
    def _crossover(self, a: Individual, b: Individual) -> Tuple[Individual, Individual]:
        if self.rng.random() > self.crossover_rate:
            return Individual(a.genome.copy()), Individual(b.genome.copy())
        point = int(self.rng.integers(1, self.genome_len))  # [1, L-1]
        child1 = np.concatenate([a.genome[:point], b.genome[point:]])
        child2 = np.concatenate([b.genome[:point], a.genome[point:]])
        return Individual(child1), Individual(child2)

    # ---------- mutación ----------
    def _mutate(self, ind: Individual) -> None:
        mask = self.rng.random(self.genome_len) < self.mutation_rate
        if mask.any():
            ind.genome[mask] = self.rng.integers(0, ACTION_SPACE_SIZE, size=mask.sum(), dtype=np.int8)

    # ---------- nueva generación ----------
    def next_generation(self, pop: List[Individual]) -> List[Individual]:
        # elitismo
        elite = min(pop, key=lambda x: x.fitness)
        new_pop: List[Individual] = [Individual(elite.genome.copy(), elite.fitness)]

        while len(new_pop) < self.pop_size:
            p1 = self._tournament(pop)
            p2 = self._tournament(pop)
            c1, c2 = self._crossover(p1, p2)
            self._mutate(c1)
            new_pop.append(c1)
            if len(new_pop) < self.pop_size:
                self._mutate(c2)
                new_pop.append(c2)

        return new_pop
