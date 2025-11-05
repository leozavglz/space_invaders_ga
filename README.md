# Proyecto: Invasores del Espacio (Algoritmos Genéticos en Python)

Este proyecto implementa una versión simplificada del clásico *Space Invaders*, en la que un defensor aprende a jugar utilizando un **algoritmo genético (Genetic Algorithm, GA)**.  
El desarrollo se realizó en Python empleando **Pygame** para el entorno visual y **NumPy** para los cálculos evolutivos.

La parte correspondiente al modelo genético se encuentra en la rama **`Genetica-Leo`**.

---

## 1. Descripción general

El juego se desarrolla en un tablero de 12x11 celdas.  
- Un **defensor** se mueve en la parte inferior (acciones: izquierda, derecha, disparar o quedarse quieto).  
- Un **invasor** desciende en zigzag desde la parte superior, dejando caer bombas de manera aleatoria.  
- Solo puede existir **un misil en el aire** a la vez.

El objetivo del defensor es **destruir al invasor** antes de que este aterrice o lo impacte con una bomba.

El algoritmo genético busca encontrar una **secuencia óptima de acciones** (genoma) que maximice la eficacia del defensor.

---

## 2. Representación genética

Cada individuo de la población representa una posible estrategia del defensor.  
El genoma se codifica como un **vector de enteros** donde cada número corresponde a una acción:

| Acción | Código |
|---------|---------|
| STAY | 0 |
| LEFT | 1 |
| RIGHT | 2 |
| SHOOT | 3 |

Cada genoma tiene longitud 200, lo que equivale a 200 pasos de simulación por partida.

---

## 3. Función de evaluación (Fitness)

El fitness de cada individuo se calcula con base en el desempeño del defensor durante varios episodios (8 por defecto):

- Se suman las **distancias horizontales** entre el misil y el invasor cuando están en la misma fila.  
- Se aplican **bonificaciones o penalizaciones**:
  - Golpear al invasor: **–50**
  - Dejar que aterrice: **+50**
- El valor final acumulado constituye el **fitness total**.  
  Un valor **menor** indica un mejor desempeño.

---

## 4. Parámetros del algoritmo genético

| Parámetro | Valor |
|------------|--------|
| Generaciones | 50 |
| Población | 80 individuos |
| Longitud del genoma | 200 |
| Tasa de mutación | 0.05 |
| Tasa de crossover | 0.9 |
| Selección | Torneo (k=4) |
| Episodios por evaluación | 8 |
| Elitismo | Mantiene el mejor individuo |

La configuración del entorno y del GA se almacena automáticamente en el archivo `metrics.json`.

---

## 5. Estructura del proyecto

```
space_invaders_ga/
├─ requirements.txt
├─ README.md
└─ src/
   ├─ config.py
   ├─ ai/
   │  └─ genetic.py
   ├─ game/
   │  ├─ constants.py
   │  ├─ entities.py
   │  ├─ environment.py
   │  └─ renderer.py
   ├─ play_manual.py
   ├─ play_random.py
   ├─ play.py
   ├─ play_best.py
   ├─ train_ga.py
   └─ plot_metrics.py
```

---

## 6. Instalación y ejecución

### a) Entorno virtual
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### b) Requisitos principales
```
pygame>=2.6
numpy>=1.26
matplotlib>=3.8
```

### c) Entrenar el agente genético
```bash
python -m src.train_ga --generations 50 --pop-size 80 --episodes 8 --seed 42
```

Este comando genera los archivos:
- `metrics.json` → métricas completas por generación  
- `best_genome.npy` → mejor individuo (formato binario)  
- `best_genome.json` → versión legible del genoma

---

## 7. Reproducir el mejor agente

El mejor individuo puede reproducirse visualmente en Pygame mediante el script `play_best.py`:

```bash
python -m src.play_best --genome best_genome.npy --fps 8 --seed 123
```

Si el archivo `best_genome.npy` no existe, puede crearse a partir del JSON:

```bash
python - << 'PY'
import json, numpy as np
g = json.load(open("best_genome.json","r",encoding="utf-8"))["genome"]
np.save("best_genome.npy", np.array(g, dtype="int8"))
print("Archivo best_genome.npy generado.")
PY
```

---

## 8. Resultados obtenidos

El entrenamiento registró un mejor desempeño en la **generación 39**, con un **fitness global de –351**.  
Esto indica que el agente logró múltiples impactos exitosos contra el invasor, mostrando aprendizaje a través de la evolución de estrategias.

Las gráficas generadas (`metrics_plot.png` y `metrics_plot_std.png`) muestran la evolución del fitness y la variabilidad de la población a lo largo de las generaciones.

---

## 9. Ajustes de velocidad y entorno

Para controlar la velocidad de movimiento del invasor se añadió el parámetro `invader_step_every` en `EnvConfig` (archivo `config.py`).  
Aumentar este valor (por ejemplo, 2 o 3) reduce la frecuencia de movimiento y produce una experiencia visual más lenta y estable.

Ejemplo:
```python
@dataclass(frozen=True)
class EnvConfig:
    rows: int = 12
    cols: int = 11
    invader_step_every: int = 2  # mueve el invasor cada 2 ticks
    invader_zigzag_drop: int = 1
    bomb_prob: float = 0.05
    missile_speed: int = 1
    bomb_speed: int = 1
    max_steps: int = 400
```

---


