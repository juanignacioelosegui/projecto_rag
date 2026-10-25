"""
El piso de ruido: lo que un conjunto de ~45 ítems puede sostener.

    import noise
    noise.paired(a_values, b_values)     # {"gana", "pierde", "empata", "p"}
    noise.boot_ci(values)                # IC95 de una sola fila
    noise.detectable(n)                  # cuántos discordantes hacen falta para p < 0.05

Con ~45 resultados binarios cerca del 85 %, el IC95 de **una** fila abarca unos ±10 puntos: dos
filas que difieren en 4 puntos no se distinguen leyendo sus promedios. Así que las comparaciones
se leen **pareadas** —solo los ítems que cambiaron de resultado llevan información— y el test de
signos bilateral sobre los discordantes da `p = 2 × 0.5ⁿ` cuando todos se mueven para el mismo
lado: 5-0 da 0,06 y 6-0 da 0,03. Vive acá porque las Etapas 3, 4, 5 y 6 tienen que acotar sus
filas con el mismo criterio.
"""

from __future__ import annotations

import math
import random

SEED = 0


def sign_test(wins: int, losses: int) -> float:
    """McNemar exacto (binomial bilateral) sobre los discordantes."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) * 0.5 ** n
    return min(1.0, 2 * tail)


def detectable(discordants: int = 0, alpha: float = 0.05) -> int:
    """El mínimo de discordantes TODOS para el mismo lado que baja de `alpha`. Es lo que hay que
    decir cuando un resultado da nulo: con menos que esto, la prueba no podía mostrar nada."""
    n = max(discordants, 1)
    while sign_test(n, 0) >= alpha:
        n += 1
    return n


def paired(a: list, b: list) -> dict:
    """`a` es la línea de base y `b` la fila nueva, item por item (None = no computa)."""
    w = l = t = 0
    for x, y in zip(a, b):
        if x is None or y is None:
            continue
        w += bool(y) and not bool(x)
        l += bool(x) and not bool(y)
        t += bool(x) == bool(y)
    return {"gana": w, "pierde": l, "empata": t, "p": round(sign_test(w, l), 3)}


def boot_ci(values: list[int], reps: int = 2000, seed: int = SEED) -> tuple[float, float]:
    """IC95 por bootstrap del promedio de una fila. Sirve para mostrar cuán anchas son las filas
    de una en una -- no para comparar dos (para eso está `paired`)."""
    vals = [v for v in values if v is not None]
    if not vals:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(vals)
    means = sorted(sum(rng.choices(vals, k=n)) / n for _ in range(reps))
    return means[int(0.025 * reps)], means[int(0.975 * reps)]
