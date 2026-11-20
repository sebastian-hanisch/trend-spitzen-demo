"""Vehikel B (Verweilzeit-Mischung, ausgedacht): die gemeinsame Gewichtstabelle der Linie (`tsp_mixture.json`, unveränderte Kopie, Quelle und Stand in `tsp_mixture_quelle.json`).

Hier nur gebraucht für die Wochenstruktur-Tabelle in Experiment 5: der Pegel der ganzen Woche (aus der Tabelle) gegen die eingefrorenen Pegel nur Werktage und nur Wochenende (`tsp_grid.json`).
Überlebensfunktion einer lognormalen Mischung: S(x) = Σ w_i · ½ erfc((ln x − μ_i) / (s_i √2)); kein Zufall."""

import json
import math
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def components():
    d = json.loads((HERE / "tsp_mixture.json").read_text(encoding="utf-8"))
    return tuple(tuple(c) for c in d["komponenten"])


def survival(x: float, comps=None) -> float:
    comps = components() if comps is None else comps
    lx = math.log(x)
    return sum(w * 0.5 * math.erfc((lx - mu) / (s * math.sqrt(2.0))) for w, mu, s in comps)


def level(p: float, comps=None) -> float:
    """Wert x mit Überlebensfunktion p (Bisektion in ln x)."""
    comps = components() if comps is None else comps
    lo, hi = -3.0, 14.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if survival(math.exp(mid), comps) > p:
            lo = mid
        else:
            hi = mid
    return math.exp(0.5 * (lo + hi))
