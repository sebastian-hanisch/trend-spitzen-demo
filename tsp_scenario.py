"""Reihen für die Experimente: Jahresmaxima Gumbel mit Lage-Trend, gezogen mit der SplitMix64-Vorlage (`tsp_rng.py`, per Konstruktion plattformunabhängig: Ganzzahlarithmetik; auf Linux nicht geprüft).

Ein Gumbel-Wert kostet zwei `exact_log`-Aufrufe (grob 0,1 ms auf dieser Maschine). Ziehung: E = −ln(1−u) ist exponentialverteilt, X = −ln E ist Standard-Gumbel
(P(X ≤ x) = exp(−e^{−x}) wegen P(E > s) = e^{−s}); E = 0 (u = 0, Wahrscheinlichkeit 2^-53) wird neu gezogen."""

import numpy as np

import tsp_rng as R

MASK = R.MASK
GOLDEN = 0x9E3779B97F4A7C15


def derive_seed(base: int, *tags: int) -> int:
    """Abgeleiteter Zufalls-Seed: mischt `base` und die Marken `tags` über SplitMix64 (gleiche Eingabe, gleicher Seed; verschiedene Marken, unabhängige Ströme)."""
    s = base & MASK
    for t in tags:
        s = R.SplitMix64((s ^ ((int(t) + 1) * GOLDEN)) & MASK).next_u64()
    return s


def gumbel(rng: R.SplitMix64) -> float:
    """Ein Standard-Gumbel-Wert (Lage 0, Skala 1)."""
    while True:
        e = rng.expovariate(1.0)
        if e > 0.0:
            return -R.exact_log(e)


def gumbel_series(m: int, beta: float, seed: int) -> np.ndarray:
    """Jahresmaxima x_t = β·t + G_t für t = 0 … m−1 (unabhängig, σ = 1)."""
    rng = R.SplitMix64(seed)
    return np.array([beta * t + gumbel(rng) for t in range(m)])


def block_maxima(n_blocks: int, m: int, beta: float, seed: int) -> np.ndarray:
    """Maxima von n_blocks Blöcken zu je m Jahren (Jahr für Jahr gezogen, Lage β·t im Jahr t); jedes Maximum ist ein Ergebnis der Nutzungsdauer."""
    rng = R.SplitMix64(seed)
    out = np.empty(n_blocks)
    for i in range(n_blocks):
        out[i] = max(beta * t + gumbel(rng) for t in range(m))
    return out
