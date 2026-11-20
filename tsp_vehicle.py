"""Vehikel A: M/M/1-Wartezeit am Tor (Dienstrate 1, Ankunftsrate ρ), ein Tag = 256 aufeinanderfolgende Kunden (Annahme), Tage unabhängig, jeder Tag mit stationärem Start.

Das Vehikel ist ausgedacht und trägt keine Aussage über einen echten Betrieb. Exakt: im Gleichgewicht ist P(W > x) = ρ·e^{−(1−ρ)x}, P(W = 0) = 1−ρ.
Die Gumbel-Näherung des Tagesmaximums (Rechnung, nicht Simulation) hat Skala 1/(1−ρ) und Lage ln(n·θ·ρ)/(1−ρ) mit dem Extremalindex θ = (1−ρ)² (θ steht in der Demo „Spitzen-Cluster“ der Linie, Quelle dort; hier nur zitiert)."""

import math

import numpy as np

CUSTOMERS_PER_DAY = 256


def tail_probability(rho: float, x):
    """Exaktes Gleichgewicht der M/M/1-Wartezeit: P(W > x) = ρ·e^{−(1−ρ)x} für x ≥ 0."""
    if not 0.0 < rho < 1.0:
        raise ValueError("Die Auslastung ρ muss in (0, 1) liegen (ρ ≥ 1: keine stationäre Wartezeit).")
    return rho * np.exp(-(1.0 - rho) * np.asarray(x, float))


def gumbel_approximation(rho: float, n: int = CUSTOMERS_PER_DAY) -> tuple[float, float]:
    """(Lage, Skala) der Gumbel-Näherung des Tagesmaximums aus n Kunden: ln(n·θ·ρ)/(1−ρ) und 1/(1−ρ), θ = (1−ρ)²."""
    if not 0.0 < rho < 1.0:
        raise ValueError("Die Auslastung ρ muss in (0, 1) liegen.")
    g = 1.0 - rho
    return math.log(n * g * g * rho) / g, 1.0 / g


def approximate_level(rho: float, p: float = 0.01, n: int = CUSTOMERS_PER_DAY) -> float:
    """Pegel des Tagesmaximums nach der Gumbel-Näherung zur Überschreitungswahrscheinlichkeit p je Tag."""
    mu, s = gumbel_approximation(rho, n)
    return mu - s * math.log(-math.log1p(-p))


def lindley_days(rho, n: int, draw) -> np.ndarray:
    """Tagesmaxima der Wartezeit: `rho` ist ein Feld mit der Last je Tag (Länge = Zahl der Tage), n Kunden je Tag, Start im Gleichgewicht der Tageslast
    (P(W = 0) = 1−ρ, sonst Exp(1−ρ)), Tage unabhängig. `draw(k)` liefert k Gleichverteilte aus [0, 1) (bitgleiche Ströme: tsp_vecrng.VecStream.uniform, oder ein Aufruf der Vorlage)."""
    rho = np.asarray(rho, float)
    days = rho.size
    if np.any(rho <= 0) or np.any(rho >= 1):
        raise ValueError("Die Auslastung ρ muss in (0, 1) liegen.")
    u = draw(2 * days)
    w = np.where(u[:days] < rho, -np.log1p(-u[days:]) / (1.0 - rho), 0.0)
    mx = w.copy()
    for _ in range(n - 1):
        u = draw(2 * days)
        s = -np.log1p(-u[:days])
        a = -np.log1p(-u[days:]) / rho
        w = np.maximum(0.0, w + s - a)
        np.maximum(mx, w, out=mx)
    return mx
