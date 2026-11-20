"""Vektorisierter SplitMix64-Strom: dieselben Gleichverteilten wie `tsp_rng.SplitMix64.random()`, nur schneller (Ganzzahlarithmetik in numpy uint64, bitgleich).

Nur für die eingefrorenen Tabellen (tools/tsp_make_grid.py): dort werden Millionen Werte gebraucht. Die Umrechnung in Gumbel- und Exponentialwerte läuft dort mit `numpy.log`
(auf dem letzten Bit plattformabhängig); die Tabellen werden deshalb nur als Band geprüft. Die App zieht live mit `tsp_rng` (`exact_log`)."""

import numpy as np

GAMMA = np.uint64(0x9E3779B97F4A7C15)
M1 = np.uint64(0xBF58476D1CE4E5B9)
M2 = np.uint64(0x94D049BB133111EB)
TWO53 = float(1 << 53)


class VecStream:
    def __init__(self, seed: int):
        self.state = np.uint64(seed & ((1 << 64) - 1))

    def uniform(self, n: int) -> np.ndarray:
        """Die nächsten n Gleichverteilten in [0, 1), 53 Bit."""
        with np.errstate(over="ignore"):
            k = np.arange(1, n + 1, dtype=np.uint64)
            z = self.state + GAMMA * k
            self.state = self.state + GAMMA * np.uint64(n)
            z = (z ^ (z >> np.uint64(30))) * M1
            z = (z ^ (z >> np.uint64(27))) * M2
            z = z ^ (z >> np.uint64(31))
        return (z >> np.uint64(11)).astype(np.float64) / TWO53

    def gumbel(self, n: int) -> np.ndarray:
        """Standard-Gumbel-Werte wie `tsp_scenario.gumbel` (E = −ln(1−u), X = −ln E; u = 0 wird durch die kleinste positive Zahl ersetzt)."""
        e = -np.log1p(-self.uniform(n))
        return -np.log(np.maximum(e, 1e-300))

    def exponential(self, n: int, mean: float = 1.0) -> np.ndarray:
        return -mean * np.log1p(-self.uniform(n))
