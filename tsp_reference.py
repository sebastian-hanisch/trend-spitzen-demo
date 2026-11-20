"""Lädt die eingefrorenen Rechnungen (`tsp_grid.json`, erzeugt von `tools/tsp_make_grid.py`; Quelle und Stand je Teil im Feld „quelle“ und „stand“).

Die schweren Zufallsmessungen (4 000 Läufe je Trennschärfe-Zelle, 400 Läufe je RMSE-Zelle, 400 000 Tage je Last des Vehikels A) laufen nicht live; die App zeigt sie neben den live gerechneten Größen."""

import json
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def grid():
    return json.loads((HERE / "tsp_grid.json").read_text(encoding="utf-8"))


def meta(part):
    g = grid()[part]
    return {"quelle": g["quelle"], "stand": g["stand"]}


def _find(part, **keys):
    for c in grid()[part]["daten"]:
        if all(abs(c[k] - v) < 1e-9 for k, v in keys.items()):
            return c
    return None


def typ1(m):
    """Fehlerquote erster Art (β = 0) bei m Jahren oder None."""
    return _find("typ1", m=m)


def power_cell(beta, m):
    """Trennschärfe-Zelle (β, m) oder None (β = 0 und m außerhalb von 20, 50, 100 gibt es nicht)."""
    return _find("power", beta=beta, m=m)


def years80(beta):
    return _find("years80", beta=beta)


def blocks(beta):
    return _find("blocks", beta=beta)


def rmse_cell(m, beta):
    return _find("rmse", m=m, beta=beta)


def bias_cell():
    return grid()["bias"]["daten"]


def coverage_cell(m, beta):
    return _find("coverage", m=m, beta=beta)


def vehicle_c1(rho=None):
    d = grid()["vehicle_c1"]["daten"]
    return d if rho is None else next((c for c in d if abs(c["rho"] - rho) < 1e-9), None)


def vehicle_c2(rho_end):
    return next((c for c in grid()["vehicle_c2"]["daten"] if abs(c["rho_bis"] - rho_end) < 1e-9), None)


def weekly():
    return grid()["weekly"]["daten"]


def rmse_counts():
    """Zählung der 12 RMSE-Zellen: wie oft das Trendmodell (c) gegen die stationäre Anpassung (a) klar besser, klar schlechter oder im Rauschen liegt."""
    out = {"c besser": 0, "c schlechter": 0, "Rauschen": 0}
    for c in grid()["rmse"]["daten"]:
        out[c["diff_c_a"]["urteil"]] += 1
    return out
