"""Definition von Vehikel B (Verweilzeit-Mischung) als Testhilfe: eigene Aufzählung aus der Definition, unabhängig vom Erzeuger der Tabelle."""

import math

import numpy as np

ART_P = (0.35, 0.30, 0.25, 0.10)
U_SCALE = (1.0, 0.3, 0.3, 1.0)
S_BASE = (0.55, 0.25, 0.35, 0.90)
UK = (-0.42714751093638936, 0.3791185374387331, -0.2611985213877257, -0.07775197048031927, -0.02260299210315629, -0.2222653956256827,
      -0.410337810534883, 0.19466784065791196, 0.10831743391646849, -0.5858589189036569, 0.7042228963136555, 0.2905490717255771)


def _median(art, zoll, we, vor):
    if art == 0:
        return 3.5 * math.exp(0.7 * zoll + 0.30 * we)
    if art == 1:
        return 0.6 + 0.85 * vor
    if art == 2:
        return 0.3 + 0.8 * vor
    return 7.0 * math.exp(0.30 * we)


def aufzaehlung():
    """Eigene Aufzählung (Schleifen, getrennt von der Aufzählung des Erzeugers): Rückgabe (Zahl der Schritte, Zahl der Schritte mit Gewicht, Feld (μ, s) → Gewicht)."""
    xg, wg = np.polynomial.legendre.leggauss(48)
    schritte = mit_gewicht = 0
    summe = {}
    for art in range(4):
        for kunde in range(12):
            for zoll, pz in (((0, 0.75), (1, 0.25)) if art == 0 else ((0, 1.0), (1, 0.0))):
                for we, pw in ((0, 5 / 7), (1, 2 / 7)):
                    if art in (1, 2):
                        lo, hi = (1.0, 10.0) if art == 1 else (0.5, 6.0)
                        knoten = [(lo + (hi - lo) * (x + 1) / 2, w / 2) for x, w in zip(xg, wg)]
                    else:
                        knoten = [(0.0, 1.0)]
                    for vor, wv in knoten:
                        schritte += 1
                        gew = ART_P[art] / 12 * pz * pw * wv
                        if gew > 0:
                            mit_gewicht += 1
                            key = (round(math.log(_median(art, zoll, we, vor)) + UK[kunde] * U_SCALE[art], 10), S_BASE[art])
                            summe[key] = summe.get(key, 0.0) + gew
    return schritte, mit_gewicht, summe
