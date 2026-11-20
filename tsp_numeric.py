"""Kleine Zahlenwerkzeuge ohne scipy: Nelder-Mead, Brent-Wurzelsuche, Normalverteilung, Wilson-Intervall.

Die App rechnet nur mit numpy und der Standardbibliothek; scipy steht in `requirements-dev.txt` und wird nur in Tests als zweiter Rechenweg benutzt."""

import math

import numpy as np

SQRT2 = math.sqrt(2.0)
Z95 = 1.959963984540054          # Quantil der Standardnormalverteilung zu 0,975 (zweiseitig 95 %)
CHI2_95_DF1 = 3.841458820694124  # 95-%-Quantil der χ²-Verteilung mit einem Freiheitsgrad (= Z95²)


def norm_cdf(x: float) -> float:
    """Verteilungsfunktion der Standardnormalverteilung (math.erf, exakt auf Rundung)."""
    return 0.5 * (1.0 + math.erf(x / SQRT2))


def wilson(k: int, n: int, z: float = Z95):
    """Wilson-Intervall (zweiseitig 95 %) für einen Anteil k/n."""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2.0 * n)) / d
    h = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / d
    return c - h, c + h


def nelder_mead(f, x0, scale, tol_x=1e-8, tol_f=1e-11, maxit=1500):
    """Nelder-Mead mit numpy (Anfangssimplex: Startpunkt plus halbe Schrittweiten je Achse). Rückgabe (Minimalstelle, Minimum)."""
    x0 = np.asarray(x0, float)
    n = len(x0)
    sim = np.vstack([x0] + [x0 + scale * 0.5 * np.eye(n)[i] for i in range(n)])
    fv = np.array([f(s) for s in sim])
    for _ in range(maxit):
        o = np.argsort(fv)
        sim, fv = sim[o], fv[o]
        if np.max(np.abs(sim[1:] - sim[0])) <= tol_x and np.max(np.abs(fv[1:] - fv[0])) <= tol_f:
            break
        c = sim[:-1].mean(axis=0)
        xr = c + (c - sim[-1])
        fr = f(xr)
        if fr < fv[0]:
            xe = c + 2 * (c - sim[-1])
            fe = f(xe)
            sim[-1], fv[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < fv[-2]:
            sim[-1], fv[-1] = xr, fr
        else:
            xc = c + 0.5 * (xr - c) if fr < fv[-1] else c + 0.5 * (sim[-1] - c)
            fc = f(xc)
            if fc < min(fr, fv[-1]):
                sim[-1], fv[-1] = xc, fc
            else:
                sim[1:] = sim[0] + 0.5 * (sim[1:] - sim[0])
                fv[1:] = [f(s) for s in sim[1:]]
    i = int(np.argmin(fv))
    return sim[i], float(fv[i])


def brent(f, a, b, xtol=1e-12, maxiter=200):
    """Nullstelle von f in [a, b] (Vorzeichenwechsel nötig), Brent-Verfahren."""
    fa, fb = f(a), f(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0:
        raise ValueError("kein Vorzeichenwechsel")
    c, fc = a, fa
    d = e = b - a
    for _ in range(maxiter):
        if fb * fc > 0:
            c, fc = a, fa
            d = e = b - a
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol = 2.0 * 2.220446049250313e-16 * abs(b) + 0.5 * xtol
        m = 0.5 * (c - b)
        if abs(m) <= tol or fb == 0.0:
            return b
        if abs(e) >= tol and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:
                p = 2.0 * m * s
                q = 1.0 - s
            else:
                q = fa / fc
                r = fb / fc
                p = s * (2.0 * m * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0:
                q = -q
            p = abs(p)
            if 2.0 * p < min(3.0 * m * q - abs(tol * q), abs(e * q)):
                e, d = d, p / q
            else:
                d = e = m
        else:
            d = e = m
        a, fa = b, fb
        b += d if abs(d) > tol else (tol if m > 0 else -tol)
        fb = f(b)
    return b
