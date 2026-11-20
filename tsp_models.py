"""Kern: Gumbel- und GEV-Anpassung mit und ohne Lage-Trend, exakte Größen des Gumbel-Trend-Modells (Mischung, Risiko, Design Life Level, Trennschärfe).
Nur Standardbibliothek und numpy (kein scipy in der App).

Modell (ausgedacht, keine Daten): Jahresmaxima Gumbel mit Lage μ_t = β·t (t = 0 … m−1, σ = 1), unabhängig; β in Einheiten σ je Jahr, Gesamtanstieg D = β·(m−1).
„Heute“ ist das letzte Jahr der Reihe (t = m−1). Pegel = Wiederkehrwert zur Überschreitungswahrscheinlichkeit p im Jahr t: x_t(p) = β·t − ln(−ln(1−p)).
Vorzeichen-Konvention der Linie: ξ > 0 Fréchet-Typ, ξ = 0 Gumbel, ξ < 0 Weibull-Typ; `scipy.stats.genextreme` hat c = −ξ (geprüft in tests/test_oracle_gev_trend.py).
Zeit in den Anpassungen zentriert: c_t = t − (m−1)/2, Lage μ_t = a + b·c_t; die Lage im letzten Jahr ist a + b·(m−1)/2."""

import math
from dataclasses import dataclass

import numpy as np

import tsp_numeric as N

EULER = 0.5772156649015329
P_HUNDRED = 0.01                                   # Überschreitungswahrscheinlichkeit je Jahr des 100-Jahres-Pegels
Y100 = -math.log(-math.log(1.0 - P_HUNDRED))       # Grundhöhe des 100-Jahres-Pegels der Standard-Gumbel-Verteilung (4,600149)
XI_MIN = -1.0                                      # Nebenbedingung der GEV-Anpassung (Wahl dieser Demo)
HUGE = 1e300
XI_GUMBEL = 1e-9


class FitError(ValueError):
    """Anpassung nicht möglich (zu wenige Jahre, alle Werte gleich): die App zeigt eine Meldung statt einer Zahl."""


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Exakte Größen des Gumbel-Trend-Modells
# ---------------------------------------------------------------------------------------------------------------------------------------------
def level_year(beta: float, t: float, p: float = P_HUNDRED) -> float:
    """Pegel des Jahres t zur Überschreitungswahrscheinlichkeit p: β·t − ln(−ln(1−p)) (exakt)."""
    return beta * t - math.log(-math.log1p(-p))


def mixture_cdf(x: float, beta: float, m: int) -> float:
    """Verteilungsfunktion der stationären Mischung der m Jahresverteilungen: (1/m)·Σ_t exp(−e^{−(x − β t)}) (exakt)."""
    t = np.arange(m, dtype=float)
    with np.errstate(over="ignore"):
        return float(np.mean(np.exp(-np.exp(-(x - beta * t)))))


def mixture_quantile(beta: float, m: int, q: float = 1.0 - P_HUNDRED) -> float:
    """Quantil q der Mischung (Brent auf F_mix − q); liegt zwischen dem Pegel des ersten und dem des letzten Jahres."""
    if m < 1:
        raise ValueError("m >= 1")
    first, last = level_year(beta, 0, 1.0 - q), level_year(beta, m - 1, 1.0 - q)      # das Quantil der Mischung liegt zwischen den Pegeln des ersten und des letzten Jahres
    lo, hi = min(first, last) - 1.0, max(first, last) + 1.0
    return N.brent(lambda x: mixture_cdf(x, beta, m) - q, lo, hi, xtol=1e-13)


def log_sum_exp_years(beta: float, m: int, start: int = 0) -> float:
    """ln S mit S = Σ_{t=start}^{start+m−1} e^{β t} (Log-Summe, läuft auch bei großem β nicht über)."""
    e = beta * (start + np.arange(m, dtype=float))
    mx = float(np.max(e))
    return mx + math.log(float(np.sum(np.exp(e - mx))))


def risk_in_years(beta: float, m: int, p: float = P_HUNDRED, start: int = 0) -> float:
    """Wahrscheinlichkeit, dass der Pegel von heute (Überschreitungswahrscheinlichkeit p im Jahr 0 der Zählung) in m Jahren mindestens einmal überschritten wird:
    1 − (1−p)^S mit S = Σ e^{β t}. start = 0: die Jahre 0 … m−1 (das heutige Jahr eingeschlossen); start = 1: die Jahre 1 … m (m Jahre ab morgen)."""
    ln_s = log_sum_exp_years(beta, m, start)
    ex = math.exp(ln_s) * math.log1p(-p) if ln_s < 700 else -math.inf
    return float(-math.expm1(ex))


def design_life_level(beta: float, m: int, q: float = 0.05) -> float:
    """Design Life Level (Rootzén/Katz 2013, Definition): oberes Quantil der Verteilung des Maximums über m Jahre zum Risiko q: ln S − ln(−ln(1−q)) (exakt, Gumbel)."""
    if not 0.0 < q < 1.0:
        raise ValueError("0 < q < 1")
    return log_sum_exp_years(beta, m) - math.log(-math.log1p(-q))


def return_period_after(beta: float, t: float, p: float = P_HUNDRED) -> float:
    """Wiederkehrperiode (Mittel der geometrischen Wartezeit) des heutigen Pegels im Jahr t nach heute: 1/(1 − (1−p)^{e^{β t}})."""
    ex = math.exp(beta * t) * math.log1p(-p) if beta * t < 700 else -math.inf
    pr = -math.expm1(ex)
    return math.inf if pr <= 0 else 1.0 / pr


def harmonic(m: int) -> float:
    return float(sum(1.0 / k for k in range(1, m + 1)))


def expected_records(beta: float, m: int) -> float:
    """Erwartete Zahl neuer Höchstwerte in m Jahren: Σ_k λ_k/Σ_{i≤k} λ_i mit λ_t = e^{β t} (exakt; ohne Trend H_m)."""
    e = beta * np.arange(m, dtype=float)
    return float(np.sum(np.exp(e - np.logaddexp.accumulate(e))))      # λ_k / Σ_{i≤k} λ_i in Log-Form (läuft bei großem β nicht über)


def count_records(x) -> int:
    """Zahl der neuen Höchstwerte in der Reihe (das erste Jahr zählt)."""
    best, c = -math.inf, 0
    for v in x:
        if v > best:
            c += 1
            best = float(v)
    return c


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Erkennbarkeit: Information, Verschiebungsparameter, Trennschärfe
# ---------------------------------------------------------------------------------------------------------------------------------------------
def gumbel_information() -> dict:
    """Fisher-Information der Gumbel-Verteilung (μ, σ) bei σ = 1 in geschlossener Form: (1; −(1−γ); (1−γ)² + π²/6)."""
    return {"mu_mu": 1.0, "mu_sigma": -(1.0 - EULER), "sigma_sigma": (1.0 - EULER) ** 2 + math.pi ** 2 / 6.0}


def sum_centered_squares(m: int) -> float:
    """Σ (t − t̄)² über t = 0 … m−1 = m(m²−1)/12."""
    return m * (m * m - 1) / 12.0


def noncentrality(beta: float, m: int) -> float:
    """Verschiebungsparameter λ = β²·m(m²−1)/12 (σ = 1)."""
    return beta * beta * sum_centered_squares(m)


def power(beta: float, m: int, z: float = N.Z95) -> float:
    """Asymptotische Trennschärfe des Likelihood-Quotienten-Tests (Niveau 5 %; Gumbel mit gegen ohne Lage-Trend, ξ = 0 bekannt): Φ(√λ − 1,96) + Φ(−√λ − 1,96)."""
    r = math.sqrt(noncentrality(beta, m))
    return N.norm_cdf(r - z) + N.norm_cdf(-r - z)


def years_for_power(beta: float, target: float = 0.8, m_max: int = 5000):
    """Kleinstes m ≥ 3 mit Formel-Trennschärfe ≥ target (None, wenn bis m_max nicht erreicht, insbesondere bei β = 0)."""
    if beta == 0.0:
        return None
    for m in range(3, m_max + 1):
        if power(beta, m) >= target:
            return m
    return None


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Gumbel mit Lage-Trend: ML per Newton-Verfahren mit analytischem Gradienten und Hesse-Matrix
# ---------------------------------------------------------------------------------------------------------------------------------------------
def centered_time(m: int) -> np.ndarray:
    return np.arange(m, dtype=float) - (m - 1) / 2.0


@dataclass(frozen=True)
class GumbelFit:
    a: float           # Lage in der Mitte der Reihe
    b: float           # Trend je Jahr (σ je Jahr, wenn σ = 1)
    sigma: float
    nll: float         # negative Log-Likelihood im Optimum
    m: int
    trend: bool

    def loc(self, t) -> np.ndarray:
        """Angepasste Lage in den Jahren t (0 … m−1)."""
        return self.a + self.b * (np.asarray(t, float) - (self.m - 1) / 2.0)

    def level(self, p: float = P_HUNDRED, t: float | None = None) -> float:
        """Pegel zur Überschreitungswahrscheinlichkeit p im Jahr t (Standard: letztes Jahr)."""
        t = self.m - 1 if t is None else t
        return float(self.loc(t) - self.sigma * math.log(-math.log1p(-p)))


def _gumbel_nll(theta, x, c):
    a, b, s = theta
    if not s > 0:
        return HUGE
    z = (x - a - b * c) / s
    with np.errstate(over="ignore"):
        e = np.exp(-z)
    v = x.size * math.log(s) + float(np.sum(z) + np.sum(e))
    return v if math.isfinite(v) else HUGE


def gumbel_nll_grad_hess(theta, x, c):
    """Negative Log-Likelihood, Gradient und Hesse-Matrix der Gumbel-Verteilung mit Lage a + b·c und Skala s, nach θ = (a, b, s)."""
    a, b, s = theta
    z = (x - a - b * c) / s
    e = np.exp(-z)
    h1, h2 = 1.0 - e, e                            # h'(z), h''(z) für h(z) = z + e^{−z}
    ua, ub, us = -1.0 / s, -c / s, -z / s          # ∂z/∂θ
    f = x.size * math.log(s) + float(np.sum(z) + np.sum(e))
    g = np.array([np.sum(h1 * ua), np.sum(h1 * ub), x.size / s + np.sum(h1 * us)])
    U = np.vstack([ua * np.ones_like(z), ub, us])
    H = (U * h2) @ U.T
    # zweite Ableitungen der inneren Funktionen z(θ): ∂u_a/∂s = 1/s², ∂u_b/∂s = c/s², ∂u_s/∂a = 1/s², ∂u_s/∂b = c/s², ∂u_s/∂s = 2z/s²
    d_as = float(np.sum(h1)) / s ** 2
    d_bs = float(np.sum(h1 * c)) / s ** 2
    d_ss = float(np.sum(h1 * 2.0 * z)) / s ** 2
    H[0, 2] += d_as
    H[2, 0] += d_as
    H[1, 2] += d_bs
    H[2, 1] += d_bs
    H[2, 2] += d_ss - x.size / s ** 2
    return f, g, H


def fit_gumbel(x, trend: bool = True, maxit: int = 100) -> GumbelFit:
    """Gumbel-ML mit (trend=True) oder ohne Lage-Trend (b = 0), Newton-Verfahren mit Rückschritt. Braucht m ≥ 3 mit Trend, m ≥ 2 ohne, und nicht alle Werte gleich."""
    x = np.asarray(x, float)
    m = x.size
    if m < (3 if trend else 2):
        raise FitError(f"Die Anpassung braucht mindestens {3 if trend else 2} Jahresmaxima.")
    if not float(np.ptp(x)) > 0:
        raise FitError("Alle Jahresmaxima sind gleich: keine Anpassung möglich.")
    c = centered_time(m)
    b0 = float(np.sum(c * (x - x.mean())) / np.sum(c * c)) if trend else 0.0
    res = x - x.mean() - b0 * c
    s0 = float(res.std(ddof=1)) * math.sqrt(6.0) / math.pi
    if not s0 > 0:
        raise FitError("Die Reihe liegt exakt auf einer Geraden: keine Anpassung möglich.")
    theta = np.array([float(x.mean()) - EULER * s0, b0, s0])
    idx = [0, 1, 2] if trend else [0, 2]
    f, g, H = gumbel_nll_grad_hess(theta, x, c)
    for _ in range(maxit):
        gi, Hi = g[idx], H[np.ix_(idx, idx)]
        try:
            w, V = np.linalg.eigh(Hi)
            w = np.maximum(np.abs(w), 1e-8)           # nicht positiv definit: Betrag der Eigenwerte (fester Abstieg)
            step_i = V @ ((V.T @ gi) / w)
        except np.linalg.LinAlgError:
            step_i = gi
        step = np.zeros(3)
        step[idx] = step_i
        t = 1.0
        while True:
            new = theta - t * step
            fn = _gumbel_nll(new, x, c)
            if fn <= f + 1e-4 * t * float(g @ step) or t < 1e-12:
                break
            t *= 0.5
        if t < 1e-12:
            break
        done = abs(fn - f) <= 1e-14 * (1.0 + abs(f)) and float(np.max(np.abs(t * step))) <= 1e-10 * (1.0 + float(np.max(np.abs(theta))))
        theta, f = new, fn
        f, g, H = gumbel_nll_grad_hess(theta, x, c)
        if done or float(np.max(np.abs(g[idx]))) < 1e-10:
            break
    return GumbelFit(float(theta[0]), float(theta[1]), float(theta[2]), float(f), m, trend)


def likelihood_ratio(x) -> tuple[float, GumbelFit, GumbelFit]:
    """Likelihood-Quotienten-Statistik 2(ℓ_Trend − ℓ_stationär) der Gumbel-Anpassungen (zu vergleichen mit CHI2_95_DF1 = 3,8415) und beide Anpassungen."""
    f0, f1 = fit_gumbel(x, False), fit_gumbel(x, True)
    return max(0.0, 2.0 * (f0.nll - f1.nll)), f0, f1


# ---------------------------------------------------------------------------------------------------------------------------------------------
# GEV (ξ frei), stationär und mit Lage-Trend: Nelder-Mead
# ---------------------------------------------------------------------------------------------------------------------------------------------
def gev_level(p, mu, sigma, xi):
    """Pegel zur Überschreitungswahrscheinlichkeit p je Jahr: μ + σ/ξ·((−ln(1−p))^{−ξ} − 1); ξ = 0 als Grenzwert (Coles 2001, Gl. 3.4)."""
    with np.errstate(divide="ignore", over="ignore"):
        y = -np.log1p(-np.asarray(p, float))
        if abs(xi) < XI_GUMBEL:
            return mu - sigma * np.log(y)
        return mu + sigma * (np.power(y, -xi) - 1.0) / xi


def gev_nll(q, x, c=None):
    """Negative Log-Likelihood der GEV mit Lage a + b·c (c = None: stationär, q = (a, ln σ, ξ); sonst q = (a, b, ln σ, ξ)); HUGE außerhalb des Trägers oder für ξ < −1."""
    if c is None:
        a, ls, xi = q
        loc = a
    else:
        a, b, ls, xi = q
        loc = a + b * c
    if xi < XI_MIN or not math.isfinite(ls):
        return HUGE
    z = (x - loc) * math.exp(-ls)
    if abs(xi) < 1e-7:
        return x.size * ls + float(np.sum(z) + np.sum(np.exp(-z)))
    t = 1.0 + xi * z
    if np.any(t <= 0):
        return HUGE
    lt = np.log(t)
    v = x.size * ls + (1.0 + 1.0 / xi) * float(np.sum(lt)) + float(np.sum(np.exp(-lt / xi)))
    return v if math.isfinite(v) else HUGE


@dataclass(frozen=True)
class GevFit:
    a: float
    b: float            # 0 bei der stationären Anpassung
    sigma: float
    xi: float
    nll: float
    m: int
    trend: bool

    @property
    def edge(self) -> bool:
        """ξ̂ klebt an der Nebenbedingung ξ = −1."""
        return self.xi <= XI_MIN + 1e-3

    def level(self, p: float = P_HUNDRED, t: float | None = None) -> float:
        t = self.m - 1 if t is None else t
        return float(gev_level(p, self.a + self.b * (t - (self.m - 1) / 2.0), self.sigma, self.xi))


def _best_nm(fn, starts, scale):
    best = None
    for p0 in starts:
        p1, f1 = N.nelder_mead(fn, p0, scale, 1e-7, 1e-10)
        p2, f2 = N.nelder_mead(fn, p1, scale * 0.1, 1e-8, 1e-11)
        p, f = (p2, f2) if f2 <= f1 else (p1, f1)
        if best is None or f < best[1]:
            best = (p, f)
    return best


def fit_gev(x, trend: bool = False) -> GevFit:
    """GEV-ML (ξ ≥ −1, ξ nach oben frei) ohne oder mit Lage-Trend. Start aus der Gumbel-Anpassung (ξ₀ = 0,02 und −0,1), zwei Nelder-Mead-Durchgänge je Start."""
    x = np.asarray(x, float)
    m = x.size
    if m < (5 if trend else 4):
        raise FitError(f"Die GEV-Anpassung braucht mindestens {5 if trend else 4} Jahresmaxima (mehr als freie Parameter).")
    g = fit_gumbel(x, trend)
    c = centered_time(m) if trend else None
    if trend:
        starts = [np.array([g.a, g.b, math.log(g.sigma), xi0]) for xi0 in (0.02, -0.1)]
        scale = np.array([g.sigma, abs(g.b) + 0.01, 0.1, 0.1])
    else:
        starts = [np.array([g.a, math.log(g.sigma), xi0]) for xi0 in (0.02, -0.1)]
        scale = np.array([g.sigma, 0.1, 0.1])
    p, f = _best_nm(lambda q: gev_nll(q, x, c), starts, scale)
    if not f < HUGE / 10:
        raise FitError("Die GEV-Anpassung fand keinen gültigen Punkt.")
    if trend:
        return GevFit(float(p[0]), float(p[1]), math.exp(float(p[2])), float(p[3]), float(f), m, True)
    return GevFit(float(p[0]), 0.0, math.exp(float(p[1])), float(p[2]), float(f), m, False)


def delta_interval(fit: GevFit, x, p: float = P_HUNDRED, level: float = 0.95):
    """Delta-Intervall (beobachtete Information, numerische Hesse-Matrix in (μ, ln σ, ξ)) für den Pegel der stationären GEV-Anpassung.
    Rückgabe (unten, oben, Standardfehler) oder None, wenn die Hesse-Matrix nicht positiv definit ist."""
    if fit.trend:
        raise ValueError("nur für die stationäre Anpassung")
    x = np.asarray(x, float)
    th = np.array([fit.a, math.log(fit.sigma), fit.xi])
    fn = lambda q: gev_nll(q, x)
    h = np.array([1e-3, 1e-3, 1e-3])
    H = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            ei, ej = np.zeros(3), np.zeros(3)
            ei[i], ej[j] = h[i], h[j]
            H[i, j] = (fn(th + ei + ej) - fn(th + ei - ej) - fn(th - ei + ej) + fn(th - ei - ej)) / (4.0 * h[i] * h[j])
    if not np.all(np.isfinite(H)) or np.any(np.linalg.eigvalsh(H) <= 0):
        return None
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None
    lf = lambda q: float(gev_level(p, q[0], math.exp(q[1]), q[2]))
    grad = np.array([(lf(th + np.eye(3)[i] * 1e-5) - lf(th - np.eye(3)[i] * 1e-5)) / 2e-5 for i in range(3)])
    var = float(grad @ cov @ grad)
    if not (var > 0 and math.isfinite(var)):
        return None
    zq = 1.959963984540054 if level == 0.95 else None
    if zq is None:
        raise ValueError("nur Niveau 0,95")
    se = math.sqrt(var)
    z = lf(th)
    return z - zq * se, z + zq * se, se


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Gumbel mit Trend in Lage und ln σ (Vehikel A, Modell (c))
# ---------------------------------------------------------------------------------------------------------------------------------------------
def gumbel_loc_logscale_nll(q, x, c):
    """Negative Log-Likelihood der Gumbel-Verteilung mit Lage a + b·c und Skala exp(s0 + s1·c); q = (a, b, s0, s1)."""
    a, b, s0, s1 = q
    ls = s0 + s1 * c
    z = (x - a - b * c) * np.exp(-ls)
    with np.errstate(over="ignore"):
        v = float(np.sum(ls) + np.sum(z) + np.sum(np.exp(-z)))
    return v if math.isfinite(v) else HUGE


@dataclass(frozen=True)
class GumbelLogscaleFit:
    a: float
    b: float
    s0: float
    s1: float
    nll: float
    m: int

    def level(self, p: float = P_HUNDRED) -> float:
        """Pegel im letzten Jahr."""
        cl = (self.m - 1) / 2.0
        return float(self.a + self.b * cl - math.exp(self.s0 + self.s1 * cl) * math.log(-math.log1p(-p)))


def fit_gumbel_loc_logscale(x) -> GumbelLogscaleFit:
    """Gumbel (ξ = 0) mit Trend in Lage und ln σ (4 Parameter), Nelder-Mead, Start aus der Gumbel-Trend-Anpassung."""
    x = np.asarray(x, float)
    m = x.size
    if m < 5:
        raise FitError("Die Anpassung braucht mindestens 5 Tage.")
    g = fit_gumbel(x, True)
    c = centered_time(m)
    p0 = np.array([g.a, g.b, math.log(g.sigma), 0.0])
    scale = np.array([g.sigma, abs(g.b) + 0.01, 0.1, 0.01])
    p, f = _best_nm(lambda q: gumbel_loc_logscale_nll(q, x, c), [p0], scale)
    return GumbelLogscaleFit(float(p[0]), float(p[1]), float(p[2]), float(p[3]), float(f), m)
