"""Erzeugt `tsp_grid.json`: die schweren Rechnungen der Demo, eingefroren (Quelle und Stand je Feld).

Inhalt (Zufall: SplitMix64-Strom der Vorlage, vektorisiert in `tsp_vecrng.py` und bitgleich zur Vorlage; Umrechnung mit numpy.log, daher nur als Band prüfbar):
  typ1      Fehlerquote erster Art des Trendtests (β = 0), m = 10, 20, 30, 50, 100, R = 4 000 Läufe
  power     Trennschärfe, β = 0,01 / 0,02 / 0,04 × m = 20 / 50 / 100, R = 4 000 Läufe (Formel gegen Simulation)
  years80   Simulation beim kleinsten m mit Formel-Trennschärfe ≥ 0,8 (β = 0,005 / 0,01 / 0,02 / 0,04), R = 2 000 Läufe
  blocks    400 000 Blöcke zu 50 Jahren je β: Anteil über dem Design Life Level (q = 5 %) und über dem heutigen 100-Jahres-Pegel
  rmse      12 Zellen (m = 20, 50, 100 × β = 0, 0,01, 0,02, 0,04), je 400 Läufe: stationäre GEV, Gumbel mit Trend (ξ = 0 bekannt), GEV mit Trend; gepaarter Standardfehler; Delta-Abdeckung
  bias      m = 50, β = 0,02: Verzerrung der stationären GEV, 3 000 Läufe
  coverage  Delta-Intervall der stationären GEV für das letzte und das mittlere Jahr, vier Zellen, 1 000 Läufe
  vehicle_a Vehikel A (M/M/1, 256 Kunden je Tag): Skala und Lage über der Last (400 000 Tage je Last) und falsches Modell bei wachsender Last (300 Läufe zu 100 Tagen)
  weekly    Vehikel B, Wochenstruktur: Pegel zu p = 1e-5 nur Werktage und nur Wochenende (exakt, ohne Zufall)

Aufruf:  python tools/tsp_make_grid.py              erzeugt die Datei (mehrere Minuten, 6 Prozesse)
         python tools/tsp_make_grid.py --check      rechnet vier kleine Zellen neu und vergleicht als Band"""

import argparse
import json
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import tsp_models as M  # noqa: E402
import tsp_numeric as N  # noqa: E402
import tsp_scenario as S  # noqa: E402
import tsp_vecrng as V  # noqa: E402
import tsp_vehicle as VH  # noqa: E402

FILE = ROOT / "tsp_grid.json"
BASE = 20261121          # Zufalls-Seed-Basis dieser Demo (nicht die der Vormessung)
STAND = "2026-11-20"
PROCESSES = 6
Y100 = M.Y100


def seed(*tags):
    return S.derive_seed(BASE, *tags)


def series(vs, R, m, beta):
    """R Reihen zu m Jahren: Gumbel mit Lage β·t."""
    return beta * np.arange(m, dtype=float) + vs.gumbel(R * m).reshape(R, m)


def lr_hits(X):
    hits = failed = 0
    for x in X:
        try:
            hits += M.likelihood_ratio(x)[0] > N.CHI2_95_DF1
        except M.FitError:
            failed += 1
    return int(hits), failed


# ------------------------------------------------------------------------------------------------------------------------------------------
def t_typ1(m, R=4000):
    hits, failed = lr_hits(series(V.VecStream(seed(1, m)), R, m, 0.0))
    return {"m": m, "R": R, "hits": hits, "ausgefallen": failed}


def t_power(beta, m, R=4000, tag=2):
    hits, failed = lr_hits(series(V.VecStream(seed(tag, m, round(beta * 1000))), R, m, beta))
    return {"beta": beta, "m": m, "R": R, "hits": hits, "ausgefallen": failed, "formel": M.power(beta, m)}


def t_years80(beta, R=2000):
    m = M.years_for_power(beta, 0.8)
    out = t_power(beta, m, R, tag=3)
    return out


def t_blocks(beta, m=50, n_blocks=400_000, chunk=20_000):
    vs = V.VecStream(seed(4, round(beta * 1000)))
    dll = M.design_life_level(beta, m, 0.05)
    over_dll = over_today = 0
    t = np.arange(m, dtype=float)
    for _ in range(n_blocks // chunk):
        mx = (beta * t + vs.gumbel(chunk * m).reshape(chunk, m)).max(axis=1)
        over_dll += int(np.sum(mx > dll))
        over_today += int(np.sum(mx > Y100))
    return {"beta": beta, "m": m, "blocks": n_blocks, "ueber_dll": over_dll, "ueber_heute": over_today,
            "dll": dll, "risiko_exakt": M.risk_in_years(beta, m)}


def err_stats(e):
    return {"bias": float(np.mean(e)), "sd": float(np.std(e, ddof=1)), "rmse": float(math.sqrt(np.mean(e * e))), "med_abs": float(np.median(np.abs(e)))}


def paired_diff(ea, ec, vs, reps=2000):
    """RMSE(c) − RMSE(a) mit gepaartem Bootstrap-Standardfehler (gleiche Reihen, Indizes aus dem Strom der Vorlage)."""
    n = ea.size
    ix = np.minimum((vs.uniform(reps * n) * n).astype(np.int64), n - 1).reshape(reps, n)
    d = np.sqrt(np.mean(ec[ix] ** 2, axis=1)) - np.sqrt(np.mean(ea[ix] ** 2, axis=1))
    d0 = math.sqrt(np.mean(ec ** 2)) - math.sqrt(np.mean(ea ** 2))
    se = float(np.std(d, ddof=1))
    z = abs(d0) / se if se > 0 else 0.0
    return {"diff": float(d0), "se": se, "z": float(z), "urteil": "Rauschen" if z < 2 else ("c besser" if d0 < 0 else "c schlechter")}


def stationary_runs(X, beta, with_cov=True):
    """Stationäre GEV je Reihe: Fehler am letzten Jahr, ξ̂, Delta-Abdeckung (letztes und mittleres Jahr)."""
    m = X.shape[1]
    t_last, t_mid = beta * (m - 1) + Y100, beta * (m - 1) / 2 + Y100
    ea, xi, edge, cl, cm, none, failed = [], [], 0, 0, 0, 0, 0
    for x in X:
        try:
            f = M.fit_gev(x, False)
        except M.FitError:
            failed += 1
            continue
        lev = f.level()
        ea.append(lev - t_last)
        xi.append(f.xi)
        edge += f.edge
        if with_cov:
            ci = M.delta_interval(f, x)
            if ci is None:
                none += 1
            else:
                cl += bool(ci[0] <= t_last <= ci[1])
                cm += bool(ci[0] <= t_mid <= ci[1])
    return np.array(ea), np.array(xi), edge, cl, cm, none, failed


def t_rmse(m, beta, R=400):
    vs = V.VecStream(seed(5, m, round(beta * 1000)))
    X = series(vs, R, m, beta)
    t_last = beta * (m - 1) + Y100
    ea, eb, ec, xa, xc, edge_a, edge_c, cl, cm, none, failed = [], [], [], [], [], 0, 0, 0, 0, 0, 0
    t_mid = beta * (m - 1) / 2 + Y100
    for x in X:
        try:
            fa = M.fit_gev(x, False)
            fb = M.fit_gumbel(x, True)
            fc = M.fit_gev(x, True)
        except M.FitError:
            failed += 1
            continue
        ea.append(fa.level() - t_last)
        eb.append(fb.level() - t_last)
        ec.append(fc.level() - t_last)
        xa.append(fa.xi)
        xc.append(fc.xi)
        edge_a += fa.edge
        edge_c += fc.edge
        ci = M.delta_interval(fa, x)
        if ci is None:
            none += 1
        else:
            cl += bool(ci[0] <= t_last <= ci[1])
            cm += bool(ci[0] <= t_mid <= ci[1])
    ea, eb, ec = np.array(ea), np.array(eb), np.array(ec)
    n = ea.size
    a, b, c = err_stats(ea), err_stats(eb), err_stats(ec)
    b["besser_als_a"] = float(np.mean(np.abs(eb) < np.abs(ea)))
    c["besser_als_a"] = float(np.mean(np.abs(ec) < np.abs(ea)))
    a["xi_mittel"], c["xi_mittel"] = float(np.mean(xa)), float(np.mean(xc))
    a["am_rand"], c["am_rand"] = int(edge_a), int(edge_c)
    a["anteil_unter_wahr"] = float(np.mean(ea < 0))
    return {"m": m, "beta": beta, "R": n, "ausgefallen": failed, "a": a, "b": b, "c": c,
            "diff_c_a": paired_diff(ea, ec, V.VecStream(seed(6, m, round(beta * 1000)))),
            "abdeckung": {"letztes": cl / n, "mittleres": cm / n, "kein_intervall": none}}


def t_bias(m=50, beta=0.02, R=3000):
    X = series(V.VecStream(seed(7, m, round(beta * 1000))), R, m, beta)
    ea, xi, edge, _cl, _cm, _none, failed = stationary_runs(X, beta, with_cov=False)
    d = err_stats(ea)
    d.update({"m": m, "beta": beta, "R": int(ea.size), "se_bias": d["sd"] / math.sqrt(ea.size), "anteil_unter_wahr": float(np.mean(ea < 0)),
              "xi_mittel": float(np.mean(xi)), "am_rand": int(edge), "ausgefallen": failed})
    return d


def t_coverage(m, beta, R=1000):
    X = series(V.VecStream(seed(8, m, round(beta * 1000))), R, m, beta)
    ea, xi, edge, cl, cm, none, failed = stationary_runs(X, beta)
    n = ea.size
    return {"m": m, "beta": beta, "R": int(n), "letztes": cl / n, "mittleres": cm / n, "kein_intervall": none, "ausgefallen": failed,
            "se_punkte": 100 * math.sqrt(0.25 / n)}


# ------------------------------------------------------------------------------------------------------------------------------------------
def big_days(rho, n_days, vs, chunk=20_000):
    out = []
    for _ in range(n_days // chunk):
        out.append(VH.lindley_days(np.full(chunk, rho), VH.CUSTOMERS_PER_DAY, vs.uniform))
    return np.concatenate(out)


def ks_distance(sample, cdf):
    x = np.sort(sample)
    n = x.size
    f = cdf(x)
    return float(max(np.max(np.arange(1, n + 1) / n - f), np.max(f - np.arange(0, n) / n)))


def quantile_with_se(x, q=0.99, parts=20):
    """Quantil und sein Standardfehler aus der Streuung über `parts` gleich große Teilstichproben (bei so wenigen Teilen eher zu klein geschätzt)."""
    qs = [float(np.quantile(p, q)) for p in np.array_split(x, parts)]
    return float(np.quantile(x, q)), float(np.std(qs, ddof=1) / math.sqrt(parts))


def t_vehicle_c1(rho, n_days=400_000):
    vs = V.VecStream(seed(9, round(rho * 100)))
    mx = big_days(rho, n_days, vs)
    f = M.fit_gumbel(mx, False)
    mu_c, s_c = VH.gumbel_approximation(rho)
    q99, q_se = quantile_with_se(mx)
    return {"rho": rho, "tage": int(mx.size), "sigma_dach": f.sigma, "mu_dach": f.a, "sigma_rechnung": s_c, "mu_rechnung": mu_c,
            "pegel_simulation": q99, "pegel_simulation_se": q_se, "pegel_rechnung": VH.approximate_level(rho),
            "ks_rechnung": ks_distance(mx, lambda v: np.exp(-np.exp(-(v - mu_c) / s_c)))}


def log_ratio_stats(r):
    r = r[np.isfinite(r) & (r > 0)]
    return {"median": float(np.median(r)), "q10": float(np.quantile(r, 0.1)), "q90": float(np.quantile(r, 0.9)), "anteil_unter_1": float(np.mean(r < 1)),
            "sd_ln": float(np.std(np.log(r), ddof=1)), "rmse_ln": float(math.sqrt(np.mean(np.log(r) ** 2))), "n": int(r.size)}


def t_vehicle_c2(rho_end, R=300, m=100, start=0.70):
    vs = V.VecStream(seed(10, round(rho_end * 100)))
    truth_days = big_days(rho_end, 400_000, V.VecStream(seed(11, round(rho_end * 100))))
    truth, truth_se = quantile_with_se(truth_days)
    rhos = np.tile(np.linspace(start, rho_end, m), R)
    X = VH.lindley_days(rhos, VH.CUSTOMERS_PER_DAY, vs.uniform).reshape(R, m)
    la, lb, lc, xa, xb = [], [], [], [], []
    for x in X:
        fa = M.fit_gev(x, False)
        fb = M.fit_gev(x, True)
        fc = M.fit_gumbel_loc_logscale(x)
        la.append(fa.level())
        lb.append(fb.level())
        lc.append(fc.level())
        xa.append(fa.xi)
        xb.append(fb.xi)
    # Kontrolle: alle Tage bei ρ_end, stationäre GEV
    Kd =VH.lindley_days(np.full(R * m, rho_end), VH.CUSTOMERS_PER_DAY, V.VecStream(seed(12, round(rho_end * 100))).uniform).reshape(R, m)
    lk, xk = [], []
    for x in Kd:
        fk = M.fit_gev(x, False)
        lk.append(fk.level())
        xk.append(fk.xi)
    out = {"rho_von": start, "rho_bis": rho_end, "tage_je_lauf": m, "R": R, "wahr": truth, "wahr_se": truth_se, "wahr_tage": int(truth_days.size),
           "naeherung": VH.approximate_level(rho_end),
           "stationaer": log_ratio_stats(np.array(la) / truth), "lage_trend": log_ratio_stats(np.array(lb) / truth),
           "lage_skala_trend": log_ratio_stats(np.array(lc) / truth), "kontrolle_ohne_trend": log_ratio_stats(np.array(lk) / truth),
           "xi_mittel": {"stationaer": float(np.mean(xa)), "lage_trend": float(np.mean(xb)), "kontrolle": float(np.mean(xk))}}
    return out


# ------------------------------------------------------------------------------------------------------------------------------------------
ART_P = (0.35, 0.30, 0.25, 0.10)
U_SCALE = (1.0, 0.3, 0.3, 1.0)
S_BASE = (0.55, 0.25, 0.35, 0.90)
UK = (-0.42714751093638936, 0.3791185374387331, -0.2611985213877257, -0.07775197048031927, -0.02260299210315629, -0.2222653956256827,
      -0.410337810534883, 0.19466784065791196, 0.10831743391646849, -0.5858589189036569, 0.7042228963136555, 0.2905490717255771)


def weekly_components(we_only):
    """Komponenten (Gewicht, μ, s) der Verweilzeit-Mischung von Vehikel B bei festem Wochentag-Typ (we_only = 0: nur Werktage, 1: nur Wochenende), aus der Definition der Linie
    (Art × Kunde × Zoll × Vorlauf per Gauß-Legendre mit 48 Knoten; Medianformeln wie in der Vorlage `verweilzeit-demo`)."""
    xg, wg = np.polynomial.legendre.leggauss(48)
    comps = []
    for art in range(4):
        for kunde in range(12):
            for zoll, pz in (((0, 0.75), (1, 0.25)) if art == 0 else ((0, 1.0), (1, 0.0))):
                if art in (1, 2):
                    lo, hi = (1.0, 10.0) if art == 1 else (0.5, 6.0)
                    knoten = [(lo + (hi - lo) * (x + 1) / 2, w / 2) for x, w in zip(xg, wg)]
                else:
                    knoten = [(0.0, 1.0)]
                for vor, wv in knoten:
                    if art == 0:
                        med = 3.5 * math.exp(0.7 * zoll + 0.30 * we_only)
                    elif art == 1:
                        med = 0.6 + 0.85 * vor
                    elif art == 2:
                        med = 0.3 + 0.8 * vor
                    else:
                        med = 7.0 * math.exp(0.30 * we_only)
                    gew = ART_P[art] / 12 * pz * wv
                    if gew > 0:
                        comps.append((gew, math.log(med) + UK[kunde] * U_SCALE[art], S_BASE[art]))
    return comps


def mixture_level(comps, p):
    """x mit Überlebensfunktion p (Bisektion in ln x; Überlebensfunktion einer lognormalen Mischung mit math.erfc)."""
    def sf(lx):
        return sum(w * 0.5 * math.erfc((lx - mu) / (s * math.sqrt(2.0))) for w, mu, s in comps)
    lo, hi = -3.0, 14.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if sf(mid) > p:
            lo = mid
        else:
            hi = mid
    return math.exp(0.5 * (lo + hi))


def t_weekly():
    wd, we = weekly_components(0), weekly_components(1)
    lw, le = mixture_level(wd, 1e-5), mixture_level(we, 1e-5)
    return {"p": 1e-5, "werktage": lw, "wochenende": le, "faktor": le / lw, "summe_gewichte_werktage": float(sum(c[0] for c in wd)), "summe_gewichte_wochenende": float(sum(c[0] for c in we))}


# ------------------------------------------------------------------------------------------------------------------------------------------
JOBS = (
    [("typ1", (m,)) for m in (10, 20, 30, 50, 100)]
    + [("power", (b, m)) for b in (0.01, 0.02, 0.04) for m in (20, 50, 100)]
    + [("years80", (b,)) for b in (0.005, 0.01, 0.02, 0.04)]
    + [("blocks", (b,)) for b in (0.0, 0.01, 0.02, 0.04)]
    + [("rmse", (m, b)) for m in (20, 50, 100) for b in (0.0, 0.01, 0.02, 0.04)]
    + [("bias", ())]
    + [("coverage", mb) for mb in ((30, 0.0), (50, 0.02), (100, 0.02), (100, 0.04))]
    + [("vehicle_c1", (r,)) for r in (0.70, 0.75, 0.80, 0.85, 0.90)]
    + [("vehicle_c2", (r,)) for r in (0.80, 0.85, 0.90)]
    + [("weekly", ())]
)
FUNCS = {"typ1": t_typ1, "power": t_power, "years80": t_years80, "blocks": t_blocks, "rmse": t_rmse, "bias": t_bias, "coverage": t_coverage,
         "vehicle_c1": t_vehicle_c1, "vehicle_c2": t_vehicle_c2, "weekly": t_weekly}
ORDER_HEAVY = {"vehicle_c2": 0, "rmse": 1, "bias": 2, "coverage": 3, "vehicle_c1": 4, "power": 5, "typ1": 6, "years80": 7, "blocks": 8, "weekly": 9}


def run_job(job):
    kind, args = job
    t0 = time.perf_counter()
    res = FUNCS[kind](*args)
    return kind, args, res, time.perf_counter() - t0


QUELLEN = {
    "typ1": "Fehlerquote des Likelihood-Quotienten-Tests (Gumbel mit gegen ohne Lage-Trend, ξ = 0 bekannt) bei β = 0; je m 4 000 Läufe; Schwelle χ²₁ 95 % = 3,8415",
    "power": "Trennschärfe desselben Tests, je Zelle 4 000 Läufe; `formel` = Φ(√λ − 1,96) + Φ(−√λ − 1,96), λ = β² m (m² − 1)/12",
    "years80": "kleinstes m mit Formel-Trennschärfe ≥ 0,8, dort 2 000 Läufe Simulation",
    "blocks": "400 000 Blöcke zu m Jahren, Jahr für Jahr gezogen (Lage β·t); Zählung der Blöcke über dem Design Life Level (q = 5 %) und über dem heutigen 100-Jahres-Pegel; exakte Werte daneben",
    "rmse": "je Zelle 400 Läufe; Fehler = Pegel-Dach minus wahrer Pegel des letzten Jahres in σ; a stationäre GEV, b Gumbel mit Lage-Trend (ξ = 0 bekannt), c GEV mit Lage-Trend; gepaarter Bootstrap-Standardfehler (2 000 Wiederholungen)",
    "bias": "stationäre GEV, 3 000 Läufe, m = 50, β = 0,02",
    "coverage": "Anteil der Läufe, in denen das 95-%-Delta-Intervall (numerische Hesse-Matrix) der stationären GEV den wahren Pegel deckt; kein Intervall zählt als nicht gedeckt",
    "vehicle_c1": "Vehikel A (M/M/1, Tag = 256 Kunden, stationärer Start), 400 000 Tage je Last; Gumbel-ML der Tagesmaxima gegen die Gumbel-Näherung der Rechnung",
    "vehicle_c2": "Vehikel A bei wachsender Last (linear von ρ = 0,70 auf ρ_bis über 100 Tage), 300 Läufe; wahr = 0,99-Quantil des Tagesmaximums bei ρ_bis aus 400 000 Tagen; Modelle: stationäre GEV, GEV mit Lage-Trend, Gumbel mit Trend in Lage und ln σ",
    "weekly": "Vehikel B: Pegel je Container zu p = 1e-5 nur Werktage und nur Wochenende (exakt aus der Definition der Linie, ohne Zufall)",
}


def assemble(results):
    out = {"_meta": {"beschreibung": "Eingefrorene Rechnungen der Demo trend-spitzen-demo; Erzeuger tools/tsp_make_grid.py; Zufall: SplitMix64 (tsp_vecrng, bitgleich zur Vorlage), Umrechnung numpy.log",
                     "stand": STAND, "zufalls_seed_basis": BASE, "numpy": np.__version__, "python": sys.version.split()[0]}}
    for kind in FUNCS:
        rows = [(args, res) for k, args, res, _t in results if k == kind]
        data = [res for _a, res in sorted(rows, key=lambda r: json.dumps(r[0]))] if len(rows) > 1 or kind not in ("bias", "weekly") else rows[0][1]
        out[kind] = {"quelle": QUELLEN[kind], "stand": STAND, "daten": data}
    return out


def compute():
    jobs = sorted(JOBS, key=lambda j: ORDER_HEAVY[j[0]])
    results = []
    t0 = time.perf_counter()
    with Pool(PROCESSES) as pool:
        for kind, args, res, secs in pool.imap_unordered(run_job, jobs):
            results.append((kind, args, res, secs))
            print(f"  {kind} {args}: {secs:6.1f} s (gesamt {time.perf_counter() - t0:6.1f} s)", flush=True)
    return assemble(results)


def check():
    """Vier kleine Zellen neu rechnen und als Band mit der Datei vergleichen."""
    d = json.loads(FILE.read_text(encoding="utf-8"))
    ok = True
    r = t_typ1(10)
    s = next(x for x in d["typ1"]["daten"] if x["m"] == 10)
    good = abs(r["hits"] - s["hits"]) <= 3
    print(f"typ1 m=10: neu {r['hits']} gegen Datei {s['hits']} von {s['R']}: {'ok' if good else 'ABWEICHUNG'}")
    ok &= good
    r = t_rmse(20, 0.0)
    s = next(x for x in d["rmse"]["daten"] if x["m"] == 20 and x["beta"] == 0.0)
    good = all(abs(r[k]["med_abs"] - s[k]["med_abs"]) <= 1e-3 * max(1.0, abs(s[k]["med_abs"])) and abs(r[k]["bias"] - s[k]["bias"]) <= 1e-2 * max(1.0, abs(s[k]["bias"])) for k in ("a", "b", "c"))
    print(f"rmse m=20, beta=0: Median des Betrags a/b/c neu {r['a']['med_abs']:.4f}/{r['b']['med_abs']:.4f}/{r['c']['med_abs']:.4f}, Datei {s['a']['med_abs']:.4f}/{s['b']['med_abs']:.4f}/{s['c']['med_abs']:.4f}: {'ok' if good else 'ABWEICHUNG'}")
    ok &= good
    vs = V.VecStream(seed(99))
    mx = big_days(0.80, 40_000, vs)
    f = M.fit_gumbel(mx, False)
    s = next(x for x in d["vehicle_c1"]["daten"] if x["rho"] == 0.80)
    good = abs(f.sigma - s["sigma_dach"]) < 0.1 and abs(f.a - s["mu_dach"]) < 0.15
    print(f"Vehikel A rho=0,80 (40 000 Tage, anderer Strom): sigma {f.sigma:.3f} gegen {s['sigma_dach']:.3f}, mu {f.a:.3f} gegen {s['mu_dach']:.3f}: {'ok' if good else 'ABWEICHUNG'}")
    ok &= good
    r = t_weekly()
    s = d["weekly"]["daten"]
    good = abs(r["werktage"] - s["werktage"]) < 1e-6 * s["werktage"] and abs(r["wochenende"] - s["wochenende"]) < 1e-6 * s["wochenende"]
    print(f"weekly: {r['werktage']:.4f} / {r['wochenende']:.4f} gegen Datei {s['werktage']:.4f} / {s['wochenende']:.4f}: {'ok' if good else 'ABWEICHUNG'}")
    ok &= good
    return bool(ok)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="vier kleine Zellen neu rechnen und als Band vergleichen")
    args = ap.parse_args()
    if args.check:
        sys.exit(0 if check() else 1)
    t0 = time.perf_counter()
    data = compute()
    FILE.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"geschrieben: {FILE} ({time.perf_counter() - t0:.0f} s)")
