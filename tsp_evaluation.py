"""Auswertung: live gerechnete Größen, Zusammenfassungen und die Verdikte der Experimente als reine Funktionen `verdict_*(…) -> (Art, Text)`.

Alle exakten Größen (Mischung, Risiko, Design Life Level, Wiederkehrperiode, Trennschärfe-Formel) kommen aus `tsp_models`; die eingefrorenen Zufallsmessungen aus `tsp_reference`.
Zähleinheiten stehen bei jeder Zahl: Jahre, Läufe (je Lauf eine Reihe), Blöcke (Nutzungsdauern), Tage (Vehikel A)."""

import math
from dataclasses import dataclass, field

import numpy as np

import tsp_constants as C
import tsp_models as M
import tsp_numeric as N
import tsp_reference as REF
import tsp_rng as R
import tsp_scenario as S
from tsp_texts import de, pct, pint


@dataclass(frozen=True)
class Settings:
    m: int = C.DEFAULTS["m_sl"]
    beta: float = C.DEFAULTS["beta_sl"]
    seed: int = C.DEFAULT_SEED
    nutz: int = C.DEFAULTS["nutz_sl"]
    q: float = C.DEFAULTS["q_sl"]
    runs: int = C.DEFAULT_RUNS
    last_end: float = C.DEFAULTS["last_sel"]
    modell: str = C.DEFAULTS["modell_sel"]


def series_seed(settings: Settings) -> int:
    return S.derive_seed(settings.seed, 1)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Experimente 1 und 3: eine gezogene Reihe
# ---------------------------------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Truths:
    first: float
    mixture: float
    last: float


def truths(beta: float, m: int) -> Truths:
    """Wahre 100-Jahres-Pegel (exakt): erstes Jahr, Mischung aller m Jahre, letztes Jahr."""
    return Truths(M.level_year(beta, 0), M.mixture_quantile(beta, m), M.level_year(beta, m - 1))


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    x: np.ndarray = field(repr=False)
    lr: float
    detected: bool
    stat: M.GumbelFit
    trend: M.GumbelFit
    records: int
    records_expected_trend: float
    records_expected_no_trend: float
    truth: Truths
    gev_stat: M.GevFit | None
    gev_error: str
    interval: tuple | None


def analyse(settings: Settings) -> Analysis:
    x = S.gumbel_series(settings.m, settings.beta, series_seed(settings))
    lr, f0, f1 = M.likelihood_ratio(x)
    gev, err, ci = None, "", None
    try:
        gev = M.fit_gev(x, False)
        ci = M.delta_interval(gev, x)
    except M.FitError as e:
        err = str(e)
    return Analysis(settings, x, lr, lr > N.CHI2_95_DF1, f0, f1, M.count_records(x), M.expected_records(settings.beta, settings.m), M.harmonic(settings.m),
                    truths(settings.beta, settings.m), gev, err, ci)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Experiment 2: exakte Größen und die Simulation von Blöcken
# ---------------------------------------------------------------------------------------------------------------------------------------------
def return_period_rows(beta: float) -> list[tuple[int, float]]:
    return [(t, M.return_period_after(beta, t)) for t in C.LEVEL_YEARS]


@dataclass(frozen=True)
class RiskTable:
    risk: float             # Jahre 0 … m−1 (das heutige Jahr eingeschlossen)
    risk_ahead: float       # Jahre 1 … m (m Jahre ab morgen)
    risk_stationary: float
    dll: float
    dll_stationary: float
    survival_check: float   # Wahrscheinlichkeit des Maximums unter dem DLL (soll 1 − q sein)


def risk_table(beta: float, nutz: int, q: float) -> RiskTable:
    dll = M.design_life_level(beta, nutz, q)
    s_stat = M.risk_in_years(0.0, nutz)
    return RiskTable(M.risk_in_years(beta, nutz), M.risk_in_years(beta, nutz, start=1), s_stat, dll, M.design_life_level(0.0, nutz, q),
                     float(math.exp(-math.exp(-dll + M.log_sum_exp_years(beta, nutz)))))


def live_blocks(settings: Settings) -> dict:
    """Live-Simulation: N_BLOCKS_LIVE Nutzungsdauern zu `nutz` Jahren, Jahr für Jahr gezogen; Anteil über dem heutigen 100-Jahres-Pegel und über dem Design Life Level."""
    mx = S.block_maxima(C.N_BLOCKS_LIVE, settings.nutz, settings.beta, S.derive_seed(settings.seed, 2))
    t = risk_table(settings.beta, settings.nutz, settings.q)
    n = mx.size
    p_today = float(np.mean(mx > M.Y100))
    p_dll = float(np.mean(mx > t.dll))
    return {"n": n, "p_heute": p_today, "p_heute_exakt": t.risk, "se_heute": math.sqrt(t.risk * (1 - t.risk) / n),
            "p_dll": p_dll, "p_dll_exakt": settings.q, "se_dll": math.sqrt(settings.q * (1 - settings.q) / n)}


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Experiment 4: eigene Läufe
# ---------------------------------------------------------------------------------------------------------------------------------------------
def summarize_errors(e) -> dict:
    e = np.asarray(e, float)
    return {"bias": float(np.mean(e)), "sd": float(np.std(e, ddof=1)) if e.size > 1 else float("nan"), "rmse": float(math.sqrt(np.mean(e * e))), "med_abs": float(np.median(np.abs(e)))}


def live_runs(settings: Settings) -> dict:
    """Eigene Läufe in der gewählten Zelle (m, β): je Lauf eine Reihe, drei Anpassungen (a stationäre GEV, b Gumbel mit Trend, c GEV mit Trend), Fehler am letzten Jahr in σ."""
    rng = R.SplitMix64(S.derive_seed(settings.seed, 3))
    m, beta = settings.m, settings.beta
    truth = M.level_year(beta, m - 1)
    ea, eb, ec, failed = [], [], [], 0
    for _ in range(settings.runs):
        x = np.array([beta * t + S.gumbel(rng) for t in range(m)])
        try:
            fa, fb, fc = M.fit_gev(x, False), M.fit_gumbel(x, True), M.fit_gev(x, True)
        except M.FitError:
            failed += 1
            continue
        ea.append(fa.level() - truth)
        eb.append(fb.level() - truth)
        ec.append(fc.level() - truth)
    ea, eb, ec = np.array(ea), np.array(eb), np.array(ec)
    n = ec.size
    if n == 0:
        return {"n": 0, "ausgefallen": failed}
    return {"n": int(n), "ausgefallen": failed, "a": summarize_errors(ea), "b": summarize_errors(eb), "c": summarize_errors(ec),
            "b_besser": float(np.mean(np.abs(eb) < np.abs(ea))), "c_besser": float(np.mean(np.abs(ec) < np.abs(ea)))}


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Verdikte (reine Funktionen)
# ---------------------------------------------------------------------------------------------------------------------------------------------
VERDICT_WORDS = {"c besser": "Trendmodell klar besser", "c schlechter": "Trendmodell klar schlechter", "Rauschen": "im Rauschen"}


def power_summary(beta: float, m: int) -> dict:
    """Formel und, falls vorhanden, eingefrorene Simulation der Trennschärfe in der Zelle (β, m)."""
    out = {"formel": M.power(beta, m), "lambda": M.noncentrality(beta, m), "gesamtanstieg": beta * (m - 1)}
    cell = REF.power_cell(beta, m) if beta > 0 else None
    if cell:
        n = cell["R"] - cell["ausgefallen"]
        out.update({"sim": cell["hits"] / n, "R": n, "wilson": N.wilson(cell["hits"], n), "se": math.sqrt(cell["formel"] * (1 - cell["formel"]) / n)})
    return out


def verdict_trend(m: int, beta: float) -> tuple[str, str]:
    """Verdikt Experiment 1: Trennschärfe (Formel, Simulation), Jahre bis 80 %, Fehlerquote erster Art."""
    t1 = REF.typ1(m)
    t1_text = (f" Ohne Trend (β = 0) meldet derselbe Test bei {m} Jahren in {pct(t1['hits'] / (t1['R'] - t1['ausgefallen']), 1)} der Läufe einen Trend "
               f"(Soll 5 %, R = {pint(t1['R'])} Läufe, Standardfehler {de(100 * math.sqrt(0.05 * 0.95 / t1['R']), 2)} Punkte).") if t1 else ""
    if beta == 0:
        return "info", ("Ohne Trend (β = 0) gibt es nichts zu erkennen; jede Meldung „Trend“ ist ein Fehlalarm."
                        + (t1_text if t1 else " Für diese Länge gibt es keine eingefrorene Fehlerquote (nur 10, 20, 30, 50 und 100 Jahre)."))
    p = power_summary(beta, m)
    m80 = M.years_for_power(beta, 0.8)
    sim = (f"; Simulation {pct(p['sim'], 1)} bei R = {pint(p['R'])} Läufe, Wilson-Band {pct(p['wilson'][0], 1)} bis {pct(p['wilson'][1], 1)}") if "sim" in p else "; für diese Zelle gibt es keine eingefrorene Simulation"
    kind = "success" if p["formel"] >= 0.8 else "warning"
    return kind, (f"β = {de(beta, 2)} σ je Jahr, {m} Jahre (Gesamtanstieg {de(p['gesamtanstieg'], 2)} σ): der Test (Gumbel, ξ = 0 bekannt) erkennt den Trend in rund {pct(p['formel'])} der Läufe "
                  f"(Formel {pct(p['formel'], 1)}{sim}). Nach der asymptotischen Formel braucht er für 80 % bei diesem β {m80} Jahre. Nicht erkannt heißt nicht ausgeschlossen: unter 80 % Trennschärfe bleibt der Trend möglich.{t1_text}")


def verdict_risk(beta: float, nutz: int, q: float) -> tuple[str, str]:
    """Verdikt Experiment 2 (alle Zahlen exakt)."""
    t = risk_table(beta, nutz, q)
    rp = {tt: M.return_period_after(beta, tt) for tt in (50, 100)}
    if beta == 0:
        return "info", (f"Ohne Trend ist der Pegel von heute der Pegel jedes Jahres: Risiko in {nutz} Jahren {pct(t.risk, 1)} (die Jahre 0 bis {nutz - 1}), "
                        f"Wiederkehrperiode in jedem Jahr 100 Jahre, Design Life Level (q = {pct(q)}) {de(t.dll, 2)} σ, gleich dem stationären.")
    return "warning", (f"β = {de(beta, 2)} σ je Jahr: der heutige 100-Jahres-Pegel wird in {nutz} Jahren (die Jahre 0 bis {nutz - 1}, das heutige eingeschlossen) mit {pct(t.risk, 1)} überschritten, ohne Trend mit {pct(t.risk_stationary, 1)}; "
                       f"zählt man {nutz} Jahre ab morgen (die Jahre 1 bis {nutz}), sind es {pct(t.risk_ahead, 1)}. Die Wiederkehrperiode (Kehrwert der Überschreitungswahrscheinlichkeit des Jahres) beträgt nach 50 Jahren {de(rp[50], 0)} Jahre, nach 100 Jahren {de(rp[100], 0)}; als mittlere Wartezeit gilt sie nur, bliebe diese Wahrscheinlichkeit stehen. "
                       f"Der Design Life Level (Risiko {pct(q)} in {nutz} Jahren) liegt bei {de(t.dll, 2)} σ gegen {de(t.dll_stationary, 2)} σ ohne Trend (+{de(t.dll - t.dll_stationary, 2)}).")


def verdict_stationary(m: int, beta: float) -> tuple[str, str]:
    """Verdikt Experiment 3: Modellfehler (exakt) gegen Schätzfehler (eingefroren)."""
    tr = truths(beta, m)
    model_err = tr.mixture - tr.last
    cell = REF.rmse_cell(m, beta)
    cov = REF.coverage_cell(m, beta)
    bias_c = REF.bias_cell()
    if beta == 0:
        txt = f"Ohne Trend gibt es keinen Modellfehler: die Mischung ist der Pegel jedes Jahres ({de(tr.mixture, 2)} σ). "
    else:
        txt = (f"m = {m}, β = {de(beta, 2)}: die Mischung liegt {de(-model_err, 2)} σ unter dem Pegel des letzten Jahres (exakt, ohne Rauschen). ")
    if beta > 0 and m == bias_c["m"] and abs(beta - bias_c["beta"]) < 1e-9:
        txt += (f"Die stationäre GEV unterschätzt ihn im Mittel um {de(-bias_c['bias'], 2)} σ (R = {pint(bias_c['R'])} Läufe, Standardfehler {de(bias_c['se_bias'], 2)}); davon sind {de(-model_err, 2)} σ Modellfehler, "
                f"der Rest ({de(-bias_c['bias'] + model_err, 2)} σ) ist Anpassungsfehler, größtenteils ein systematischer Abstand der GEV-Anpassung von der Mischung (Großstichproben-Rechnung mit 400 000 Ziehungen, etwa 0,05 bis 0,08 σ), nicht nur Rauschen; das ξ der Anpassung wird dabei im Mittel {de(bias_c['xi_mittel'], 2)}. ")
    elif cell:
        txt += f"Die stationäre GEV liegt im Mittel um {de(cell['a']['bias'], 2)} σ neben dem Pegel des letzten Jahres (R = {cell['R']} Läufe, Standardfehler {de(cell['a']['sd'] / math.sqrt(cell['R']), 2)}). "
    cov_src = cov or (cell and {"letztes": cell["abdeckung"]["letztes"], "R": cell["R"]})
    if cov_src:
        txt += f"Das 95-%-Delta-Intervall deckt den Pegel des letzten Jahres in {pct(cov_src['letztes'], 1)} der Läufe (R = {pint(cov_src['R'])})."
    return "warning" if (beta > 0 and -model_err > 0.2) else "info", txt.strip()


def verdict_trendmodel(m: int, beta: float, live: dict | None = None) -> tuple[str, str]:
    """Verdikt Experiment 4: Trendmodell bei gleichem freiem ξ gegen die stationäre Anpassung."""
    cnt = REF.rmse_counts()
    n = sum(cnt.values())
    txt = (f"Bei gleichem freiem ξ ist das Trendmodell in {cnt['c besser']} von {n} Zellen klar besser, in {cnt['c schlechter']} klar schlechter und in {cnt['Rauschen']} im Rauschen (gepaarter Standardfehler, Betrag über 2). ")
    cell = REF.rmse_cell(m, beta)
    if cell:
        d = cell["diff_c_a"]
        txt += (f"Diese Zelle (m = {m}, β = {de(beta, 2)}): RMSE stationär {de(cell['a']['rmse'], 2)} σ, mit Trend {de(cell['c']['rmse'], 2)} σ (Differenz {de(d['diff'], 2)} bei Standardfehler {de(d['se'], 2)}: {VERDICT_WORDS[d['urteil']]}); "
                f"Median des Betrags {de(cell['a']['med_abs'], 2)} gegen {de(cell['c']['med_abs'], 2)} σ (R = {cell['R']} Läufe). ")
        if m == 20:
            txt += "Bei 20 Jahren sind die RMSE-Zahlen von wenigen Läufen mit ξ̂ am Rand bestimmt; belastbar ist der Median des Betrags. "
    else:
        txt += "Für diese Länge gibt es keine eingefrorene Zelle (nur 20, 50 und 100 Jahre). "
    txt += "Die Zählung hängt vom Zufalls-Seed ab: Die Vormessung mit anderem Zufallsstrom und eine Neumessung mit anderem Zufalls-Seed gaben 4 klar besser, 5 klar schlechter, 3 im Rauschen. "
    if live and live.get("n"):
        txt += f"Eigene {live['n']} Läufe: RMSE stationär {de(live['a']['rmse'], 2)}, mit Trend {de(live['c']['rmse'], 2)} σ (Zufallsschwankung groß, kein Befund)."
    return "info", txt.strip()


def verdict_vehicle(rho_end: float, modell: str) -> tuple[str, str]:
    """Verdikt Experiment 5 (eingefrorene Läufe)."""
    c1_lo, c1_hi = REF.vehicle_c1(C.LOAD_START), REF.vehicle_c1(0.90)
    c2 = REF.vehicle_c2(rho_end)
    key = {"stationär": "stationaer", "Lage-Trend": "lage_trend", "Lage- und Skalen-Trend": "lage_skala_trend"}[modell]
    r = c2[key]
    ctrl = c2["kontrolle_ohne_trend"]
    return "warning", (f"ρ von {de(C.LOAD_START, 2)} auf {de(rho_end, 2)} wachsend: Pegel des letzten Tages (wahr {de(c2['wahr'], 1)} Abfertigungsdauern, aus {pint(c2['wahr_tage'])} Tagen); das Modell „{modell}“ "
                       f"trifft im Median das {de(r['median'], 2)}-Fache (Q10 bis Q90: {de(r['q10'], 2)} bis {de(r['q90'], 2)}) und liegt in {pct(r['anteil_unter_1'])} der Läufe darunter (R = {c2['R']} Läufe zu je {c2['tage_je_lauf']} Tagen). "
                       f"Ohne Last-Anstieg (alle Tage bei ρ = {de(rho_end, 2)}) trifft die stationäre GEV den Pegel im Median mit dem {de(ctrl['median'], 2)}-Fachen. "
                       f"Skala und Lage der Tagesmaxima wachsen beide (ρ = 0,70 auf 0,90: Skala ×{de(c1_hi['sigma_dach'] / c1_lo['sigma_dach'], 2)}, Lage ×{de(c1_hi['mu_dach'] / c1_lo['mu_dach'], 2)}, Gumbel-ML, {pint(c1_lo['tage'])} Tage je Last).")
