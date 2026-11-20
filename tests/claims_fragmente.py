"""Bausteine der Zahl-Test-Tabelle: jede Zahl der README als Text, aus der Rechnung (exakt) oder der eingefrorenen Messung gebildet; `test_claims.py` prüft, dass die README genau diese Texte enthält.

Jeder Eintrag: (Kennung, Text, Art, Beschreibung). Art: „exakt“ (Formel, Bruch, Aufzählung; im Test nachgerechnet) oder „Messung“ (eingefrorener Zufallsanteil mit Zähleinheit; im Test als Band gegen die exakte Größe geprüft)."""

import math
from fractions import Fraction as Fr

import tsp_constants as C
import tsp_evaluation as EV
import tsp_models as M
import tsp_reference as REF
import tsp_vehicle as VH
from tsp_texts import de, pct, pint

LN2 = math.log(2.0)


def table_error_rows():
    """Zeilen der Tabelle „Mischung minus Pegel des letzten Jahres“ (exakt)."""
    rows = []
    for m in (30, 50, 100):
        cells = [de(M.mixture_quantile(b, m) - M.level_year(b, m - 1), 2) for b in (0.01, 0.02, 0.04)]
        rows.append((m, cells))
    return rows


def mittlere_wartezeit(beta, t):
    """Mittlere Wartezeit in Jahren (das Jahr t mitgezählt) auf die erste Überschreitung des heutigen 100-Jahres-Pegels, wenn der Trend ab Jahr t weiterläuft:
    E[W] = Σ_{k≥0} P(W > k), P(W > k) = Π_{j<k} (1 − p_{t+j}); eigene Rechnung nur mit math, unabhängig von `tsp_models`."""
    y100 = -math.log(-math.log(0.99))
    ew, surv = 0.0, 1.0
    for j in range(5000):
        ew += surv
        surv *= math.exp(-math.exp(-(y100 - beta * (t + j))))
        if surv < 1e-16:
            break
    return ew


def fragmente():
    f = []
    add = lambda kid, text, art, beschr: f.append((kid, text, art, beschr))
    # --- exakt -----------------------------------------------------------------------------------------------------------------------------
    add("E1", de(M.Y100, 6), "exakt", "Grundhöhe y₁₀₀ = −ln(−ln 0,99)")
    t = EV.truths(0.02, 50)
    add("E2", f"{de(t.first, 3)} / {de(t.mixture, 3)} / {de(t.last, 3)}", "exakt", "Pegel erstes Jahr / Mischung / letztes Jahr bei β = 0,02, m = 50")
    add("E3", de(t.last - t.mixture, 2) + " σ", "exakt", "Modellfehler der Mischung bei β = 0,02, m = 50")
    rt = EV.risk_table(0.02, 50, 0.05)
    add("E4", f"{pct(rt.risk, 1)}", "exakt", "Risiko in 50 Jahren (Jahre 0 bis 49), β = 0,02")
    add("E5", f"{pct(rt.risk_stationary, 1)}", "exakt", "Risiko in 50 Jahren ohne Trend")
    add("E6", f"{pct(rt.risk_ahead, 1)}", "exakt", "Risiko in 50 Jahren ab morgen (Jahre 1 bis 50), β = 0,02")
    add("E7", " / ".join(de(M.return_period_after(0.02, tt), 1) for tt in C.LEVEL_YEARS), "exakt", "Wiederkehrperiode des heutigen Pegels nach 0 / 10 / 25 / 50 / 75 / 100 Jahren, β = 0,02")
    add("E8", f"{de(rt.dll, 2)} σ gegen {de(rt.dll_stationary, 2)} σ", "exakt", "Design Life Level (q = 5 %, 50 Jahre), β = 0,02 gegen ohne Trend")
    add("E9", " / ".join(pct(M.risk_in_years(b, 50), 2) for b in C.BETA_OPTIONS), "exakt", "Risiko in 50 Jahren für β = 0 / 0,01 / 0,02 / 0,04")
    add("E10", " / ".join(de(M.design_life_level(b, 50, 0.05), 3) for b in C.BETA_OPTIONS[1:]), "exakt", "Design Life Level (q = 5 %, 50 Jahre) für β = 0,01 / 0,02 / 0,04")
    add("E11", de(M.noncentrality(0.02, 50), 3), "exakt", "λ bei β = 0,02, m = 50 (= 833/200)")
    add("E12", pct(M.power(0.02, 50), 1), "exakt", "Formel-Trennschärfe bei β = 0,02, m = 50")
    add("E13", pct(M.power(1 / 49, 50), 1), "exakt", "Formel-Trennschärfe bei Gesamtanstieg 1,0 σ (β = 1/49), m = 50")
    add("E14", " / ".join(str(M.years_for_power(b)) for b in C.STEPS_YEARS80), "exakt", "Jahre bis 80 % bei β = 0,005 / 0,01 / 0,02 / 0,04")
    add("E15", f"{Fr(64, 315)} {Fr(1, 24)} {Fr(8, 21)} {Fr(97, 35)} {Fr(25, 12)}", "exakt", "Handbeispiele H1 und H2 (Brüche)")
    add("E16", de(M.mixture_quantile(LN2, 2), 4), "exakt", "H3: Mischung zweier Jahre bei e^β = 2")
    add("E17", f"{pct(M.risk_in_years(LN2, 4), 2)} gegen {pct(M.risk_in_years(0.0, 4), 2)}", "exakt", "H4: Risiko in vier Jahren, Spielzeugtrend gegen ohne Trend")
    add("E18", f"{de(1.0, 0)}; {de(M.gumbel_information()['mu_sigma'], 4)}; {de(M.gumbel_information()['sigma_sigma'], 4)}", "exakt", "Fisher-Information der Gumbel-Verteilung (μ,μ; μ,σ; σ,σ)")
    add("E19", " / ".join(de(VH.gumbel_approximation(r)[0], 2) for r in (0.70, 0.75, 0.80, 0.85, 0.90)), "exakt", "Lage der Gumbel-Näherung des Tagesmaximums, ρ = 0,70 bis 0,90")
    add("E20", " / ".join(de(VH.approximate_level(r), 1) for r in (0.70, 0.75, 0.80, 0.85, 0.90)), "exakt", "100-Tage-Pegel der Gumbel-Näherung, ρ = 0,70 bis 0,90")
    add("E21", f"{de(mittlere_wartezeit(0.02, 50), 1)} Jahre", "exakt", "mittlere Wartezeit auf eine Überschreitung des heutigen Pegels ab Jahr 50, wenn der Trend weiterläuft (β = 0,02, Jahr 50 mitgezählt); der Kehrwert 37,1 gilt nur bei gleichbleibender Wahrscheinlichkeit")
    # --- Messung: Erkennbarkeit ------------------------------------------------------------------------------------------------------------
    add("M1", " / ".join(pct(REF.typ1(m)["hits"] / REF.typ1(m)["R"], 1) for m in C.M_TYP1), "Messung", "Fehlerquote erster Art bei m = 10 / 20 / 30 / 50 / 100 Jahren (R = 4 000 Läufe)")
    pc = REF.power_cell(0.02, 50)
    add("M2", pct(pc["hits"] / pc["R"], 1), "Messung", "Trennschärfe-Simulation bei β = 0,02, m = 50 (R = 4 000 Läufe)")
    add("M3", " / ".join(pct(REF.years80(b)["hits"] / REF.years80(b)["R"], 1) for b in C.STEPS_YEARS80), "Messung", "Simulation beim kleinsten m mit Formel ≥ 80 % (R = 2 000 Läufe)")
    # --- Messung: Pegel von heute -------------------------------------------------------------------------------------------------------------
    b = REF.bias_cell()
    add("M4", f"{de(b['bias'], 2)} σ", "Messung", "Verzerrung der stationären GEV, m = 50, β = 0,02 (R = 3 000 Läufe)")
    add("M5", f"Standardfehler {de(b['se_bias'], 2)}", "Messung", "Standardfehler dieser Verzerrung")
    add("M6", " / ".join(pct(REF.coverage_cell(m_, b_)["letztes"], 1) for m_, b_ in ((30, 0.0), (50, 0.02), (100, 0.02), (100, 0.04))), "Messung", "Abdeckung des Delta-Intervalls (letztes Jahr), vier Zellen (R = 1 000 Läufe)")
    cnt = REF.rmse_counts()
    add("M7", f"{cnt['c besser']} klar besser, {cnt['c schlechter']} klar schlechter, {cnt['Rauschen']} im Rauschen", "Messung", "Trendmodell gegen stationär bei gleichem freiem ξ, 12 Zellen (R = 400 Läufe)")
    c = REF.rmse_cell(100, 0.02)
    add("M8", f"{de(c['c']['rmse'], 2)} gegen {de(c['a']['rmse'], 2)} σ", "Messung", "RMSE mit Trend gegen stationär, m = 100, β = 0,02")
    c1 = {r["rho"]: r for r in REF.vehicle_c1()}
    add("M9", f"×{de(c1[0.90]['sigma_dach'] / c1[0.70]['sigma_dach'], 2)} und ×{de(c1[0.90]['mu_dach'] / c1[0.70]['mu_dach'], 2)}", "Messung", "Vehikel A: Skala und Lage der Tagesmaxima, ρ = 0,70 auf 0,90 (400 000 Tage je Last)")
    c2 = REF.vehicle_c2(0.90)
    add("M10", " / ".join(de(c2[k]["median"], 2) for k in ("stationaer", "lage_trend", "lage_skala_trend")) + f"; {de(c2['kontrolle_ohne_trend']['median'], 2)}", "Messung",
        "Vehikel A: Median Pegel-Dach/wahr der drei Modelle und der Kontrolle (R = 300 Läufe)")
    wk = REF.weekly()
    add("M11", f"{de(wk['werktage'], 2)} / {de(wk['wochenende'], 2)}", "Messung", "Vehikel B: Pegel je Container nur Werktage / nur Wochenende (exakt)")
    add("M12", de(wk["faktor"], 2), "Messung", "Vehikel B: Faktor Wochenende gegen Werktage")
    add("M13", pint(REF.blocks(0.0)["blocks"]), "Messung", "Blöcke der eingefrorenen Simulation (Experiment 2)")
    return f
