"""Eingefrorene Tabellen (`tsp_grid.json`): Aufbau, Quelle und Stand je Teil, Zählungen über das ganze Raster, Bänder gegen die exakten Größen und Neu-Rechnen einzelner Zellen als Band.

Zufallsanteile stehen nie als Gleichheit im Test: Bänder in Standardfehlern (BAND_SE), die Formel der Trennschärfe ist asymptotisch (Band ± 3 Standardfehler + 0,03).
Neu-Rechnen: eine Zelle mit dem Strom der Vorlage und `exact_log` (plattformfest) und eine mit dem vektorisierten Strom des Werkzeugs."""

import math
import sys
from pathlib import Path

import pytest

import tsp_constants as C
import tsp_models as M
import tsp_numeric as N
import tsp_reference as REF
import tsp_scenario as S

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

TEILE = ("typ1", "power", "years80", "blocks", "rmse", "bias", "coverage", "vehicle_c1", "vehicle_c2", "weekly")


def test_jeder_teil_hat_quelle_und_stand_und_die_meta_nennt_den_seed():
    g = REF.grid()
    assert set(TEILE) <= set(g)
    for t in TEILE:
        assert g[t]["quelle"] and g[t]["stand"] == "2026-11-20", t
    assert g["_meta"]["zufalls_seed_basis"] == 20261121 and g["_meta"]["stand"]


def test_fehlerquote_erster_art_liegt_ueber_dem_soll_bei_kurzen_reihen_und_nahe_dem_soll_bei_langen():
    rates = {m: REF.typ1(m)["hits"] / REF.typ1(m)["R"] for m in C.M_TYP1}
    assert all(REF.typ1(m)["R"] == 4000 and REF.typ1(m)["ausgefallen"] == 0 for m in C.M_TYP1)
    assert rates[10] > 0.07 and rates[10] > rates[100]                         # 10 Jahre: deutlich über 5 %
    assert abs(rates[100] - 0.05) < 4 * math.sqrt(0.05 * 0.95 / 4000) + 0.01
    assert max(rates.values()) < 0.13


@pytest.mark.parametrize("beta", [0.01, 0.02, 0.04])
@pytest.mark.parametrize("m", [20, 50, 100])
def test_trennschaerfe_zellen_formel_gleich_modul_und_simulation_im_band(beta, m):
    c = REF.power_cell(beta, m)
    assert c["formel"] == pytest.approx(M.power(beta, m), abs=1e-12) and c["R"] == 4000
    sim = c["hits"] / (c["R"] - c["ausgefallen"])
    assert abs(sim - c["formel"]) < 3 * math.sqrt(c["formel"] * (1 - c["formel"]) / c["R"]) + 0.03


def test_trennschaerfe_waechst_mit_m_und_mit_beta_in_der_simulation():
    sim = lambda b, m: REF.power_cell(b, m)["hits"] / REF.power_cell(b, m)["R"]
    for b in (0.01, 0.02, 0.04):
        assert sim(b, 20) < sim(b, 50) < sim(b, 100) + 1e-9
    for m in (20, 50, 100):
        assert sim(0.01, m) < sim(0.02, m) < sim(0.04, m) + 1e-9


def test_jahre_bis_80_prozent_stehen_in_der_tabelle_und_die_simulation_liegt_im_band():
    got = {c["beta"]: c["m"] for c in REF.grid()["years80"]["daten"]}
    assert got == {0.005: 156, 0.01: 99, 0.02: 62, 0.04: 39}
    for c in REF.grid()["years80"]["daten"]:
        assert c["m"] == M.years_for_power(c["beta"]) and c["R"] == 2000
        assert abs(c["hits"] / c["R"] - c["formel"]) < 5 * math.sqrt(c["formel"] * (1 - c["formel"]) / c["R"])


def test_bloecke_gleich_exakten_werten_im_band_von_fuenf_standardfehlern():
    for c in REF.grid()["blocks"]["daten"]:
        n = c["blocks"]
        assert n == 400_000 and c["m"] == 50
        assert c["risiko_exakt"] == pytest.approx(M.risk_in_years(c["beta"], 50), abs=1e-12) and c["dll"] == pytest.approx(M.design_life_level(c["beta"], 50, 0.05), abs=1e-12)
        r = c["risiko_exakt"]
        assert abs(c["ueber_heute"] / n - r) < 5 * math.sqrt(r * (1 - r) / n)
        assert abs(c["ueber_dll"] / n - 0.05) < 5 * math.sqrt(0.05 * 0.95 / n)


def test_rmse_zellen_aufbau_und_zaehlung_ueber_alle_zwoelf_zellen():
    cells = REF.grid()["rmse"]["daten"]
    assert len(cells) == 12 and {(c["m"], c["beta"]) for c in cells} == {(m, b) for m in (20, 50, 100) for b in (0.0, 0.01, 0.02, 0.04)}
    assert all(c["R"] == 400 and c["ausgefallen"] == 0 for c in cells)
    # das richtige Modell (b, ξ = 0 bekannt) hat in allen 12 Zellen das kleinste RMSE
    assert all(c["b"]["rmse"] < min(c["a"]["rmse"], c["c"]["rmse"]) for c in cells)
    assert REF.rmse_counts() == {"c besser": 4, "c schlechter": 7, "Rauschen": 1}
    for c in cells:
        d = c["diff_c_a"]
        assert d["urteil"] == ("Rauschen" if d["z"] < 2 else ("c besser" if d["diff"] < 0 else "c schlechter"))
        assert d["diff"] == pytest.approx(c["c"]["rmse"] - c["a"]["rmse"], abs=1e-9)


def test_trendmodell_ist_bei_beta_null_und_bei_20_jahren_nie_klar_besser_und_bei_100_jahren_mit_trend_immer():
    for c in REF.grid()["rmse"]["daten"]:
        if c["beta"] == 0.0 or c["m"] == 20:
            assert c["diff_c_a"]["urteil"] != "c besser", (c["m"], c["beta"])
        if c["m"] == 100 and c["beta"] > 0:
            assert c["diff_c_a"]["urteil"] == "c besser"
    assert all(REF.rmse_cell(20, b)["c"]["med_abs"] >= REF.rmse_cell(20, b)["a"]["med_abs"] for b in C.BETA_OPTIONS)


def test_stationaere_verzerrung_ist_negativ_und_waechst_mit_beta_bei_100_jahren():
    bias = [REF.rmse_cell(100, b)["a"]["bias"] for b in (0.0, 0.01, 0.02, 0.04)]
    assert bias[0] > -0.3 and bias[1] > bias[2] > bias[3] and bias[3] < -1.0


def test_verzerrung_bei_50_jahren_beta_002_liegt_nahe_dem_modellfehler_der_mischung_plus_anpassungsfehler():
    b = REF.bias_cell()
    assert (b["m"], b["beta"], b["R"]) == (50, 0.02, 3000)
    model_err = M.mixture_quantile(0.02, 50) - M.level_year(0.02, 49)
    assert model_err == pytest.approx(-0.449, abs=0.002)
    assert b["bias"] < model_err + 0.0 and abs(b["bias"] - model_err) < 0.25      # Anpassungsfehler kleiner als der Modellfehler
    assert b["bias"] == pytest.approx(-0.54, abs=0.08) and b["se_bias"] == pytest.approx(b["sd"] / math.sqrt(b["R"]), rel=1e-9)
    assert b["xi_mittel"] < 0


def test_delta_abdeckung_sinkt_mit_dem_trend_fuer_das_letzte_nicht_fuer_das_mittlere_jahr():
    cs = {(m, b): REF.coverage_cell(m, b) for m, b in ((30, 0.0), (50, 0.02), (100, 0.02), (100, 0.04))}
    assert all(c["R"] == 1000 for c in cs.values())
    assert cs[(30, 0.0)]["letztes"] < 0.90                                       # schon ohne Trend unter den nominellen 95 %
    assert cs[(50, 0.02)]["letztes"] > cs[(100, 0.02)]["letztes"] > cs[(100, 0.04)]["letztes"]
    assert all(c["mittleres"] > 0.82 for c in cs.values())


def test_vehikel_a_skala_und_lage_wachsen_beide_und_die_naeherung_trifft_den_pegel_im_band():
    rows = sorted(REF.vehicle_c1(), key=lambda r: r["rho"])
    assert [r["rho"] for r in rows] == [0.70, 0.75, 0.80, 0.85, 0.90] and all(r["tage"] == 400_000 for r in rows)
    sig, mu = [r["sigma_dach"] for r in rows], [r["mu_dach"] for r in rows]
    assert sig == sorted(sig) and mu == sorted(mu)
    assert 2.3 < sig[-1] / sig[0] < 2.6 and 1.7 < mu[-1] / mu[0] < 1.9
    assert all(0.98 < r["pegel_simulation"] / r["pegel_rechnung"] < 1.10 for r in rows)
    ks = [r["ks_rechnung"] for r in rows]
    assert ks == sorted(ks) and ks[0] < 0.1 < 0.3 < ks[-1]
    # die Rechnung hat Lage 9,27 → 8,35 (fällt), die Simulation 10,06 → 17,99 (steigt): „Lage fast konstant“ gilt nur für die Rechnung
    assert rows[-1]["mu_rechnung"] < rows[0]["mu_rechnung"] and rows[-1]["mu_dach"] > rows[0]["mu_dach"]


def test_vehikel_a_falsches_modell_bei_wachsender_last():
    for rho_end in C.LOAD_END_OPTIONS:
        c = REF.vehicle_c2(rho_end)
        assert c["R"] == 300 and c["tage_je_lauf"] == 100 and c["wahr_tage"] == 400_000
        assert c["lage_skala_trend"]["median"] > c["stationaer"]["median"]          # das Modell mit Skalen-Trend liegt näher
        assert all(c[k]["median"] < 1.0 for k in ("stationaer", "lage_trend", "lage_skala_trend"))
        assert 0.95 < c["kontrolle_ohne_trend"]["median"] < 1.12                   # ohne Anstieg trifft die stationäre Anpassung im Median
    c9 = REF.vehicle_c2(0.90)
    assert all(c9[k]["anteil_unter_1"] > 0.95 for k in ("stationaer", "lage_trend", "lage_skala_trend"))
    assert c9["lage_trend"]["rmse_ln"] > c9["stationaer"]["rmse_ln"] - 0.01             # der Lage-Trend allein verbessert nichts
    assert c9["lage_skala_trend"]["rmse_ln"] < c9["stationaer"]["rmse_ln"]


def test_neu_rechnen_fehlerquote_m10_mit_dem_werkzeug_gleich_der_datei_als_band():
    import tsp_make_grid as G
    r = G.t_typ1(10)
    s = REF.typ1(10)
    assert abs(r["hits"] - s["hits"]) <= 3 and r["R"] == s["R"]


def test_neu_rechnen_wochenpegel_gleich_der_datei():
    import tsp_make_grid as G
    r, s = G.t_weekly(), REF.weekly()
    assert r["werktage"] == pytest.approx(s["werktage"], rel=1e-9) and r["wochenende"] == pytest.approx(s["wochenende"], rel=1e-9)


def test_neu_rechnen_trennschaerfe_mit_exact_log_strom_der_vorlage_im_band():
    """Eine Zelle (β = 0,04, m = 20) mit dem plattformfesten Strom der Vorlage: 300 Läufe, Band um die Formel und die eingefrorene Zahl."""
    R, m, beta = 300, 20, 0.04
    hits = sum(M.likelihood_ratio(S.gumbel_series(m, beta, S.derive_seed(321, i)))[0] > N.CHI2_95_DF1 for i in range(R))
    f = M.power(beta, m)
    c = REF.power_cell(beta, m)
    assert abs(hits / R - f) < 4 * math.sqrt(f * (1 - f) / R) + 0.03
    assert abs(hits / R - c["hits"] / c["R"]) < 4 * math.sqrt(f * (1 - f) / R) + 0.03
