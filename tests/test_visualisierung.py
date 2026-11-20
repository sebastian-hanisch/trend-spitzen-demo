"""Abbildungen und Zahlenformat: jede Figur baut ohne Ausnahme, alle Achsen sind fest (`fixedrange`), die Zahlen stehen mit Dezimalkomma."""

import numpy as np

import tsp_constants as C
import tsp_evaluation as EV
import tsp_models as M
import tsp_reference as REF
import tsp_texts as T
import tsp_visualization as V


def _fest(fig):
    for ax in list(fig.select_xaxes()) + list(fig.select_yaxes()):
        assert ax.fixedrange is True


def test_alle_figuren_bauen_und_alle_achsen_sind_fest():
    a = EV.analyse(EV.Settings(beta=0.02))
    figs = [
        V.fig_series(a.x, 0.02, a.stat, a.trend),
        V.fig_power(0.02, [REF.power_cell(0.02, m) for m in C.M_FROZEN], 50),
        V.fig_power(0.0, [REF.typ1(m) for m in C.M_TYP1], 50),
        V.fig_return_period(0.02),
        V.fig_levels(a.x, 0.02, a.truth, a.gev_stat),
        V.fig_levels(a.x, 0.02, a.truth, None),
        V.fig_rmse(REF.rmse_cell(50, 0.02)),
        V.fig_vehicle_scale_location(REF.vehicle_c1()),
        V.fig_vehicle_ratio(REF.vehicle_c2(0.90)),
    ]
    for f in figs:
        _fest(f)
        assert len(f.data) >= 1


def test_trennschaerfe_figur_ohne_eingefrorene_zellen_zeigt_nur_die_formel():
    f = V.fig_power(0.02, [], 75)
    assert len(f.data) == 1 and np.all(np.isfinite(f.data[0].y))


def test_serie_figur_hat_die_wahre_und_die_angepassten_lagen():
    a = EV.analyse(EV.Settings(beta=0.04))
    f = V.fig_series(a.x, 0.04, a.stat, a.trend)
    names = [t.name for t in f.data]
    assert any("wahre Lage" in n for n in names) and any("mit Trend" in n for n in names) and any("ohne Trend" in n for n in names)
    trend_line = next(t for t in f.data if "mit Trend" in t.name)
    assert np.allclose(trend_line.y, a.trend.loc(np.arange(a.x.size)))


def test_zahlenformat_dezimalkomma_minuszeichen_und_leerzeichen_als_tausender():
    assert T.de(1.2345, 2) == "1,23" and T.de(-0.5, 1) == "−0,5" and T.de(-0.0001, 2) == "0,00" and T.de(float("nan")) == "–" and T.de(None) == "–"
    assert T.pct(0.5749, 1) == "57,5 %" and T.pct(0.05) == "5 %"
    assert T.pint(400000) == "400 000"


def test_ausgedachte_modellkonstanten_stimmen_mit_dem_modul_ueberein():
    assert C.LOAD_START == 0.70 and min(C.LOAD_END_OPTIONS) > C.LOAD_START
    assert C.N_BLOCKS_LIVE == 400 and M.P_HUNDRED == 0.01
