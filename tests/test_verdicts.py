"""Verdikte als reine Funktionen: Art, jede Zahl im Text an handgesetzten Werten und an den exakten Größen; je Preset die erwartete Verdikt-Zeile (ein Preset ohne Effekt meldet keinen Effekt)."""

import pytest

import tsp_constants as C
import tsp_evaluation as EV
import tsp_models as M
import tsp_reference as REF


def _preset(name):
    p = C.PRESETS[name]
    return dict(m=p.get("m_sl", 50), beta=p.get("beta_sl", 0.0))


# --- Experiment 1 --------------------------------------------------------------------------------------------------------------------------------
def test_verdikt_trend_beta_002_50_jahre_nennt_formel_simulation_und_jahre_bis_80_prozent():
    kind, t = EV.verdict_trend(50, 0.02)
    assert kind == "warning"
    for s in ("β = 0,02 σ je Jahr, 50 Jahre (Gesamtanstieg 0,98 σ)", "rund 53 %", "Formel 53,2 %", "R = 4 000 Läufe", "Nach der asymptotischen Formel braucht er für 80 % bei diesem β 62 Jahre", "Nicht erkannt heißt nicht ausgeschlossen"):
        assert s in t, s
    assert "Soll 5 %" in t and "Standardfehler 0,34 Punkte" in t


def test_verdikt_trend_ohne_trend_meldet_keinen_trend_effekt_nur_die_fehlerquote():
    kind, t = EV.verdict_trend(50, 0.0)
    assert kind == "info"
    assert "erkennt den Trend" not in t and "Jahre braucht" not in t and "Trennschärfe" not in t
    assert "Fehlalarm" in t and "Soll 5 %" in t and "%" in t
    c = REF.typ1(50)
    assert f"{100 * c['hits'] / c['R']:.1f}".replace(".", ",") + " %" in t


def test_verdikt_trend_beta_001_100_jahre_nennt_99_jahre_und_ist_erfolg_ab_80_prozent_formel():
    kind, t = EV.verdict_trend(100, 0.01)
    assert "bei diesem β 99 Jahre" in t and kind == "success" and "rund 82 %" in t
    assert EV.verdict_trend(50, 0.01)[0] == "warning"


def test_verdikt_trend_starker_trend_ist_erfolg_und_laenge_ohne_zelle_wird_benannt():
    kind, t = EV.verdict_trend(100, 0.04)
    assert kind == "success" and "100 %" in t
    kind, t = EV.verdict_trend(75, 0.02)
    assert "keine eingefrorene Simulation" in t
    kind, t = EV.verdict_trend(75, 0.0)
    assert "keine eingefrorene Fehlerquote" in t


# --- Experiment 2 --------------------------------------------------------------------------------------------------------------------------------
def test_verdikt_risiko_beta_002_50_jahre_alle_zahlen_exakt():
    kind, t = EV.verdict_risk(0.02, 50, 0.05)
    assert kind == "warning"
    for s in ("mit 57,5 % überschritten", "ohne Trend mit 39,5 %", "die Jahre 0 bis 49", "sind es 58,2 %", "die Jahre 1 bis 50", "nach 50 Jahren 37 Jahre", "nach 100 Jahren 14;", "als mittlere Wartezeit gilt sie nur, bliebe diese Wahrscheinlichkeit stehen","bei 7,41 σ gegen 6,88 σ", "(+0,53)"):
        assert s in t, s


def test_verdikt_risiko_ohne_trend_ist_stationaer_und_meldet_keinen_effekt():
    kind, t = EV.verdict_risk(0.0, 50, 0.05)
    assert kind == "info" and "39,5 %" in t and "gleich dem stationären" in t and "+" not in t


def test_verdikt_risiko_zaehlung_der_jahre_wird_in_jedem_zweig_genannt():
    for beta in (0.0, 0.04):
        for nutz in C.NUTZ_OPTIONS:
            t = EV.verdict_risk(beta, nutz, 0.05)[1]
            assert f"die Jahre 0 bis {nutz - 1}" in t


# --- Experiment 3 --------------------------------------------------------------------------------------------------------------------------------
def test_verdikt_stationaer_50_jahre_beta_002_nennt_modellfehler_bias_und_abdeckung():
    kind, t = EV.verdict_stationary(50, 0.02)
    assert kind == "warning"
    assert "die Mischung liegt 0,45 σ unter dem Pegel des letzten Jahres (exakt" in t
    bias = REF.bias_cell()
    assert f"um {-bias['bias']:.2f} σ".replace(".", ",") in t and "R = 3 000 Läufe" in t
    cov = REF.coverage_cell(50, 0.02)
    assert f"{100 * cov['letztes']:.1f} %".replace(".", ",") in t


def test_verdikt_stationaer_ohne_trend_hat_keinen_modellfehler():
    kind, t = EV.verdict_stationary(50, 0.0)
    assert kind == "info" and "keinen Modellfehler" in t and "unter dem Pegel" not in t


def test_verdikt_stationaer_ohne_eingefrorene_zelle_nennt_nur_den_exakten_modellfehler():
    kind, t = EV.verdict_stationary(75, 0.02)
    assert "unter dem Pegel des letzten Jahres" in t and "Delta-Intervall" not in t and "R = 400" not in t


# --- Experiment 4 --------------------------------------------------------------------------------------------------------------------------------
def test_verdikt_trendmodell_nennt_die_zaehlung_ueber_zwoelf_zellen_und_die_zelle():
    kind, t = EV.verdict_trendmodel(100, 0.02)
    assert "in 4 von 12 Zellen klar besser, in 7 klar schlechter und in 1 im Rauschen" in t
    assert "Trendmodell klar besser" in t and "R = 400 Läufe" in t
    assert "Die Zählung hängt vom Zufalls-Seed ab" in t
    kind, t = EV.verdict_trendmodel(20, 0.04)
    assert "Trendmodell klar schlechter" in t and "ξ̂ am Rand" in t
    kind, t = EV.verdict_trendmodel(50, 0.02)
    assert "im Rauschen" in t


def test_verdikt_trendmodell_ohne_zelle_und_mit_eigenen_laeufen():
    kind, t = EV.verdict_trendmodel(30, 0.02, {"n": 20, "a": {"rmse": 1.5}, "b": {"rmse": 0.7}, "c": {"rmse": 2.0}})
    assert "keine eingefrorene Zelle" in t and "Eigene 20 Läufe" in t and "1,50" in t and "2,00" in t
    assert "Eigene" not in EV.verdict_trendmodel(30, 0.02, {"n": 0})[1]


# --- Experiment 5 --------------------------------------------------------------------------------------------------------------------------------
def test_verdikt_vehikel_nennt_last_modell_wahr_und_kontrolle():
    kind, t = EV.verdict_vehicle(0.90, "Lage- und Skalen-Trend")
    c = REF.vehicle_c2(0.90)
    assert kind == "warning" and "ρ von 0,70 auf 0,90" in t and "Lage- und Skalen-Trend" in t
    assert f"{c['wahr']:.1f}".replace(".", ",") in t and "400 000 Tagen" in t and "R = 300 Läufe zu je 100 Tagen" in t
    assert f"das {c['lage_skala_trend']['median']:.2f}-Fache".replace(".", ",") in t and f"dem {c['kontrolle_ohne_trend']['median']:.2f}-Fachen".replace(".", ",") in t
    assert "Skala ×2,46" in t and "Lage ×1,79" in t


@pytest.mark.parametrize("modell", C.MODELS)
@pytest.mark.parametrize("last", C.LOAD_END_OPTIONS)
def test_verdikt_vehikel_alle_modelle_und_endlasten_rechnen(modell, last):
    kind, t = EV.verdict_vehicle(last, modell)
    assert modell in t and f"{last:.2f}".replace(".", ",") in t


# --- Presets -------------------------------------------------------------------------------------------------------------------------------------
def test_jedes_preset_hat_eine_erwartete_trend_verdikt_zeile():
    erwartet = {
        "Kein Trend, 50 Jahre (Voreinstellung)": ("info", "Fehlalarm"),
        "Trend 0,02, 50 Jahre": ("warning", "rund 53 %"),
        "Trend 0,01, 50 Jahre (kaum sichtbar)": ("warning", "für 80 % bei diesem β 99 Jahre"),
        "Trend 0,04, 100 Jahre": ("success", "Gesamtanstieg 3,96 σ"),
        "Kurze Reihe, 20 Jahre": ("warning", "Gesamtanstieg 0,76 σ"),
        "Last wächst (Vehikel A)": ("info", "Fehlalarm"),
    }
    assert set(erwartet) == set(C.PRESETS)
    for name, (kind, s) in erwartet.items():
        k, t = EV.verdict_trend(**_preset(name))
        assert k == kind and s in t, name


def test_preset_ohne_trend_meldet_im_risiko_keinen_effekt_und_alle_anderen_schon():
    for name in C.PRESETS:
        beta = C.PRESETS[name].get("beta_sl", 0.0)
        kind, t = EV.verdict_risk(beta, 50, 0.05)
        assert (kind == "info") == (beta == 0.0), name


def test_exakte_zahlen_der_verdikte_stammen_aus_dem_modul():
    t = EV.risk_table(0.02, 50, 0.05)
    assert t.risk == pytest.approx(M.risk_in_years(0.02, 50), abs=1e-15) and t.survival_check == pytest.approx(0.95, abs=1e-12)
    assert t.dll - t.dll_stationary == pytest.approx(0.531, abs=2e-3)
