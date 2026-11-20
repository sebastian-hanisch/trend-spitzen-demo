"""Jeder Wert jedes Reglers der Seitenleiste rechnet ohne Ausnahme; jedes Preset zeigt seine Verdikt-Zeile und die Kernfragen stehen vor den Experimenten; Permalink-Werte werden übernommen;
kein Regler ist wirkungslos (Mode-conditional dead control)."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import tsp_constants as C
import tsp_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))


def _app():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _regler(at, key):
    for art in ("select_slider", "selectbox", "slider", "toggle", "number_input"):
        try:
            return getattr(at.sidebar, art)(key=key)
        except KeyError:
            continue
    raise AssertionError(key)


def _texte(at):
    return [e.value for kind in ("success", "info", "warning") for e in getattr(at, kind)]


def test_jede_option_jedes_auswahlreglers_rechnet_ohne_ausnahme():
    at = _app()
    n = 0
    for key, spec in P.SETTING_SPECS.items():
        if spec.options is None:
            continue
        for wert in spec.options:
            _regler(at, key).set_value(wert).run()
            assert not at.exception, (key, wert, [str(e) for e in at.exception])
            n += 1
        _regler(at, key).set_value(spec.default).run()
    assert n == sum(len(s.options) for s in P.SETTING_SPECS.values() if s.options is not None)


def test_laeufe_und_zufalls_seed_an_den_grenzen_rechnen():
    at = _app()
    for key, werte in (("runs_sl", (C.RUNS_MIN, C.RUNS_MAX)), ("seed_input", (0, C.SEED_MAX))):
        for w in werte:
            _regler(at, key).set_value(w).run()
            assert not at.exception, (key, w)


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_preset_zeigt_seine_verdikt_zeilen_und_die_kernfragen_stehen_vor_den_experimenten(name):
    at = _app()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert not at.exception
    texte = _texte(at)
    assert any("Ohne Trend (β = 0)" in t or "Gesamtanstieg" in t for t in texte)              # Experiment 1
    assert any("heutige 100-Jahres-Pegel" in t or "Pegel von heute" in t for t in texte)        # Experiment 2
    assert any("ρ von 0,70 auf" in t for t in texte)                                           # Experiment 5
    assert any(m.value.startswith("## ❓ Kernfragen") for m in at.markdown)
    subs = [h.value for h in at.subheader]
    ordnung = [next(i for i, v in enumerate(subs) if f"Experiment {n}" in v) for n in (2, 3, 4, 5)]
    assert ordnung == sorted(ordnung) and any("Wo die Annahmen enden" in v for v in subs)
    assert "st-key-kopf" in " ".join(h.proto.body for h in at.get("html"))


def test_preset_ohne_trend_meldet_im_standardlauf_keinen_trend_in_der_beispielreihe():
    at = _app()
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Der Test sagt"] == "kein Trend"


def test_preset_trend_002_zeigt_trend_in_der_beispielreihe():
    at = _app()
    at.sidebar.button(key="preset_Trend 0,02, 50 Jahre").click().run()
    assert {m.label: m.value for m in at.metric}["Der Test sagt"] == "Trend"


def test_permalink_wird_uebernommen_und_eingerastet():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.query_params["m"] = "55"             # kein gültiges m: rastet auf 50 ein
    at.query_params["beta"] = "0.02"
    at.query_params["modell"] = "Lage-Trend"
    at.query_params["last"] = "0.9"
    at.query_params["q"] = "0.07"           # rastet auf 0,05 oder 0,10 ein
    at.run()
    assert not at.exception
    assert at.session_state["m_sl"] == 50 and at.session_state["beta_sl"] == 0.02
    assert at.session_state["modell_sel"] == "Lage-Trend" and at.session_state["last_sel"] == 0.9
    assert at.session_state["q_sl"] in (0.05, 0.10)


def test_kein_regler_ist_wirkungslos_aenderung_aendert_die_anzeige():
    at = _app()
    basis = (_texte(at), [m.value for m in at.metric])
    wirkung = {}
    for key, wert in (("m_sl", 100), ("beta_sl", 0.02), ("seed_input", 12345), ("nutz_sl", 25), ("q_sl", 0.10), ("runs_sl", 10), ("last_sel", 0.90), ("modell_sel", "Lage-Trend")):
        at2 = _app()
        _regler(at2, key).set_value(wert).run()
        assert not at2.exception
        wirkung[key] = (_texte(at2), [m.value for m in at2.metric]) != basis
    assert all(wirkung.values()), wirkung
