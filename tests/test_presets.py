"""Presets: vollständig, gültige Werte, Standard-Preset = Standardeinstellungen, Permalink begrenzt und rastet ein."""

import pytest

import tsp_constants as C
import tsp_evaluation as EV
import tsp_presets as P


def test_every_preset_has_help_and_only_known_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) <= set(P.SETTING_SPECS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_valid_option_values():
    for name, preset in C.PRESETS.items():
        for key, value in preset.items():
            spec = P.SETTING_SPECS[key]
            if spec.options is not None:
                assert value in spec.options, (name, key, value)
            else:
                assert spec.lo <= value <= spec.hi


def test_default_preset_equals_the_default_settings():
    first = next(iter(C.PRESETS))
    assert C.PRESETS[first] == {}
    assert EV.Settings() == EV.Settings(m=50, beta=0.0, seed=C.DEFAULT_SEED, nutz=50, q=0.05, runs=C.DEFAULT_RUNS, last_end=0.80, modell="stationär")


def test_setting_specs_cover_every_preset_key_and_the_defaults_agree():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    for key, spec in P.SETTING_SPECS.items():
        if key in C.DEFAULTS:
            assert spec.default == C.DEFAULTS[key]
        if spec.options is not None:
            assert spec.default in spec.options


def test_presets_sind_verschieden_und_zeigen_unterschiedliche_szenarien():
    zustaende = [tuple(pr.get(k, sp.default) for k, sp in P.SETTING_SPECS.items()) for pr in C.PRESETS.values()]
    assert len(set(zustaende)) == len(zustaende)


def test_regler_optionen_enthalten_alle_eingefrorenen_zellen():
    assert set(C.M_FROZEN) <= set(C.M_OPTIONS) and 0.0 in C.BETA_OPTIONS
    assert set(C.LOAD_END_OPTIONS) == {c["rho_bis"] for c in __import__("tsp_reference").grid()["vehicle_c2"]["daten"]}
    assert set(C.M_TYP1) <= set(C.M_OPTIONS)


@pytest.mark.parametrize("wert,erwartet", [(31, 30), (1000, 150), (2, 10), (60, 50)])
def test_snap_rastet_jahre_auf_gueltige_werte_ein(wert, erwartet):
    assert P._snap(P.SETTING_SPECS["m_sl"], wert) == erwartet


def test_snap_begrenzt_zahlenregler_und_verwirft_unbekannte_texte():
    assert P._snap(P.SETTING_SPECS["runs_sl"], 1000) == C.RUNS_MAX and P._snap(P.SETTING_SPECS["runs_sl"], -3) == C.RUNS_MIN
    with pytest.raises(ValueError):
        P._snap(P.SETTING_SPECS["modell_sel"], "Weibull")
    assert P._snap(P.SETTING_SPECS["beta_sl"], 0.013) == 0.01
    assert P._snap(P.SETTING_SPECS["last_sel"], 0.87) == 0.85
