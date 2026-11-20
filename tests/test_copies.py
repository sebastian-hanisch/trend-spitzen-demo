"""Kopier-Tests: die kopierten Dateien sind unverändert (SHA-256), und wo die Quelle vorhanden ist, stimmen sie mit ihr überein.

* `tsp_rng.py` = `tools/templates/rng_splitmix64_exact.py` des Website-Repos (reine Vorlage, nicht ändern).
* `tsp_mixture.json` = gemeinsame Gewichtstabelle der Linie (Demo „Überschreitungen und die GPD“, Vehikel B, 1 224 Komponenten), unverändert kopiert."""

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PROJECTS = ROOT.parent
SHA_RNG = "d69b2b2fd67c1234cc0d79249676a662a2189d02803eea7d2edfb04b7a505a3f"
SHA_MIXTURE = "669f821c78e0bf38c47fa846232f27cd2ff126f4fceadca5369a5c9af5551c9f"


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_rng_ist_die_unveraenderte_vorlage():
    assert _sha(ROOT / "tsp_rng.py") == SHA_RNG
    vorlage = PROJECTS / "sebastianhanisch-website" / "tools" / "templates" / "rng_splitmix64_exact.py"
    if not vorlage.exists():
        pytest.skip("Vorlage nicht vorhanden (CI): nur der SHA-256-Vergleich oben gilt")
    assert (ROOT / "tsp_rng.py").read_bytes() == vorlage.read_bytes()


def test_vehikel_b_tabelle_ist_unveraenderte_kopie_der_gemeinsamen_tabelle():
    assert _sha(ROOT / "tsp_mixture.json") == SHA_MIXTURE
    d = json.loads((ROOT / "tsp_mixture.json").read_text(encoding="utf-8"))
    assert d["n_komponenten"] == len(d["komponenten"]) == 1224
    assert d["summe_gewichte"] == pytest.approx(1.0, abs=1e-12)
    quelle = PROJECTS / "extremwert-planung" / "vormessung" / "ev_v13_vehikel_b_tabelle.json"
    if quelle.exists():
        assert quelle.read_bytes() == (ROOT / "tsp_mixture.json").read_bytes()


def test_quelle_und_stand_der_tabelle_stehen_im_begleitblatt():
    q = json.loads((ROOT / "tsp_mixture_quelle.json").read_text(encoding="utf-8"))
    assert q["sha256"] == SHA_MIXTURE and q["stand"] and q["quelle"] and q["felder"]
    for feld in ("beschreibung", "n_komponenten", "summe_gewichte", "komponenten"):
        assert q["felder"][feld]["quelle"] and q["felder"][feld]["stand"]


def test_der_vektorisierte_strom_ist_keine_kopie_sondern_eigener_code_mit_gleichem_strom():
    """`tsp_vecrng.py` ist eigener Code (nicht die Vorlage); seine Gleichheit mit der Vorlage prüft test_rng_und_szenario.py."""
    assert (ROOT / "tsp_vecrng.py").read_bytes() != (ROOT / "tsp_rng.py").read_bytes()
