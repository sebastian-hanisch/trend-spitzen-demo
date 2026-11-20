"""Claims: jede Zahl der README steht hier als Text, gebildet aus der Rechnung (exakt) oder aus der eingefrorenen Messung; die README muss genau diese Zeilen der Tabelle „Zahl → Test“ enthalten.

Außerdem: Pflichtabschnitte der README, die Annahmen- und Konventionstabellen der App stehen wörtlich in der README, Links zeigen auf vorhandene Dateien, Footer gleich, Wortwahl (Zufalls-Seed)."""

import re
from pathlib import Path

import pytest

import tsp_texts as T
from claims_fragmente import fragmente, table_error_rows

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
FRAGS = fragmente()


@pytest.mark.parametrize("kid,text,art,beschr", FRAGS, ids=[f[0] for f in FRAGS])
def test_readme_enthaelt_die_zeile_der_zahl_test_tabelle(kid, text, art, beschr):
    assert f"| {kid} | {text} |" in README, f"{kid}: {text}"
    assert f"| {kid} | {text} | {art} | {beschr} |" in README


def test_kennungen_sind_eindeutig_und_alle_drei_arten_kommen_vor():
    ids = [f[0] for f in FRAGS]
    assert len(ids) == len(set(ids)) and {f[2] for f in FRAGS} == {"exakt", "Messung"}


def test_tabelle_mischung_minus_letztes_jahr_steht_in_der_readme():
    for m, cells in table_error_rows():
        assert f"| {m} | " + " | ".join(cells) + " |" in README


def test_pflichtabschnitte_der_readme():
    for h in ("## Warum dieses Problem", "## Befunde und Korrekturen gegenüber Plan und Vormessung", "## Ehrliche Grenzen", "## 🚧 Bewusst nicht enthalten / nicht gemessen",
              "## Zahl → Test", "## Quellen", "## Tests", "## Dateistruktur", "## Lokal ausführen"):
        assert h in README, h


def test_annahmen_und_konventionstabellen_der_app_stehen_woertlich_in_der_readme():
    for block in (T.ASSUMPTIONS, T.CONVENTIONS):
        for line in block.strip().splitlines():
            assert line in README, line


def test_readme_nennt_zufalls_seed_und_nie_saatwert_und_die_wichtigsten_hinweise():
    assert "Zufalls-Seed" in README and "Saatwert" not in README
    for s in ("nicht gelesen", "Port 10090", "Stufe A", "rng_splitmix64_exact", "Zufalls-Seed der Beispielreihe"):
        assert s in README, s
    assert "trägt" not in README and "tragen" not in README


def test_relative_links_der_readme_zeigen_auf_vorhandene_dateien():
    for target in re.findall(r"\]\(([^)#:]+?)\)", README):
        assert (ROOT / target).exists(), target
    for name in re.findall(r"`((?:tests/|tools/)?[a-z_]+\.(?:py|json))`", README):
        if "/" in name or name.startswith("tsp_") or name == "app.py":
            assert (ROOT / name).exists(), name


def test_readme_endet_mit_dem_footer_der_app():
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", README) if p.strip()]
    assert paragraphs[-1].startswith("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)")
