"""Gemeinsame Fixtures und Hilfen: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b).

Mini-Instanz: vier Jahre mit e^β = 2 (Spielzeugtrend β = ln 2, Handbeispiele H1 bis H6)."""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.dont_write_bytecode = True
os.environ.setdefault("OMP_NUM_THREADS", "1")


@pytest.fixture
def spielzeug():
    """Vier Jahre mit Raten λ_t = 1, 2, 4, 8 (β = ln 2)."""
    return [1, 2, 4, 8]
