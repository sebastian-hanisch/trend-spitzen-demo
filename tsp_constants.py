"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets.

Zähleinheiten: Jahr = ein Jahresmaximum (Block), m = Jahre je Reihe, Läufe = Wiederholungen eines ganzen Experiments (je Lauf eine Reihe), Blöcke = Nutzungsdauern zu m Jahren,
Tage = Tagesmaxima des Vehikels A (je Tag 256 Kunden)."""

SEED_MAX, DEFAULT_SEED = 999999, 27     # Zufalls-Seed der Beispielreihe: aus 400 Zufalls-Seeds so gewählt, dass die Reihe die Aussage der Presets zeigt (README, „Befunde und Korrekturen“)

M_OPTIONS = (10, 20, 30, 50, 75, 100, 150)          # Jahre je Reihe (Experimente 1, 3, 4)
M_FROZEN = (20, 50, 100)                            # Jahre der eingefrorenen Zellen (Trennschärfe, RMSE)
BETA_OPTIONS = (0.0, 0.01, 0.02, 0.04)              # Trend β in σ je Jahr; alle vier stehen in den eingefrorenen Zellen
NUTZ_OPTIONS = (10, 25, 50, 75, 100)                # Nutzungsdauer in Jahren (Experiment 2)
Q_OPTIONS = (0.01, 0.05, 0.10)                      # Risiko q des Design Life Level
RUNS_MIN, RUNS_MAX, RUNS_STEP, DEFAULT_RUNS = 10, 60, 10, 20   # Läufe live (Experiment 4)
LOAD_END_OPTIONS = (0.80, 0.85, 0.90)               # Last am Ende des Vehikels A (Experiment 5; Anfang ρ = 0,70)
LOAD_START = 0.70
MODELS = ("stationär", "Lage-Trend", "Lage- und Skalen-Trend")
N_BLOCKS_LIVE = 400                                 # Blöcke der Live-Simulation in Experiment 2
BAND_SE = 4.0                                       # Bänder der Tests: ± so viele Standardfehler
LEVEL_YEARS = (0, 10, 25, 50, 75, 100)              # Jahre nach heute für die Wiederkehrperiode des heutigen Pegels
STEPS_YEARS80 = (0.005, 0.01, 0.02, 0.04)
M_TYP1 = (10, 20, 30, 50, 100)

COLOR_TRUE = "#2F6B65"
COLOR_FIT = "#d95f02"
COLOR_ALT = "#4c78a8"
COLOR_PTS = "rgba(80,80,80,0.75)"

DEFAULTS = {"m_sl": 50, "beta_sl": 0.0, "nutz_sl": 50, "q_sl": 0.05, "runs_sl": DEFAULT_RUNS, "last_sel": 0.80, "modell_sel": "stationär"}

PRESETS = {
    "Kein Trend, 50 Jahre (Voreinstellung)": {},
    "Trend 0,02, 50 Jahre": {"beta_sl": 0.02},
    "Trend 0,01, 50 Jahre (kaum sichtbar)": {"beta_sl": 0.01},
    "Trend 0,04, 100 Jahre": {"beta_sl": 0.04, "m_sl": 100},
    "Kurze Reihe, 20 Jahre": {"beta_sl": 0.04, "m_sl": 20},
    "Last wächst (Vehikel A)": {"last_sel": 0.90, "modell_sel": "Lage- und Skalen-Trend"},
}
PRESET_HELP = {
    "Kein Trend, 50 Jahre (Voreinstellung)": "Jahresmaxima ohne Trend (β = 0), 50 Jahre: der Test meldet in rund 5 bis 6 % der Läufe (Soll 5 %) fälschlich einen Trend; der heutige 100-Jahres-Pegel ist der Pegel der Reihe.",
    "Trend 0,02, 50 Jahre": "Lage steigt um 0,02 σ je Jahr (Gesamtanstieg 0,98 σ): der Test erkennt das in rund jedem zweiten Lauf; der heutige 100-Jahres-Pegel wird in 50 Jahren (Jahre 0 bis 49, das heutige eingeschlossen) mit 57,5 % überschritten (ohne Trend 39,5 %).",
    "Trend 0,01, 50 Jahre (kaum sichtbar)": "Halber Trend, 50 Jahre: der Test erkennt ihn nur in etwa jedem fünften Lauf; die Mischung liegt trotzdem 0,23 σ unter dem Pegel von heute.",
    "Trend 0,04, 100 Jahre": "Starker Trend, lange Reihe (Gesamtanstieg 3,96 σ): der Test erkennt ihn in allen Läufen der eingefrorenen Messung; eine stationäre Anpassung unterschätzt den Pegel von heute deutlich.",
    "Kurze Reihe, 20 Jahre": "Trend 0,04 σ je Jahr, nur 20 Jahre: der Test erkennt ihn nur in etwa jedem fünften Lauf, und das Trendmodell mit freiem ξ schätzt den Pegel hier nicht besser als die stationäre Anpassung (Experiment 4).",
    "Last wächst (Vehikel A)": "Experiment 5: die Last des M/M/1-Tores wächst von ρ = 0,70 auf 0,90; angezeigt wird das Modell mit Trend in Lage und Skala.",
}
