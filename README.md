# Trend: der Pegel von heute, nicht der Pegel der Reihe – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-trend-spitzen-demo.streamlit.app/)**

Demo der Linie **„Extremwerte erklärt“** („wie groß wird das Größte, und wie sicher weiß man das?“) in der „Konzepte“-Reihe der Website „Sebastian Hanisch – Operations Research und Machine Learning“ (Bereich Statistik, Feld „Seltenes und Extremes“). Die Kante zur Demo davor („Spitzen-Cluster“, Abhängigkeit) ist `fix`: Bis dahin bleibt eine Annahme stehen, nämlich dass die Verteilung der Jahresmaxima sich **nicht mit der Zeit ändert** (Stationarität); in der Definition des Wiederkehrwerts (Demo „Wiederkehrwert“) steckt sie, und „Spitzen-Cluster“ behält sie bei, obwohl die Werte dort voneinander abhängen. Diese Demo hebt genau sie auf: Die Lage der Jahresmaxima wandert (μ_t = β·t). Vehikel sind ein **ausgedachtes Gumbel-Modell mit Lage-Trend** (alle Gegenproben exakt) und die **M/M/1-Wartezeit am Tor** mit wachsender Last (Vehikel A); die Wochenstruktur des Verweilzeit-Modells (Vehikel B) steht nur als Tabelle. Nachfolger ist die Demo „Welches Verfahren wann?“ (Verfahrenswahl).

Rechnet nur mit numpy und der Standardbibliothek (kein scipy, kein scikit-learn, kein statsmodels in der App); scipy steht nur in `requirements-dev.txt` und wird in den Tests als zweiter Rechenweg benutzt. Die Reihen sind **ausgedacht**: Die Zahlen sind Lehrbuch und Abstract in einem Modell und sagen nichts über Klima, Last, Lager oder Hafen.

## Warum dieses Problem

Bei einem Trend ist „der 100-Jahres-Pegel“ eine **Frage nach dem Jahr**. Eine stationäre Anpassung an die ganze Reihe liegt grob beim Pegel der **Mitte** (im Modell die Mischung aller Jahre; die ML-Anpassung der GEV an die Mischung weicht davon je nach Zelle ab, siehe „Ehrliche Grenzen“), nicht beim Pegel von **heute** (dem letzten Jahr der Reihe): Bei β = 0,02 σ je Jahr und 50 Jahren liegt die Mischung 0,45 σ unter dem Pegel des letzten Jahres (exakt, ohne Rauschen), die stationäre GEV liegt im Mittel −0,54 σ neben dem Pegel des letzten Jahres (Verzerrung bei R = 3 000 Läufen, Standardfehler 0,02). Der heutige 100-Jahres-Pegel wird in 50 Jahren mit **57,5 %** statt 39,5 % überschritten, **gezählt für die Jahre 0 bis 49, das heutige Jahr eingeschlossen**; zählt man 50 Jahre ab morgen (die Jahre 1 bis 50), sind es 58,2 %. Die App nennt die Zählung in den Kennzahlen, im Verdikt von Experiment 2, in der Preset-Hilfe, in der Gegenprobe und in der Blocktabelle. Die Wiederkehrperiode des heutigen Pegels (Kehrwert der Überschreitungswahrscheinlichkeit des jeweiligen Jahres, keine Frist) schrumpft bei β = 0,02 von 100 auf 82,0 Jahre nach 10 Jahren und auf 37,1 nach 50 (Tabelle E7). Als mittlere Wartezeit gilt dieser Kehrwert nur, bliebe die Wahrscheinlichkeit ab diesem Jahr stehen; läuft der Trend weiter, ist die mittlere Wartezeit ab Jahr 50 kürzer, 24,7 statt 37,1 Jahre.

Den Trend sieht man auf kurzer Reihe kaum: Im Gumbel-Modell mit Lage-Trend und **bekanntem ξ = 0** braucht der Test nach der asymptotischen Formel für 80 % Trennschärfe bei β = 0,01 / 0,02 / 0,04 σ je Jahr 99 / 62 / 39 Jahre; in der Simulation liegt er bei diesen Längen bei 80,3 % / 79,8 % / 76,6 % (R = 2 000 Läufe, M3). Bei β = 0,02 und 50 Jahren erkennt er den Trend in rund 53 % der Läufe (Simulation 53,2 % bei R = 4 000 Läufen, Standardfehler 0,8 Punkte; Formel 53,2 %). Mit freiem ξ ist die Trennschärfe **nicht gemessen**. Das Trendmodell mit freiem ξ ist bei gleichen Daten in der Messung dieser Demo 4 klar besser, 7 klar schlechter, 1 im Rauschen (Zählung über 12 Zellen, gepaarter Standardfehler; mit anderem Zufalls-Seed ändert sich die Aufteilung der übrigen 8 Zellen auf „klar schlechter“ und „im Rauschen“, die Vormessung gab 4 / 5 / 3); in diesen Messungen ist es bei β = 0 und bei 20 Jahren nie klar besser.

**Gilt für** Jahresmaxima Gumbel mit linearem Lage-Trend (β = 0 bis 0,04 σ je Jahr), σ = 1, unabhängig, bekannte Form des Modells; Zähleinheiten Jahre und Läufe (je Lauf eine Reihe).

## Die fünf Experimente (Kernfragen)

1. **Sieht man den Trend?** Eine gezogene Reihe mit wahrer, angepasster und stationärer Lage, Likelihood-Quotient gegen die χ²-Schwelle 3,84, Trennschärfe über m (Formel gegen eingefrorene Simulation), Jahre bis 80 %, Fehlerquote erster Art, Rekorde gegen H_m.
2. **Was heißt „100 Jahre“ bei Trend?** Wiederkehrperiode des heutigen Pegels je Jahr, Risiko in der Nutzungsdauer gegen stationär, Design Life Level (Rootzén/Katz 2013, Definition), alles exakt; Gegenprobe live mit 400 Blöcken und eingefroren mit 400 000 Blöcken zu 50 Jahren.
3. **Was liefert eine stationäre Anpassung?** Pegel-Dach gegen drei wahre Pegel (erstes Jahr, Mischung, letztes Jahr; alle exakt), Zerlegung in Modellfehler (Mischung gegen letztes Jahr) und Rest, Abdeckung des Delta-Intervalls.
4. **Hilft das Trendmodell?** RMSE der stationären GEV, des Gumbel-Trend-Modells mit bekanntem ξ (Orakel-Zeile) und der GEV mit Lage-Trend, 12 eingefrorene Zellen und eigene Läufe live.
5. **Last statt Zeit (Vehikel A).** Wächst die Last des M/M/1-Tores, wirken Skala **und** Lage der Tagesmaxima; drei Modelle gegen den Pegel des letzten Tages, eingefroren.

Presets in der Seitenleiste: „Kein Trend, 50 Jahre“, „Trend 0,02, 50 Jahre“, „Trend 0,01, 50 Jahre (kaum sichtbar)“, „Trend 0,04, 100 Jahre“, „Kurze Reihe, 20 Jahre“ und „Last wächst (Vehikel A)“; das aktive Preset ist hervorgehoben. Der **Zufalls-Seed der Beispielreihe** ist aus 400 Zufalls-Seeds so gewählt, dass die Beispielreihe die Aussage der Presets zeigt (ohne Trend kein Fehlalarm, bei β = 0,02 erkannt, bei β = 0,01 und bei 20 Jahren nicht erkannt); über viele Läufe entscheidet die Trennschärfe, nicht die eine Reihe. „Neue Reihe ziehen“ zeigt die Streuung.

## Modell

Jahresmaxima X_t = β·t + G_t, t = 0 … m−1, mit unabhängigen Standard-Gumbel-Werten G_t (Skala σ = 1); β in Einheiten σ je Jahr, Gesamtanstieg D = β·(m−1). Der **Pegel** des Jahres t zur Überschreitungswahrscheinlichkeit p ist x_t(p) = β·t − ln(−ln(1−p)); der 100-Jahres-Pegel hat die Grundhöhe y₁₀₀ = 4,600149. Die Formeln (Mischung, Risiko, Design Life Level, Trennschärfe, Rekorde, Anpassungen, Vehikel A) stehen im 📐-Expander der App; dort steht auch, was fehlt (der Formelblock ist teils Skizze).

Alle exakten Größen sind Formeln im Modul `tsp_models.py`; die Orakel-Tests rechnen sie auf einem zweiten Weg nach (Produkt der Jahresverteilungen, `brentq`, Quadratur, Aufzählung aller 24 Ordnungen, Simulation mit dem Strom der Vorlage). Einzelne Tests vergleichen Funktionen des Moduls untereinander (zum Beispiel `risk_in_years` gegen `design_life_level`); das sind Konsistenz-, keine Orakeltests.

## Methodik

* **Zufall:** nur die SplitMix64-Vorlage `tools/templates/rng_splitmix64_exact.py` des Website-Repos (unverändert kopiert nach `tsp_rng.py`, SHA-256 im Kopier-Test): Ein Gumbel-Wert kostet zwei `exact_log`-Aufrufe (rund 0,1 ms); die Ziehungen sind per Konstruktion plattformunabhängig (Ganzzahlarithmetik, `decimal`), auf Linux nicht geprüft. Die eingefrorenen Tabellen nutzen denselben Strom **vektorisiert** (`tsp_vecrng.py`, Ganzzahlarithmetik in numpy, bitgleich zur Vorlage, im Test geprüft), rechnen aber mit `numpy.log` um und sind deshalb nur als Band prüfbar.
* **Trendtest:** Likelihood-Quotient der Gumbel-Anpassung mit gegen ohne Lage-Trend (ξ = 0 bekannt, Skala frei), Newton-Verfahren mit analytischem Gradienten und Hesse-Matrix, Schwelle χ²₁ = 3,8415 (Niveau 5 %). Trennschärfe-Formel Φ(√λ − 1,96) + Φ(−√λ − 1,96) mit λ = β²·m(m²−1)/12 (Information für β gleich Σ(t − t̄)², orthogonal zu Lage und Skala); sie ist **asymptotisch**, ein Lehrbuchwerkzeug und hier nicht aus einer Quelle zitiert.
* **GEV (ξ frei, ξ ≥ −1):** eigener Nelder-Mead, Start aus der Gumbel-Anpassung mit ξ₀ = 0,02 und −0,1; stationär 3, mit Lage-Trend 4 freie Parameter. ξ-Vorzeichen: ξ > 0 Fréchet-Typ, `scipy.stats.genextreme` hat c = −ξ (so im Test).
* **Delta-Intervall:** numerische Hesse-Matrix der negativen Log-Likelihood in (μ, ln σ, ξ), Gradient des Pegels per Differenzen; „kein Intervall“ zählt als nicht gedeckt.
* **Vehikel A:** M/M/1-Wartezeit (Dienstrate 1, Ankunftsrate ρ), Tag = 256 Kunden (Annahme), Tage unabhängig, Start im Gleichgewicht (P(W = 0) = 1−ρ, sonst Exp(1−ρ)); Lindley-Rekursion über alle Tage zugleich.
* **Eingefroren** (`tsp_grid.json`, Quelle und Stand je Teil, Erzeuger `tools/tsp_make_grid.py` mit `--check`): Fehlerquote und Trennschärfe, Blöcke, RMSE-Tabelle, Verzerrung, Delta-Abdeckung, Vehikel A, Wochenstruktur. **Live:** eine Reihe samt Tests und Anpassungen, alle exakten Größen, 400 Blöcke zu m Jahren und bis zu 60 eigene Läufe.

## Befunde (gemessen, keine Behauptungen)

Zähleinheiten stehen in jeder Zeile; jede Zahl steht in der Tabelle „Zahl → Test“ weiter unten.

| Frage | Befund |
|---|---|
| Wie oft meldet der Test ohne Trend einen Trend? | Bei m = 10 / 20 / 30 / 50 / 100 Jahren in 10,1 % / 6,4 % / 6,4 % / 5,8 % / 5,9 % der Läufe (R = 4 000 Läufe je m, Soll 5 %, Standardfehler 0,35 Punkte); bei kurzen Reihen ist er zu großzügig (M1); bei m = 50 und 100 liegt der Anteil 0,8 und 0,9 Punkte über dem Soll (2,3 und 2,6 Standardfehler; eine Neumessung mit R = 12 000 Läufen gab 5,9 % und 5,2 %, nicht im Test). |
| Trifft die Formel der Trennschärfe? | In 9 Zellen (β = 0,01 / 0,02 / 0,04 × m = 20 / 50 / 100, R = 4 000 Läufe) liegt die Simulation höchstens 2,1 Punkte neben der Formel; bei β = 0,02, m = 50 Formel 53,2 %, Simulation 53,2 % (E12, M2; die Gleichheit ist Zufall, der Standardfehler der Simulation ist 0,8 Punkte, eine Neumessung mit R = 12 000 Läufen gab 52,7 %). Eine Zelle mit 1,0 σ Gesamtanstieg (β = 1/49) hat 54,9 % und ist eine andere Zelle. |
| Wie viele Jahre bis 80 %? | 156 / 99 / 62 / 39 Jahre bei β = 0,005 / 0,01 / 0,02 / 0,04 (Formel); die Simulation bei diesem m (R = 2 000 Läufe) gibt 80,5 % / 80,3 % / 79,8 % / 76,6 %; bei β = 0,04, m = 39 liegt sie 4 Standardfehler unter der Formel (E14, M3). Eine Neumessung mit R = 12 000 Läufen gab 79,8 % / 79,3 % / 78,7 % bei m = 99 / 62 / 39 (nicht im Test): Die Formel liegt dort 1,2 bis 1,6 Punkte darüber und überschätzt die Trennschärfe nicht nur bei β = 0,04. |
| Was heißt „100 Jahre“ bei β = 0,02? | Risiko 57,5 % (Jahre 0 bis 49) gegen 39,5 %; 58,2 % für die Jahre 1 bis 50; Wiederkehrperiode (Kehrwert der Überschreitungswahrscheinlichkeit des Jahres) 100,0 / 82,0 / 60,9 / 37,1 / 22,7 / 14,0 Jahre nach 0 / 10 / 25 / 50 / 75 / 100 Jahren, die mittlere Wartezeit bei weiter steigendem Trend ist ab Jahr 50 mit 24,7 Jahren kürzer; Design Life Level (Risiko 5 % in 50 Jahren) 7,41 σ gegen 6,88 σ (E4 bis E8). |
| Was liefert eine stationäre Anpassung? | Pegel des ersten Jahres, Mischung und Pegel des letzten Jahres bei β = 0,02, m = 50: 4,600 / 5,131 / 5,580; Modellfehler 0,45 σ; Verzerrung der stationären GEV im Mittel −0,54 σ (Standardfehler 0,02, R = 3 000 Läufe); Abdeckung des 95-%-Delta-Intervalls für den Pegel des letzten Jahres 83,5 % / 75,2 % / 53,8 % / 33,4 % in den Zellen (m, β) = (30, 0) / (50, 0,02) / (100, 0,02) / (100, 0,04) (R = 1 000 Läufe). |
| Hilft das Trendmodell? | Bei gleichem freiem ξ: 4 klar besser, 7 klar schlechter, 1 im Rauschen (Zählung über 12 Zellen, R = 400 Läufe je Zelle); bei m = 100 und β = 0,02 RMSE 0,80 gegen 1,18 σ. Die Aufteilung der übrigen 8 Zellen auf „klar schlechter“ und „im Rauschen“ hängt vom Zufalls-Seed ab (Vormessung 4 / 5 / 3); in den Neumessungen der Aussagenprüfung (andere Zufalls-Seeds, R = 400 je Zelle) blieben die 4 Zellen klar besser (m = 100 mit β ≥ 0,01 und m = 50 mit β = 0,04) und keine weitere Zelle wurde klar besser. Bei β = 0 und bei 20 Jahren ist es in diesen Messungen nie klar besser; die RMSE-Zahlen bei 20 Jahren sind ausreißerbestimmt. |
| Last statt Zeit? | Skala und Lage der Tagesmaxima wachsen von ρ = 0,70 auf 0,90 um ×2,46 und ×1,79 (Gumbel-ML, 400 000 Tage je Last); die Gumbel-Näherung der Rechnung trifft den 100-Tage-Pegel bis ρ = 0,80 mit höchstens 2,4 % Abweichung, bei ρ = 0,90 liegt sie 8,6 % zu niedrig (aus der Tabelle unten gerechnet: −0,7 / −1,3 / −2,4 / −3,8 / −8,6 % für ρ = 0,70 bis 0,90), Skala und Lage einzeln trifft sie nicht. Bei einer Last, die von ρ = 0,70 auf 0,90 wächst, schätzen alle drei Modelle den Pegel des letzten Tages im Median zu niedrig: 0,72 / 0,71 / 0,78; 1,07 (stationär / Lage-Trend / Lage und Skala; Kontrolle ohne Anstieg; Endlast 0,90, R = 300 Läufe zu 100 Tagen); bei den Endlasten 0,80 und 0,85 liegen die drei Modelle ebenfalls unter 1 (Tabelle unten). Die Kontrolle liegt über 1; das stützt, belegt aber nicht, dass der Anstieg die Ursache ist (M9, M10). |
| Wochenstruktur (Vehikel B)? | Pegel je Container zu p = 10⁻⁵: nur Werktage / nur Wochenende 253,36 / 342,00 Tage, Faktor 1,35 (Quadratur und Bisektion; nur Tabelle, keine Anpassung); der Faktor ist exp(0,30), der Wochenend-Parameter der Mischung, also eine Eingabe und kein Ergebnis. |

### Tabellen der Messungen

**Trennschärfe, Formel gegen Simulation** (R = 4 000 Läufe je Zelle; Test: Gumbel mit gegen ohne Trend, ξ = 0 bekannt):

| β (σ je Jahr) | m (Jahre) | Gesamtanstieg D (σ) | Formel | Simulation | Abstand (Punkte) |
|---|---|---|---|---|---|
| 0,01 | 20 | 0,19 | 5,8 % | 7,8 % | 2,1 |
| 0,01 | 50 | 0,49 | 17,5 % | 19,7 % | 2,1 |
| 0,01 | 100 | 0,99 | 82,3 % | 80,3 % | −1,9 |
| 0,02 | 20 | 0,38 | 8,1 % | 9,9 % | 1,8 |
| 0,02 | 50 | 0,98 | 53,2 % | 53,2 % | 0,0 |
| 0,02 | 100 | 1,98 | 100,0 % | 100,0 % | 0,0 |
| 0,04 | 20 | 0,76 | 17,8 % | 19,7 % | 1,9 |
| 0,04 | 50 | 1,96 | 98,3 % | 97,7 % | −0,6 |
| 0,04 | 100 | 3,96 | 100,0 % | 100,0 % | 0,0 |

**Jahre bis 80 % Trennschärfe** (kleinstes m mit Formel ≥ 0,8; Simulation R = 2 000 Läufe):

| β (σ je Jahr) | m (Jahre) | Formel | Simulation |
|---|---|---|---|
| 0,005 | 156 | 80,3 % | 80,5 % |
| 0,010 | 99 | 81,2 % | 80,3 % |
| 0,020 | 62 | 80,5 % | 79,8 % |
| 0,040 | 39 | 80,3 % | 76,6 % |

**Fehlerquote erster Art** (β = 0, R = 4 000 Läufe je m):

| m (Jahre) | Anteil der Läufe mit Meldung „Trend“ |
|---|---|
| 10 | 10,1 % |
| 20 | 6,4 % |
| 30 | 6,4 % |
| 50 | 5,8 % |
| 100 | 5,9 % |

**Mischung minus Pegel des letzten Jahres** (exakt, in σ):

| m (Jahre) | β = 0,01 | β = 0,02 | β = 0,04 |
|---|---|---|---|
| 30 | −0,14 | −0,28 | −0,52 |
| 50 | −0,23 | −0,45 | −0,82 |
| 100 | −0,45 | −0,83 | −1,39 |

**Blöcke, Risiko und Design Life Level** (400 000 Blöcke zu 50 Jahren je β, Jahr für Jahr gezogen):

| β | Risiko exakt | Anteil der Blöcke über dem heutigen 100-Jahres-Pegel | Design Life Level (q = 5 %) | Anteil über dem Design Life Level |
|---|---|---|---|---|
| 0,00 | 39,50 % | 39,45 % | 6,882 | 5,01 % |
| 0,01 | 47,73 % | 47,70 % | 7,138 | 4,96 % |
| 0,02 | 57,47 % | 57,50 % | 7,414 | 4,95 % |
| 0,04 | 79,27 % | 79,18 % | 8,024 | 4,99 % |

**Delta-Abdeckung** (95-%-Intervall der stationären GEV, R = 1 000 Läufe, Standardfehler 1,6 Punkte):

| m (Jahre) | β | Deckt den Pegel des letzten Jahres | Deckt den Pegel des mittleren Jahres |
|---|---|---|---|
| 30 | 0,00 | 83,5 % | 83,5 % |
| 50 | 0,02 | 75,2 % | 84,4 % |
| 100 | 0,02 | 53,8 % | 88,2 % |
| 100 | 0,04 | 33,4 % | 90,4 % |

**Drei Anpassungen an dieselben Reihen** (R = 400 Läufe je Zelle; RMSE in σ von (a) stationäre GEV / (b) Gumbel mit Trend, ξ = 0 bekannt / (c) GEV mit Trend; Median des Betrags von (a) / (c); RMSE(c) − RMSE(a) mit gepaartem Standardfehler):

| m (Jahre) | β | RMSE (a) / (b) / (c) | Median des Betrags (a) / (c) | (c) − (a) (Standardfehler) | Urteil (Betrag über 2 Standardfehlern: klar) |
|---|---|---|---|---|---|
| 100 | 0,01 | 0,93 / 0,46 / 0,81 | 0,70 / 0,56 | −0,12 (0,02) | klar besser |
| 100 | 0,02 | 1,18 / 0,43 / 0,80 | 0,98 / 0,53 | −0,38 (0,04) | klar besser |
| 100 | 0,04 | 1,75 / 0,44 / 0,79 | 1,61 / 0,54 | −0,97 (0,04) | klar besser |
| 100 | 0,00 | 0,72 / 0,46 / 0,75 | 0,46 / 0,53 | 0,03 (0,01) | klar schlechter |
| 20 | 0,01 | 2,73 / 1,02 / 4,11 | 1,49 / 1,62 | 1,38 (0,47) | klar schlechter |
| 20 | 0,02 | 6,62 / 1,02 / 11,74 | 1,41 / 1,57 | 5,12 (1,73) | klar schlechter |
| 20 | 0,04 | 2,47 / 1,00 / 8,11 | 1,42 / 1,43 | 5,63 (2,47) | klar schlechter |
| 20 | 0,00 | 2,22 / 1,04 / 6,68 | 1,16 / 1,46 | 4,46 (1,96) | klar schlechter |
| 50 | 0,01 | 1,19 / 0,65 / 1,28 | 0,89 / 0,84 | 0,09 (0,04) | klar schlechter |
| 50 | 0,02 | 1,24 / 0,62 / 1,25 | 0,96 / 0,86 | 0,01 (0,04) | Rauschen |
| 50 | 0,04 | 1,38 / 0,60 / 1,15 | 1,08 / 0,72 | −0,23 (0,06) | klar besser |
| 50 | 0,00 | 1,13 / 0,64 / 1,25 | 0,66 / 0,76 | 0,12 (0,03) | klar schlechter |

Das Gumbel-Trend-Modell (b) hat in allen 12 Zellen das kleinste RMSE; das ist der Gewinn aus dem **bekannten Rand** (ξ = 0 vorab richtig bekannt), nicht aus dem Trend, und nur als Orakel-Zeile gemeint. Bei 20 Jahren klebt ξ̂ der GEV mit Trend in 9 / 14 / 12 / 12 von 400 Läufen (β = 0 / 0,01 / 0,02 / 0,04) am Rand ξ = −1; die RMSE-Zahlen dort hängen an wenigen solchen Läufen und an der Randbehandlung des Schätzers, belastbar ist der Median des Betrags.

**Vehikel A, Skala und Lage über der Last** (Gumbel-ML der Tagesmaxima, in Klammern die Näherung der Rechnung; 400 000 Tage je Last; Pegel = 0,99-Quantil des Tagesmaximums in Abfertigungsdauern):

| ρ | Skala | Lage | Pegel (Simulation) | KS-Abstand der Rechnung |
|---|---|---|---|---|
| 0,70 | 3,00 (3,33) | 10,06 (9,27) | 24,77 (24,60) | 0,092 |
| 0,75 | 3,48 (4,00) | 11,18 (9,94) | 28,71 (28,34) | 0,122 |
| 0,80 | 4,21 (5,00) | 12,65 (10,52) | 34,34 (33,52) | 0,171 |
| 0,85 | 5,30 (6,67) | 14,69 (10,59) | 42,90 (41,26) | 0,251 |
| 0,90 | 7,38 (10,00) | 17,99 (8,35) | 59,46 (54,35) | 0,399 |

**Vehikel A, falsches Modell bei wachsender Last** (R = 300 Läufe zu je 100 Tagen, Last linear von ρ = 0,70 auf ρ_bis; wahr = 0,99-Quantil des Tagesmaximums bei ρ_bis aus 400 000 Tagen):

| ρ_bis | wahr | Modell | Median Schätzwert / wahr | Q10 bis Q90 | Anteil unter wahr | RMSE von ln(Verhältnis) |
|---|---|---|---|---|---|---|
| 0,80 | 34,16 | stationäre GEV | 0,874 | 0,77 bis 0,99 | 91,3 % | 0,170 |
| 0,80 | 34,16 | GEV mit Lage-Trend | 0,893 | 0,79 bis 1,02 | 87,7 % | 0,151 |
| 0,80 | 34,16 | Gumbel, Trend in Lage und ln σ | 0,933 | 0,80 bis 1,06 | 77,0 % | 0,135 |
| 0,80 | 34,16 | Kontrolle ohne Anstieg | 1,020 | 0,89 bis 1,19 | 41,7 % | 0,120 |
| 0,85 | 42,93 | stationäre GEV | 0,799 | 0,70 bis 0,94 | 96,3 % | 0,246 |
| 0,85 | 42,93 | GEV mit Lage-Trend | 0,814 | 0,71 bis 0,95 | 96,3 % | 0,230 |
| 0,85 | 42,93 | Gumbel, Trend in Lage und ln σ | 0,876 | 0,76 bis 1,00 | 89,0 % | 0,175 |
| 0,85 | 42,93 | Kontrolle ohne Anstieg | 1,058 | 0,91 bis 1,23 | 35,3 % | 0,131 |
| 0,90 | 59,58 | stationäre GEV | 0,719 | 0,60 bis 0,85 | 99,0 % | 0,357 |
| 0,90 | 59,58 | GEV mit Lage-Trend | 0,714 | 0,61 bis 0,84 | 99,3 % | 0,361 |
| 0,90 | 59,58 | Gumbel, Trend in Lage und ln σ | 0,775 | 0,66 bis 0,91 | 98,3 % | 0,278 |
| 0,90 | 59,58 | Kontrolle ohne Anstieg | 1,074 | 0,89 bis 1,32 | 32,3 % | 0,172 |

## Befunde und Korrekturen gegenüber Plan und Vormessung

Zufallsanteile sind in dieser Demo mit dem SplitMix64-Strom und anderem Zufalls-Seed neu gemessen (Basis 20261121, die Vormessung benutzte `numpy.default_rng`); Abweichungen stehen hier mit alt → neu.

* **U3 aus der Zweitprüfung umgesetzt:** Die 57,5 % zählen die Jahre 0 bis 49 (das heutige Jahr eingeschlossen); die Jahre 1 bis 50 geben 58,2 %. App und README nennen die geltende Zählung bei diesen Zahlen (App: Kennzahlen, Verdikt, Preset-Hilfe, Gegenprobe, Blocktabelle).
* **Fehlerquote erster Art:** 9,38 / 7,40 / 6,40 / 5,55 / 5,62 % → 10,1 % / 6,4 % / 6,4 % / 5,8 % / 5,9 % bei m = 10 / 20 / 30 / 50 / 100 (R = 4 000 beide; Standardfehler 0,35 Punkte; die Abweichungen liegen bei 2 Standardfehlern). Bei 10 Jahren sind es nun rund 10 % statt 9,4 %.
* **Trennschärfe-Zellen:** Formel unverändert (53,2 / 17,5 / 82,3 / 17,8 %); Simulation bei (β, m) = (0,02, 50) 52,8 → 53,2 %, (0,01, 50) 19,2 → 19,7 %, (0,01, 100) 80,8 → 80,3 %, (0,04, 20) 19,2 → 19,7 %. In (0,01, 50) und (0,04, 20) liegt die Simulation 2 Punkte **über** der Formel (3,6 und 3,1 Standardfehler), in (0,01, 100) 2 Punkte darunter: Die Formel ist asymptotisch und trifft in 9 Zellen auf höchstens 2,1 Punkte, aber nicht überall im Rahmen des reinen Zufalls. Systematisch ist davon nur die kurze Reihe (0,04, 20): Eine Neumessung mit R = 12 000 Läufen gab dort 19,2 % bei Formel 17,8 % (3,8 Standardfehler), und die Fehlerquote erster Art bei m = 20 lag in derselben Neumessung bei 7,1 % statt 5 % (M1: 6,4 %). Bei (0,01, 50) gab die Neumessung 18,2 % gegen 17,5 % (2 Standardfehler); der eingefrorene Wert 19,7 % ist dort überwiegend Zufallsausschlag und keine Formelabweichung (Neumessungen nicht im Test). Die README-Anker „rund 53 %“ gelten nur bei β = 0,02 und 50 Jahren.
* **Jahre bis 80 %, Simulation bei diesem m:** 81,4 / 79,4 / 80,0 / 78,5 % (R = 1 000) → 80,5 % / 80,3 % / 79,8 % / 76,6 % (R = 2 000). Bei β = 0,04, m = 39 liegt die Simulation 4 Standardfehler unter der Formel (80,3 %): Die Formel überschätzt dort die Trennschärfe. Eine Neumessung mit R = 12 000 Läufen (numpy-Zufall) gab 79,8 % / 79,3 % / 78,7 % bei m = 99 / 62 / 39 gegen Formel 81,2 % / 80,5 % / 80,3 %: Die Überschätzung von 1,2 bis 1,6 Punkten zeigt sich auch bei β = 0,02 und β = 0,01, nicht nur bei β = 0,04; bei β = 0,04 erreicht die Simulation 80 % erst mit 40 Jahren (81,7 %).
* **Verzerrung der stationären GEV** bei m = 50, β = 0,02: −0,57 ± 0,02 σ → −0,54 σ (Standardfehler 0,02, R = 3 000 Läufe); ξ̂ im Mittel −0,03 → −0,04. Der Modellfehler der Mischung (exakt, 0,45 σ) und die Zerlegung bleiben; der Rest von 0,09 σ ist dabei zum größeren Teil ein systematischer Abstand der GEV-Anpassung von der Mischung und nicht Rauschen (siehe „Ehrliche Grenzen“).
* **Delta-Abdeckung (letztes Jahr):** 71,7 / 53,0 / 34,7 % (R = 300, drei Zellen) und 85,3 % ohne Trend bei m = 30 → 83,5 % / 75,2 % / 53,8 % / 33,4 % für die Zellen (m, β) = (30, 0) / (50, 0,02) / (100, 0,02) / (100, 0,04) (R = 1 000 Läufe, Standardfehler 1,6 Punkte). Die Abdeckung des mittleren Jahres liegt bei 83,5 % bis 90,4 %.
* **Trendmodell gegen stationär (12 Zellen):** 4 klar besser, 5 klar schlechter, 3 im Rauschen → **4 klar besser, 7 klar schlechter, 1 im Rauschen**. Die Änderung steckt in den vier Zellen bei 20 Jahren: Alle vier sind jetzt klar schlechter (vorher zwei im Rauschen), dazu ist m = 50, β = 0,02 weiter im Rauschen. Die RMSE-Zahlen bei m = 20 sind ausreißerbestimmt und schwanken stark mit dem Zufalls-Seed und der Randbehandlung des Schätzers (stationär 2,89 / 3,02 / 2,64 / 2,50 → 2,22 / 2,73 / 6,62 / 2,47; Trend 6,30 / 7,48 / 4,45 / 17,08 → 6,68 / 4,11 / 11,74 / 8,11). Der Median des Betrags bei 20 Jahren, im Plan „1,5 gegen 1,3 σ“, ist in dieser Messung 1,16 bis 1,49 (stationär) gegen 1,43 bis 1,62 σ (mit Trend), in jeder der vier Zellen mit Trend mindestens so groß wie stationär. Bei m = 100 und β = 0,02: RMSE mit Trend gegen stationär 0,75 gegen 1,26 σ → 0,80 gegen 1,18 σ.
* **Vehikel A, Skala und Lage:** ×2,47 und ×1,80 → ×2,46 und ×1,79 (400 000 Tage je Last); Pegel (Simulation) bei ρ = 0,70 / 0,80 / 0,90 24,70 / 34,48 / 59,34 → 24,77 / 34,34 / 59,46. **„Lage fast konstant“ aus der Vormessung V2 gilt nur für die Näherung der Rechnung (Lage 9,27 → 8,35), nicht für die Simulation (Lage steigt)**; die Folgerung „ein Lage-Trend allein ist das falsche Modell“ bleibt, mit der anderen Begründung: Skala und Lage wachsen beide.
* **Vehikel A, falsches Modell** (ρ von 0,70 auf 0,90): Median Schätzwert/wahr 0,717 / 0,699 / 0,793 → 0,719 / 0,714 / 0,775; Anteil unter wahr 98,0 / 98,7 / 97,0 % → 99,0 % / 99,3 % / 98,3 %; Kontrolle ohne Anstieg 1,058 (37,7 % darunter) → 1,074 (32,3 %); RMSE von ln(Verhältnis) 0,358 / 0,367 / 0,260 → 0,357 / 0,361 / 0,278; wahr 59,47 → 59,58. Das Modell mit Skalen-Trend liegt im RMSE von ln(Verhältnis) niedriger als die anderen; der Lage-Trend allein liegt bei der Endlast 0,90 nicht niedriger als stationär (0,361 gegen 0,357), bei den Endlasten 0,80 und 0,85 etwas niedriger (0,151 gegen 0,170 und 0,230 gegen 0,246). Einen gepaarten Standardfehler gibt es bei R = 300 Läufen nicht; die kleinen Unterschiede des Lage-Trends allein liegen im Bereich der Streuung. Alle drei Modelle liegen im Median unter dem wahren Pegel.
* **Wochenstruktur (Vehikel B):** Pegel nur Werktage / nur Wochenende 253,44 / 342,11 → 253,36 / 342,00 Tage (aus der Definition der Linie neu gerechnet, Quadratur und Bisektion; Faktor weiter 1,35 = exp(0,30), der Wochenend-Parameter der Mischung, also eine Eingabe); die Abweichung von 0,03 % kommt aus der Vormessung (V2 6), ihre Ursache ist nicht geklärt. Der Test prüft die Wochenpegel gegen eine eigene Aufzählung und den Pegel der ganzen Woche gegen die kopierte Tabelle (287,158 je Container, gemeinsame Definition der Linie).
* **Nicht übernommen:** Die Zahl „−0,57 σ“ und die Zellen „klar schlechter 5“ als feste Aussagen; sie stehen nur noch als Alt-Werte in dieser Liste. „Lage fast konstant“ und „55 % bei β = 0,02“ werden nirgends geführt (die 55 % gehören zu 1,0 σ Gesamtanstieg, β = 1/49).
* **Zählung in der App:** Jedes Risiko-Verdikt nennt „die Jahre 0 bis m−1“, daneben die Variante „ab morgen“ (U3).

## Wo die Annahmen enden

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Jahresmaxima Gumbel mit linearer Lage, σ = 1, unabhängig** (ausgedacht, β = 0 bis 0,04 σ je Jahr) | Alle Zahlen gelten nur in diesem Modell; Kovariaten außer der Zeit, ein Trend in Skala oder ξ, Sprünge und Wochenstruktur sind **nicht gemessen** (Vehikel B steht nur als Tabelle). | – |
| **Der Test kennt ξ = 0** (Gumbel mit gegen ohne Trend) | Die Trennschärfe-Formel gilt für diesen Test; mit freiem ξ ist die Trennschärfe **nicht gemessen**. Die Zeile „Jahre bis 80 %“ gilt nur mit dieser Voraussetzung. | – |
| **Trennschärfe-Formel asymptotisch** | Bei kurzen Reihen ist die Fehlerquote des Tests höher als 5 % (Tabelle in Experiment 1); die Formel weicht von der Simulation in einzelnen Zellen um mehrere Standardfehler ab, ist also ein Band, keine Gleichheit. | – |
| **Unabhängige Jahre** (kein Cluster) | Trend **und** Cluster zugleich ist nicht gemessen; Abhängigkeit allein behandelt die Demo „Spitzen-Cluster“ der Linie. | „Spitzen-Cluster“ (Abhängigkeit) |
| **Block = Jahr** | Trend innerhalb des Blocks und ein Schwellenverfahren mit zeitabhängiger Schwelle sind nicht gemessen. | „Überschreitungen und die GPD“, „Schwellenwahl“ (Schwelle unter Stationarität) |
| **Modellwahl nicht gemessen** | Ob ein Test „Trend ja/nein“ vor der Anpassung den Schaden vermeidet, wird hier nicht gemessen; Experiment 4 vergleicht nur feste Modelle auf denselben Reihen. | „Welches Verfahren wann?“ (Verfahrenswahl) |
| **Vehikel A: Tage unabhängig, Last linear, 256 Kunden je Tag** (Annahmen) | Die Gumbel-Näherung der Rechnung trifft Skala und Lage einzeln nicht; bei wachsender Last (ρ von 0,70 auf 0,80, 0,85 oder 0,90) schätzen alle drei Modelle den Pegel des letzten Tages im Median zu niedrig (Zahlen in Experiment 5). | – |

| Größe | Bedeutung in dieser Demo |
|---|---|
| ξ | Formparameter der GEV: ξ > 0 Fréchet-Typ, ξ = 0 Gumbel, ξ < 0 Weibull-Typ (endlicher Rand; nicht die Weibull-Ausfallverteilung). `scipy.stats.genextreme` hat c = −ξ (nur im Test). Der **Extremwertindex ξ** ist nicht der **Extremalindex θ** aus „Spitzen-Cluster“. |
| Pegel | Wiederkehrwert: Wert zur Überschreitungswahrscheinlichkeit p im Jahr t, x_t(p) = β·t − ln(−ln(1−p)); „100-Jahres-Pegel“ heißt p = 1/100. Die **Wiederkehrperiode** im Jahr t ist 1/p, der Kehrwert der Überschreitungswahrscheinlichkeit dieses Jahres; sie wäre die mittlere Wartezeit (geometrisch) nur, bliebe p stehen, und ist keine Frist. |
| Jahr, Läufe | Ein Jahr ist ein Jahresmaximum (Block). Ein Lauf ist die Wiederholung eines ganzen Experiments (eine Reihe, eine Anpassung); ein Block (Experiment 2) ist eine Nutzungsdauer zu m Jahren. |
| β, D | Trend β in Einheiten σ je Jahr; Gesamtanstieg D = β·(m−1) in σ über die Reihe von m Jahren. |
| Heute | Das letzte Jahr der Reihe, t = m−1. „50 Jahre einschließlich des heutigen“ heißt die Jahre 0 bis 49; „50 Jahre ab morgen“ die Jahre 1 bis 50. |
| Mischung | Die stationäre Verteilung, die entsteht, wenn man ein Jahr der Reihe zufällig zieht: F_mix(x) = (1/m)·Σ_t exp(−e^{−(x−β t)}). Eine stationäre Anpassung liegt grob beim 0,99-Quantil der Mischung; die ML-Anpassung der GEV an die Mischung weicht davon je nach Zelle ab. |
| Design Life Level | Oberes Quantil der Verteilung des Maximums über die Nutzungsdauer (Rootzén/Katz 2013, Definition): Maximum über m Jahre ≤ x mit Wahrscheinlichkeit 1 − q. |
| λ | Verschiebungsparameter der Trennschärfe, λ = β²·m(m²−1)/12 (σ = 1). |

## Ehrliche Grenzen

* **Ausgedacht.** Das Modell, der Trend (β = 0 bis 0,04 σ je Jahr, frei gewählt) und die Last von Vehikel A sind Annahmen. In den gelesenen Quellen gibt es keine reale Reihe von Jahresmaxima, Lasten oder Wartezeiten eines Hafens oder Lagers, an der die Rechnung geprüft wäre; die App behauptet nichts über Nutzen oder Verhältnisse in Logistik, Lager oder Hafen.
* **Eine Reihe ist ein Punkt.** Die Beispielreihe in Experiment 1, 3 und 4 ist eine einzige Ziehung; Aussagen über Erkennbarkeit und Fehler gelten nur als Anteil oder Mittel über Läufe (Zähleinheit in jeder Zeile).
* **Die Formel der Trennschärfe gilt für den Test mit bekanntem ξ = 0** (Gumbel mit gegen ohne Trend); mit freiem ξ ist die Trennschärfe nicht gemessen (eine Stichprobe von 700 Läufen der Aussagenprüfung bei β = 0,02, m = 50 gab denselben Anteil von rund 54 %, aber eine höhere Fehlerquote; nicht in dieser Demo). Wer „Jahre bis 80 %“ in „Welches Verfahren wann?“ übernimmt, muss die Voraussetzung mitnehmen.
* **Nicht erkannt heißt nicht ausgeschlossen:** Unter 80 % Trennschärfe bleibt ein Trend möglich; dass das Trendmodell bei kurzen Reihen schlechter schätzt, gilt für diese Modelle in diesem Modell, nicht für ein Verfahren der Modellwahl (nicht gemessen).
* **Das Trendmodell hilft hier erst spät:** In dieser Messung ist es bei freiem ξ erst bei m = 50 und β = 0,04 oder bei m = 100 und β ≥ 0,01 klar besser; die Messung gilt für dieses Modell und beweist nichts für Betriebsdaten. Ob sie zu Befunden der Literatur passt (zum Beispiel Slater u. a. 2021, nur die Zusammenfassung gelesen), ist nicht geprüft.
* **Die stationäre GEV-Anpassung zielt nur grob auf das Mischungsquantil:** Die ML-Anpassung der GEV an die Mischung liegt in Großstichproben-Rechnungen (400 000 Ziehungen aus der Mischung, scipy, zwei unabhängige Läufe, Streuung bis etwa 0,05 σ; nicht im Test) bei β = 0,02 etwa 0,05 bis 0,09 σ unter dem Mischungsquantil (m = 50 und 100), bei m = 100 und β = 0,04 etwa 0,3 σ darüber. Die „Mitte“ der Mischung ist also nur eine grobe Beschreibung dessen, was die stationäre Anpassung schätzt; der Rest von 0,09 σ in der Verzerrungs-Zerlegung ist zum größeren Teil dieser systematische Abstand.
* **Streuung der Standardfehler:** Die Standardfehler der Quantile von Vehikel A sind aus 20 Teilstichproben geschätzt und bei so wenigen Teilen eher zu klein; die Zufallsanteile der Live-Läufe schwanken deutlich stärker als die eingefrorenen Zahlen.

## 🚧 Bewusst nicht enthalten / nicht gemessen

* Kovariaten außer der Zeit, ein Trend im **Skalenparameter** oder in **ξ** des Gumbel/GEV-Modells (Vehikel A hat einen Skalen-Trend nur über die Last, als Modell (c) mit Trend in ln σ).
* **Profil-Likelihood-Intervalle** unter Trend, **Bootstrap-Überdeckung** unter Trend und die **Abdeckung des Design Life Level** (nicht nur der Formel): nicht gemessen; Experiment 2 prüft nur die Wahrscheinlichkeiten der exakten Formel.
* **Trend innerhalb des Blocks** (Blöcke kürzer als die Zeitskala des Trends) und **Schwellenverfahren unter Trend** (POT mit zeitabhängiger Schwelle): nicht gemessen.
* **Trend und Cluster zugleich** („Spitzen-Cluster“ behandelt die Abhängigkeit allein): nicht gemessen.
* **Periodische** Nichtstationarität (Wochenstruktur, Vehikel B): nur als exakter Pegelfaktor, nicht als Erkennungs- oder Anpassungsmessung.
* **Modellwahl** (erst testen, dann Trendmodell): nicht gemessen; die App vergleicht feste Modelle auf denselben Reihen.
* **Mehrdimensionale Extreme** und CVaR: kein Bezug, nicht enthalten.
* Der Erklär-Expander „💬 In einfachen Worten“ und die Lesehilfe im 📐-Expander stehen in der App (aus der Spezifikation der Erklärseite `trend-spitzen-erklaert` im Website-Repo erzeugt, nicht von Hand ändern); die Seite selbst ist noch nicht veröffentlicht. Kein PDF-Export, kein `modul_frisch.py`; der Footer verweist auf die Linie „Extremwerte erklärt“.
* Gelaufen: die Aussagenprüfung (Neumessungen oben) und die Linux-Prüfung mit Docker (Python 3.12: von 289 Tests 288 bestanden und 1 übersprungen, kein Fehlschlag). Nicht gelaufen: Streamlit Cloud (noch nicht deployt), `claim_review.py`.

## Zahl → Test

Jede Zahl der README steht hier mit ihrer Art: **exakt** (Formel, Bruch, Aufzählung; `tests/test_handbeispiele.py` und die Orakel-Tests rechnen sie auf einem zweiten Weg nach) oder **Messung** (eingefrorener Zufallsanteil mit Zähleinheit; `tests/test_tabellen_und_baender.py` prüft die Zelle gegen die exakte Größe als Band und rechnet einzelne Zellen neu). `tests/test_claims.py` prüft, dass diese Zeilen genau so in der README stehen und aus der Rechnung bzw. der Datei gebildet sind. Alles, was hier nicht steht (Tabellen der Messungen oben, Zeiten), ist **Messung dieser README, nicht im Test** (die Tabellen werden aus `tsp_grid.json` erzeugt, nicht von Hand geschrieben).

| Kennung | Zahl | Art | Beschreibung |
|---|---|---|---|
| E1 | 4,600149 | exakt | Grundhöhe y₁₀₀ = −ln(−ln 0,99) |
| E2 | 4,600 / 5,131 / 5,580 | exakt | Pegel erstes Jahr / Mischung / letztes Jahr bei β = 0,02, m = 50 |
| E3 | 0,45 σ | exakt | Modellfehler der Mischung bei β = 0,02, m = 50 |
| E4 | 57,5 % | exakt | Risiko in 50 Jahren (Jahre 0 bis 49), β = 0,02 |
| E5 | 39,5 % | exakt | Risiko in 50 Jahren ohne Trend |
| E6 | 58,2 % | exakt | Risiko in 50 Jahren ab morgen (Jahre 1 bis 50), β = 0,02 |
| E7 | 100,0 / 82,0 / 60,9 / 37,1 / 22,7 / 14,0 | exakt | Wiederkehrperiode des heutigen Pegels nach 0 / 10 / 25 / 50 / 75 / 100 Jahren, β = 0,02 |
| E8 | 7,41 σ gegen 6,88 σ | exakt | Design Life Level (q = 5 %, 50 Jahre), β = 0,02 gegen ohne Trend |
| E9 | 39,50 % / 47,73 % / 57,47 % / 79,27 % | exakt | Risiko in 50 Jahren für β = 0 / 0,01 / 0,02 / 0,04 |
| E10 | 7,138 / 7,414 / 8,024 | exakt | Design Life Level (q = 5 %, 50 Jahre) für β = 0,01 / 0,02 / 0,04 |
| E11 | 4,165 | exakt | λ bei β = 0,02, m = 50 (= 833/200) |
| E12 | 53,2 % | exakt | Formel-Trennschärfe bei β = 0,02, m = 50 |
| E13 | 54,9 % | exakt | Formel-Trennschärfe bei Gesamtanstieg 1,0 σ (β = 1/49), m = 50 |
| E14 | 156 / 99 / 62 / 39 | exakt | Jahre bis 80 % bei β = 0,005 / 0,01 / 0,02 / 0,04 |
| E15 | 64/315 1/24 8/21 97/35 25/12 | exakt | Handbeispiele H1 und H2 (Brüche) |
| E16 | 5,0051 | exakt | H3: Mischung zweier Jahre bei e^β = 2 |
| E17 | 13,99 % gegen 3,94 % | exakt | H4: Risiko in vier Jahren, Spielzeugtrend gegen ohne Trend |
| E18 | 1; −0,4228; 1,8237 | exakt | Fisher-Information der Gumbel-Verteilung (μ,μ; μ,σ; σ,σ) |
| E19 | 9,27 / 9,94 / 10,52 / 10,59 / 8,35 | exakt | Lage der Gumbel-Näherung des Tagesmaximums, ρ = 0,70 bis 0,90 |
| E20 | 24,6 / 28,3 / 33,5 / 41,3 / 54,3 | exakt | 100-Tage-Pegel der Gumbel-Näherung, ρ = 0,70 bis 0,90 |
| E21 | 24,7 Jahre | exakt | mittlere Wartezeit auf eine Überschreitung des heutigen Pegels ab Jahr 50, wenn der Trend weiterläuft (β = 0,02, Jahr 50 mitgezählt); der Kehrwert 37,1 gilt nur bei gleichbleibender Wahrscheinlichkeit |
| M1 | 10,1 % / 6,4 % / 6,4 % / 5,8 % / 5,9 % | Messung | Fehlerquote erster Art bei m = 10 / 20 / 30 / 50 / 100 Jahren (R = 4 000 Läufe) |
| M2 | 53,2 % | Messung | Trennschärfe-Simulation bei β = 0,02, m = 50 (R = 4 000 Läufe) |
| M3 | 80,5 % / 80,3 % / 79,8 % / 76,6 % | Messung | Simulation beim kleinsten m mit Formel ≥ 80 % (R = 2 000 Läufe) |
| M4 | −0,54 σ | Messung | Verzerrung der stationären GEV, m = 50, β = 0,02 (R = 3 000 Läufe) |
| M5 | Standardfehler 0,02 | Messung | Standardfehler dieser Verzerrung |
| M6 | 83,5 % / 75,2 % / 53,8 % / 33,4 % | Messung | Abdeckung des Delta-Intervalls (letztes Jahr), vier Zellen (R = 1 000 Läufe) |
| M7 | 4 klar besser, 7 klar schlechter, 1 im Rauschen | Messung | Trendmodell gegen stationär bei gleichem freiem ξ, 12 Zellen (R = 400 Läufe) |
| M8 | 0,80 gegen 1,18 σ | Messung | RMSE mit Trend gegen stationär, m = 100, β = 0,02 |
| M9 | ×2,46 und ×1,79 | Messung | Vehikel A: Skala und Lage der Tagesmaxima, ρ = 0,70 auf 0,90 (400 000 Tage je Last) |
| M10 | 0,72 / 0,71 / 0,78; 1,07 | Messung | Vehikel A: Median Pegel-Dach/wahr der drei Modelle und der Kontrolle (R = 300 Läufe) |
| M11 | 253,36 / 342,00 | Messung | Vehikel B: Pegel je Container nur Werktage / nur Wochenende (exakt) |
| M12 | 1,35 | Messung | Vehikel B: Faktor Wochenende gegen Werktage |
| M13 | 400 000 | Messung | Blöcke der eingefrorenen Simulation (Experiment 2) |

## Quellen

* **Rootzén/Katz 2013** (Design Life Level, Water Resour. Res. 49): **gelesen** (lokal); benutzt wird nur die Definition (oberes Quantil der Verteilung des Maximums über die Nutzungsdauer, Test `tests/test_literatur_trend.py`); das Beispiel der Einleitung (heute 100-jährlich, später etwa alle 90 bzw. 70 Jahre) gehört zu einem anderen Modell und wird hier nicht nachgerechnet.
* **Coles 2001**, Kap. 6 (nichtstationäre Modelle, Lage-Trend, Kovariaten) und Kap. 3 (Pegel): **gelesen** (nur als Befund benutzt, nichts wörtlich).
* **Slater u. a. 2021** (HESS 25): nicht im Original gelesen (Zusammenfassung); die Aussagenprüfung fand eine Aussage „Trendmodell hilft erst spät“ in den eingesehenen Teilen nicht wörtlich, deshalb wird sie hier nicht als Beobachtung der Quelle geführt.
* **Serinaldi/Kilsby 2015**: nur ein Auszug gelesen; **Montanari/Koutsoyiannis 2014**: nur Abstract; beide als Gegenpositionen genannt, offener Streit, kein Ergebnis dieser Linie. **Smith 1989**: nur Abstract. **Davison/Smith 1990**: nicht gelesen.
* **Extremalindex θ = (1−ρ)²** der Näherung von Vehikel A: aus der Demo „Spitzen-Cluster“ der Linie zitiert, dort mit Quelle auf **Stufe A** (Peshkova nur Kurzseite, Hooghiemstra/Meester nicht gelesen); in dieser Demo nicht neu hergeleitet.
* Keine Neuheit: alles Lehrbuch und Stand der Technik; neu ist nur die Vorführbarkeit mit Zähleinheit und die exakten Gegenproben.

## Tests

**289 Tests**, alle grün; die ganze Suite lief seriell in 608 s (eine Messung, Windows, Python 3.14, Rechner durch andere Prozesse belegt; auf einem ruhigen Rechner kürzer, nicht gemessen).

* 26 Handbeispiel-Tests (`tests/test_handbeispiele.py`, H1 bis H6 als exakte Prüfungen): Brüche 64/315, 1/24, 8/21, 97/35, 25/12, Quadratik der Mischung, Risiko 1 − 0,99¹⁵, Design Life Level, λ = 833/200, Skalen 10/3 bis 10; Gegenproben per Aufzählung aller 24 Ordnungen, Quadratur und Simulation.
* 75 Orakel-Tests (scipy nur dort): Gumbel-Anpassung gegen Nelder-Mead und `gumbel_r` von scipy, Gradient und Hesse-Matrix gegen Differenzen, Fisher-Information gegen Quadratur (`tests/test_oracle_gumbel_fit.py`); GEV-Likelihood gegen `genextreme.logpdf` mit c = −ξ, Optimum, Pegel, Delta-Intervall gegen eine unabhängige Hesse-Matrix (`tests/test_oracle_gev_trend.py`); Mischungsquantil gegen `brentq` und gegen 10⁶ gezogene Maxima, Risiko, Design Life Level und Wiederkehrperiode gegen das Produkt der Jahresverteilungen, mittlere Wartezeit bei weiter steigendem Trend gegen das Produkt und gegen eine Simulation (`tests/test_oracle_mixture.py`); Lindley-Rekursion gegen die exakte Randverteilung und eine Schleifenfassung mit denselben Zufallszahlen, Vehikel-B-Tabelle gegen eine eigene Aufzählung (`tests/test_oracle_vehicle.py`).
* 6 Literatur-Tests (`tests/test_literatur_trend.py`, nur die Definition des Design Life Level), zusätzlich zu den 75 Orakel-Tests.
* 27 Eigenschafts- und Randfalltests (20 bis 50 Zufallsinstanzen mit festem Zufalls-Seed): Lage-Äquivarianz, Skalierung, Zeitumkehr, Monotonie von Risiko, Design Life Level und Trennschärfe, m = 1, 2, 3, β < 0, β = 0, sehr großes β, q und p an den Rändern, Reihe ohne Streuung.
* 9 Zufalls- und Szenariotests (vektorisierter Strom bitgleich zur Vorlage, abgeleitete Zufalls-Seeds, festgehaltene Reihe); 24 Tabellentests (Aufbau, Bänder, Zählungen über das ganze Raster, Neu-Rechnen einzelner Zellen als Band); 4 Kopiertests; 41 Claim-Tests; 25 Verdikttests und 11 Preset-Tests; 5 Abbildungstests.
* Oberfläche: 3 AppTest-Rauchtests, 12 Regler- und Preset-Tests (jeder Wert jedes Reglers, jedes Preset mit seiner Verdikt-Zeile, Permalink), 18 Layouttests, 2 Footertests, 1 Test der Laufzeit-Importe. `tests/test_app.py` ist die unveränderte Portfolio-Vorlage.
* Plattform: Zufallsanteile nur als Band, kein Test auf das letzte Bit; Gleitkomma-Toleranzen relativ 1e-9.
* Nicht im Repo: das Scratch-Orakel mit 200 Zufallsinstanzen (nur die kleine, schnelle Fassung steht in den Tests).

### Live-Rechenzeiten (dieser Rechner; Streamlit Cloud ist langsamer, nicht gemessen)

Eine Reihe samt Test und stationärer GEV mit Delta-Intervall: 0,02 bis 0,03 s (m = 50 bis 150). 400 Blöcke zu 10 / 50 / 100 Jahren: 0,4 / 1,6 / 3,8 s. Eigene Läufe (drei Anpassungen je Lauf): 20 Läufe bei m = 20 bis 50 rund 1,0 s, 60 Läufe bei m = 100 bis 150 rund 3,5 bis 4,0 s. Wochenpegel der kopierten Tabelle: 0,05 s. Obergrenzen der Regler: Nutzungsdauer ≤ 100 Jahre, eigene Läufe ≤ 60, Reihe ≤ 150 Jahre. Ein erster Aufruf der App mit allen Experimenten braucht etwa 3 s, jeder Preset-Wechsel etwa 2 s (AppTest). Nachmessung bei der Korrektur (Rechnerlast nicht kontrolliert): Reihe 0,02 s, 400 Blöcke 0,3 / 1,5 / 3,0 s, 20 Läufe 0,7 s, 60 Läufe 2,6 bis 3,4 s, Wochenpegel 0,06 s. Eine Messung der Aussagenprüfung auf einem belegten Rechner gab 1,2- bis 3,3-mal längere Zeiten; die Angaben gelten also für einen wenig belegten Rechner.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App (fünf Experimente, Kernfragen, Annahmen, Formeln) |
| `tsp_models.py` | Kern: exakte Größen des Gumbel-Trend-Modells, Gumbel-Newton, GEV-Nelder-Mead, Delta-Intervall, Trennschärfe |
| `tsp_numeric.py` | Nelder-Mead, Brent, Normalverteilung, Wilson-Intervall (ohne scipy) |
| `tsp_scenario.py` | Reihen und Blöcke aus der SplitMix64-Vorlage, abgeleitete Zufalls-Seeds |
| `tsp_rng.py` | SplitMix64-Vorlage, unverändert kopiert |
| `tsp_vecrng.py` | vektorisierter Strom (bitgleich), nur für die eingefrorenen Tabellen |
| `tsp_vehicle.py` | Vehikel A: M/M/1-Wartezeit, Lindley-Rekursion, Gumbel-Näherung |
| `tsp_vehicle_b.py` | Vehikel B: Überlebensfunktion der kopierten Gewichtstabelle |
| `tsp_evaluation.py` | Einstellungen, live gerechnete Größen, Verdikte als reine Funktionen |
| `tsp_reference.py` | Zugriff auf die eingefrorenen Rechnungen |
| `tsp_grid.json` | eingefrorene Rechnungen (Quelle und Stand je Teil) |
| `tsp_mixture.json`, `tsp_mixture_quelle.json` | gemeinsame Gewichtstabelle der Linie (unverändert kopiert), Quelle und Stand je Feld |
| `tsp_constants.py`, `tsp_presets.py`, `tsp_layout.py`, `tsp_texts.py`, `tsp_visualization.py` | Regler und Presets, Permalink, fester Kopf, Texte, Abbildungen |
| `tools/tsp_make_grid.py` | erzeugt `tsp_grid.json` (6 Prozesse, rund 2 Minuten), `--check` rechnet vier kleine Zellen neu |
| `tests/` | siehe Abschnitt „Tests“; `tests/claims_fragmente.py` bildet die Zahlen der Tabelle „Zahl → Test“ |
| `requirements.txt`, `requirements-dev.txt` | Laufzeit (streamlit, numpy, plotly, pandas) und Tests (pytest, scipy) |

## Bewusst nicht umgesetzt

Siehe den Abschnitt „🚧 Bewusst nicht enthalten / nicht gemessen“ oben: Kovariaten, Trend in Skala und ξ, Profil-Likelihood und Bootstrap unter Trend, Schwellenverfahren unter Trend, Trend plus Cluster, Modellwahl.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py --server.port 10090      # Port 10090 für die lokale Vorschau
```

Die eingefrorenen Tabellen neu erzeugen: `python tools/tsp_make_grid.py` (nur nötig, wenn sich Kern oder Raster ändern), prüfen: `python tools/tsp_make_grid.py --check`.

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, pandas und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Extremwerte: Blöcke, Schwelle, Pegel](https://sebastianhanisch.net/konzepte-extremwert.html).
