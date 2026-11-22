"""Zahlenformat (deutsches Dezimalkomma) und die festen Texte der App."""

import math


def de(x, nd=2):
    """Zahl mit Dezimalkomma und nd Nachkommastellen (U+2212 als Minuszeichen)."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "–"
    s = f"{x:.{nd}f}".replace(".", ",")
    if s.startswith("-") and float(x) != 0 and float(s.replace(",", ".")) != 0:
        s = "−" + s[1:]
    elif s.startswith("-"):
        s = s[1:]
    return s


def pct(p, nd=0):
    """Anteil 0..1 als Prozent mit Dezimalkomma."""
    return de(100 * p, nd) + " %"


def pint(n):
    """Ganzzahl mit Leerzeichen als Tausendertrenner."""
    return f"{int(n):,}".replace(",", " ")


INTRO = """
Die Jahresmaxima einer Reihe haben oft eine **Lage, die mit der Zeit wandert**. Dann ist „der 100-Jahres-Pegel“ keine Zahl mehr, sondern eine **Frage nach dem Jahr**: Der Pegel von **heute** (dem letzten Jahr der Reihe) liegt höher als der
Pegel, den eine **stationäre Anpassung an die ganze Reihe** liefert, und die **Wiederkehrperiode** (Kehrwert der Überschreitungswahrscheinlichkeit des Jahres, keine Frist) schrumpft von Jahr zu Jahr. Die Demo rechnet das in einem **ausgedachten Modell** (Jahresmaxima Gumbel mit Lage
μ_t = β·t, σ = 1, unabhängig; keine Daten) mit **exakten Gegenproben** durch und zählt jede Zahl in **Jahren** oder **Läufen**. Vier Fragen: Sieht man den Trend, was heißt „100 Jahre“, was liefert eine stationäre Anpassung, hilft das Trendmodell? Dazu eine
fünfte Seite: eine wachsende **Last** (M/M/1-Tor) statt der Zeit. Die Zahlen sind Lehrbuch/Abstract in einem Modell und sagen nichts über Klima, Last, Lager oder Hafen.

Weiter unten: die Experimente **🔬 1 bis 5**, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**. Ohne Formeln erklärt es der Expander „💬 In einfachen Worten“ gleich darunter.
"""

HOW = """
1. **Reihe ziehen.** Jahresmaxima x_t = β·t + G_t für t = 0 … m−1 mit Standard-Gumbel-Rauschen G_t (σ = 1, unabhängig); „heute“ ist das letzte Jahr t = m−1.
2. **Trend prüfen.** Likelihood-Quotienten-Test: Gumbel **mit** gegen **ohne** Lage-Trend (ξ = 0 bekannt, Skala frei), Schwelle χ²₁ = 3,84 (Niveau 5 %). Die Trennschärfe hat eine geschlossene Formel, die gegen Simulationen geprüft ist.
3. **Pegel von heute vergleichen.** Der wahre 100-Jahres-Pegel des letzten Jahres ist β·(m−1) + 4,60 (exakt). Eine stationäre Anpassung liegt grob beim Pegel der **Mischung** aller Jahre; ein Trendmodell schätzt den des letzten Jahres, mit mehr Parametern.
"""

ASSUMPTIONS = """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Jahresmaxima Gumbel mit linearer Lage, σ = 1, unabhängig** (ausgedacht, β = 0 bis 0,04 σ je Jahr) | Alle Zahlen gelten nur in diesem Modell; Kovariaten außer der Zeit, ein Trend in Skala oder ξ, Sprünge und Wochenstruktur sind **nicht gemessen** (Vehikel B steht nur als Tabelle). | – |
| **Der Test kennt ξ = 0** (Gumbel mit gegen ohne Trend) | Die Trennschärfe-Formel gilt für diesen Test; mit freiem ξ ist die Trennschärfe **nicht gemessen**. Die Zeile „Jahre bis 80 %“ gilt nur mit dieser Voraussetzung. | – |
| **Trennschärfe-Formel asymptotisch** | Bei kurzen Reihen ist die Fehlerquote des Tests höher als 5 % (Tabelle in Experiment 1); die Formel weicht von der Simulation in einzelnen Zellen um mehrere Standardfehler ab, ist also ein Band, keine Gleichheit. | – |
| **Unabhängige Jahre** (kein Cluster) | Trend **und** Cluster zugleich ist nicht gemessen; Abhängigkeit allein behandelt die Demo „Spitzen-Cluster“ der Linie. | „Spitzen-Cluster“ (Abhängigkeit) |
| **Block = Jahr** | Trend innerhalb des Blocks und ein Schwellenverfahren mit zeitabhängiger Schwelle sind nicht gemessen. | „Überschreitungen und die GPD“, „Schwellenwahl“ (Schwelle unter Stationarität) |
| **Modellwahl nicht gemessen** | Ob ein Test „Trend ja/nein“ vor der Anpassung den Schaden vermeidet, wird hier nicht gemessen; Experiment 4 vergleicht nur feste Modelle auf denselben Reihen. | „Welches Verfahren wann?“ (Verfahrenswahl) |
| **Vehikel A: Tage unabhängig, Last linear, 256 Kunden je Tag** (Annahmen) | Die Gumbel-Näherung der Rechnung trifft Skala und Lage einzeln nicht; bei wachsender Last (ρ von 0,70 auf 0,80, 0,85 oder 0,90) schätzen alle drei Modelle den Pegel des letzten Tages im Median zu niedrig (Zahlen in Experiment 5). | – |
"""

CONVENTIONS = """
| Größe | Bedeutung in dieser Demo |
|---|---|
| ξ | Formparameter der GEV: ξ > 0 Fréchet-Typ, ξ = 0 Gumbel, ξ < 0 Weibull-Typ (endlicher Rand; nicht die Weibull-Ausfallverteilung). `scipy.stats.genextreme` hat c = −ξ (nur im Test). Der **Extremwertindex ξ** ist nicht der **Extremalindex θ** aus „Spitzen-Cluster“. |
| Pegel | Wiederkehrwert (englisch Return Level): Wert zur Überschreitungswahrscheinlichkeit p im Jahr t, x_t(p) = β·t − ln(−ln(1−p)); „100-Jahres-Pegel“ heißt p = 1/100. Die **Wiederkehrperiode** im Jahr t ist 1/p, der Kehrwert der Überschreitungswahrscheinlichkeit dieses Jahres; sie wäre die mittlere Wartezeit (geometrisch) nur, bliebe p stehen, und ist keine Frist. |
| Jahr, Läufe | Ein Jahr ist ein Jahresmaximum (Block). Ein Lauf ist die Wiederholung eines ganzen Experiments (eine Reihe, eine Anpassung); ein Block (Experiment 2) ist eine Nutzungsdauer zu m Jahren. |
| β, D | Trend β in Einheiten σ je Jahr; Gesamtanstieg D = β·(m−1) in σ über die Reihe von m Jahren. |
| Heute | Das letzte Jahr der Reihe, t = m−1. „50 Jahre einschließlich des heutigen“ heißt die Jahre 0 bis 49; „50 Jahre ab morgen“ die Jahre 1 bis 50. |
| Mischung | Die stationäre Verteilung, die entsteht, wenn man ein Jahr der Reihe zufällig zieht: F_mix(x) = (1/m)·Σ_t exp(−e^{−(x−β t)}). Eine stationäre Anpassung liegt grob beim 0,99-Quantil der Mischung; die ML-Anpassung der GEV an die Mischung weicht davon je nach Zelle ab. |
| Design Life Level | Oberes Quantil der Verteilung des Maximums über die Nutzungsdauer (Rootzén/Katz 2013, Definition): Maximum über m Jahre ≤ x mit Wahrscheinlichkeit 1 − q. |
| λ | Verschiebungsparameter der Trennschärfe, λ = β²·m(m²−1)/12 (σ = 1). |
"""

VEHIKEL_TEXT = (
    "Die Reihen sind ausgedacht; in den gelesenen Quellen gibt es keine reale Reihe von Jahresmaxima, Lasten oder Wartezeiten eines Hafens oder Lagers, an der die Rechnung geprüft wäre. "
    "Der Trend ist eine angenommene Lageverschiebung der Jahresmaxima; was die Zahlen für ein echtes Klima, eine echte Last oder einen echten Betrieb heißen, wird hier nicht behauptet. "
    "Das Gate-Modell (Warteplätze, harte Obergrenze der Länge) ist kein Vehikel, die Gate-Wartezeit mit unendlichem Puffer (M/M/1) schon."
)

FORMULAS = r"""
**Modell.** Jahresmaxima $X_t = \beta\, t + G_t$ für $t = 0,\dots,m-1$ mit unabhängigen Standard-Gumbel-Werten $G_t$ (Verteilungsfunktion $\exp(-e^{-g})$). Der **Pegel** des Jahres $t$ zur Überschreitungswahrscheinlichkeit $p$ ist

$$x_t(p) = \beta\, t - \ln\bigl(-\ln(1-p)\bigr), \qquad y_{100} = -\ln(-\ln 0{,}99) = 4{,}600149 .$$

**Mischung** (der Pegel, um den eine stationäre Anpassung grob liegt):

$$F_{\mathrm{mix}}(x) = \frac1m \sum_{t=0}^{m-1} \exp\bigl(-e^{-(x-\beta t)}\bigr), \qquad F_{\mathrm{mix}}(x_{\mathrm{mix}}) = 0{,}99 .$$

**Maximum über die Nutzungsdauer, Risiko und Design Life Level.** Mit $S = \sum_{t=0}^{m-1} e^{\beta t}$:

$$P\bigl(\max_t X_t \le x\bigr) = \exp\bigl(-e^{-x}\, S\bigr), \qquad \text{Risiko} = 1 - 0{,}99^{S}, \qquad x_q = \ln S - \ln\bigl(-\ln(1-q)\bigr) .$$

**Wiederkehrperiode des heutigen Pegels** im Jahr $t$ nach heute: $T_t = 1 / \bigl(1 - 0{,}99^{\,e^{\beta t}}\bigr)$, der Kehrwert der Überschreitungswahrscheinlichkeit dieses Jahres; als mittlere Wartezeit gilt $T_t$ nur, bliebe diese Wahrscheinlichkeit stehen.

**Trendtest.** Likelihood-Quotient der Gumbel-Anpassung mit gegen ohne Lage-Trend (ξ = 0 bekannt), Schwelle $\chi^2_1 = 3{,}84$. Mit zentrierter Zeit $c_t = t - (m-1)/2$ ist die Information für $\beta$ gleich $\sum c_t^2 = m(m^2-1)/12$ (bei $\sigma = 1$, orthogonal zu den anderen Parametern), also

$$\lambda = \beta^2\,\frac{m(m^2-1)}{12}, \qquad \text{Trennschärfe} \approx \Phi\bigl(\sqrt\lambda - 1{,}96\bigr) + \Phi\bigl(-\sqrt\lambda - 1{,}96\bigr)$$

(asymptotisch, nicht zentrales $\chi^2$). **Fisher-Information** der Gumbel-Verteilung $(\mu,\sigma)$ bei $\sigma=1$: $\bigl(1;\ -(1-\gamma);\ (1-\gamma)^2 + \pi^2/6\bigr)$.

**Rekorde.** Mit $E_t = e^{-X_t} \sim \mathrm{Exp}(\lambda_t)$, $\lambda_t = e^{\beta t}$, ist das Jahr $k$ ein neuer Höchstwert mit Wahrscheinlichkeit $\lambda_k / \sum_{i \le k} \lambda_i$; ohne Trend ist die erwartete Zahl der Rekorde $H_m = \sum_{k=1}^m 1/k$.

**Anpassungen.** Gumbel mit Trend: Newton-Verfahren auf $\ell = -m\ln\sigma - \sum (z_t + e^{-z_t})$, $z_t = (x_t - a - b\,c_t)/\sigma$. GEV (ξ frei, $\xi \ge -1$, Lage $a + b\,c_t$): Nelder-Mead auf der negativen Log-Likelihood
$m \ln\sigma + (1 + 1/\xi)\sum \ln(1 + \xi z_t) + \sum (1 + \xi z_t)^{-1/\xi}$.

**Vehikel A.** M/M/1-Wartezeit $W$ (Dienstrate 1, Ankunftsrate $\rho$), Gleichgewicht: $P(W > x) = \rho\, e^{-(1-\rho)x}$. Gumbel-Näherung des Tagesmaximums aus $n = 256$ Kunden (Rechnung): Skala $1/(1-\rho)$, Lage $\ln(n\,\theta\,\rho)/(1-\rho)$ mit $\theta = (1-\rho)^2$ (Quelle: Demo „Spitzen-Cluster“ der Linie).

*Skizze, nicht vollständig:* die Herleitung der Trennschärfe-Formel (Likelihood-Quotienten-Test, nicht zentrales $\chi^2$) ist nur angedeutet; die ML-Anpassungen sind als Verfahren genannt, nicht ausgeführt.
"""
