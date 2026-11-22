"""Trend: der Pegel von heute, nicht der Pegel der Reihe - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Demo der Linie „Extremwerte erklärt“: Jahresmaxima Gumbel mit Lage-Trend (ausgedacht): sieht man den Trend, was heißt „100 Jahre“, was liefert eine stationäre Anpassung, hilft das Trendmodell?
Rechnet nur mit numpy und der Standardbibliothek (kein scipy); die schweren Läufe sind eingefroren (tsp_grid.json, Skript tools/tsp_make_grid.py).

Lauffähig mit: streamlit run app.py
"""

import time

import pandas as pd
import streamlit as st

import tsp_constants as C
import tsp_evaluation as EV
import tsp_models as M
import tsp_reference as REF
import tsp_texts as T
import tsp_vehicle_b as VB
import tsp_visualization as V
from tsp_layout import dunkles_design, kopf_css
from tsp_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from tsp_texts import de, pct, pint

st.set_page_config(page_title="Trend – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return EV.analyse(settings)


@st.cache_data(show_spinner=False)
def _blocks(settings):
    t0 = time.perf_counter()
    r = EV.live_blocks(settings)
    return r, time.perf_counter() - t0


@st.cache_data(show_spinner=False)
def _runs(settings):
    t0 = time.perf_counter()
    r = EV.live_runs(settings)
    return r, time.perf_counter() - t0


@st.cache_data(show_spinner=False)
def _week_all():
    return VB.level(1e-5)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in tsp_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("📈 Trend: der Pegel von heute, nicht der Pegel der Reihe")
st.markdown(T.INTRO)

# erklaerseite:einfach:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/trend-spitzen-erklaert.json, nicht von Hand ändern)
with st.expander("💬 In einfachen Worten"):
    st.markdown(
        """
Wenn die Höchstwerte eines Jahres im Lauf der Zeit langsam steigen, ist der Wert, der nur einmal in hundert Jahren erreicht wird, keine feste Zahl mehr. Er hängt davon ab, für welches Jahr man fragt: Wer alle Jahre einer Reihe zusammen auswertet, erhält grob den Wert der mittleren Jahre, nicht den von heute. Die Demo rechnet das in einem ausgedachten Modell exakt nach und zählt jede Zahl in Jahren oder Läufen. Sie zeigt außerdem, wie schwer ein solcher Anstieg in einer kurzen Reihe zu erkennen ist und dass ein Modell mit Anstieg dort oft schlechter schätzt als eines ohne. Alle Reihen sind ausgedacht und sagen nichts über Klima, Lager oder Hafen.

[Ausführlich und verständlich erklärt, mit Fachteil](https://sebastianhanisch.net/trend-spitzen-erklaert.html)
        """
    )
# erklaerseite:einfach:end

with st.expander("So funktioniert die Trendprüfung", expanded=True):
    st.markdown(T.HOW)

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Die Reihe (Experimente 1, 3 und 4)**")
    m = st.select_slider("Jahre m", C.M_OPTIONS, key="m_sl", help="Länge der Reihe in Jahren (ein Jahresmaximum je Jahr); „heute“ ist das letzte Jahr. Eingefroren sind die Zellen für 20, 50 und 100 Jahre.")
    beta = st.select_slider("Trend β (σ je Jahr)", C.BETA_OPTIONS, key="beta_sl", format_func=lambda v: de(v, 2),
                            help="Anstieg der Lage je Jahr in Einheiten σ; 0 heißt kein Trend. Der Gesamtanstieg ist β·(m−1).")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1, help="Legt die gezogene Reihe (und die Läufe und Blöcke) fest; derselbe Zufalls-Seed gibt dieselben Ziehungen (Ganzzahlarithmetik, auf Linux nicht geprüft).")
    st.button("🎲 Neue Reihe ziehen", width="stretch", on_click=randomize_seed)
    st.markdown("**Nutzungsdauer und Risiko (Experiment 2)**")
    nutz = st.select_slider("Nutzungsdauer (Jahre)", C.NUTZ_OPTIONS, key="nutz_sl", help="Länge der Nutzungsdauer, über die das Risiko und der Design Life Level gerechnet werden; auch die Länge der Blöcke der Simulation.")
    q = st.select_slider("Risiko q des Design Life Level", C.Q_OPTIONS, key="q_sl", format_func=lambda v: pct(v), help="Der Design Life Level wird mit dieser Wahrscheinlichkeit in der Nutzungsdauer überschritten.")
    st.markdown("**Eigene Läufe (Experiment 4)**")
    runs = st.slider("Läufe (live)", C.RUNS_MIN, C.RUNS_MAX, key="runs_sl", step=C.RUNS_STEP,
                     help="Unabhängige Wiederholungen in der gewählten Zelle (m, β). Die eingefrorene Tabelle hat 400 Läufe je Zelle; live ist die Zufallsschwankung deutlich größer.")
    st.markdown("**Last statt Zeit (Experiment 5)**")
    last_end = st.select_slider("Last am Ende (ρ)", C.LOAD_END_OPTIONS, key="last_sel", format_func=lambda v: de(v, 2),
                                help="Die Last des M/M/1-Tores wächst linear von ρ = 0,70 auf diesen Wert (100 Tage); eingefroren sind drei Endlasten.")
    modell = st.selectbox("Modell", C.MODELS, key="modell_sel", help="Welches Modell in Experiment 5 hervorgehoben wird: stationäre GEV, GEV mit Lage-Trend oder Gumbel mit Trend in Lage und Skala.")

sync_query_params({k: st.session_state[k] for k in ("m_sl", "beta_sl", "seed_input", "nutz_sl", "q_sl", "runs_sl", "last_sel", "modell_sel")})

settings = EV.Settings(int(m), float(beta), int(seed), int(nutz), float(q), int(runs), float(last_end), str(modell))

# --- Experiment 1: Sieht man den Trend? -----------------------------------------------------------------------------------------------------

a = _analysis(settings)
st.markdown("## 🎯 Experiment 1: Sieht man den Trend?")
st.caption(f"Eine gezogene Reihe aus {settings.m} Jahresmaxima, Lage β·t mit β = {de(settings.beta, 2)} σ je Jahr (Gesamtanstieg {de(settings.beta * (settings.m - 1), 2)} σ), Standard-Gumbel-Rauschen, unabhängig. "
           "Der Test vergleicht Gumbel mit gegen ohne Lage-Trend (ξ = 0 bekannt, Skala frei).")
c1, c2, c3, c4 = st.columns(4)
c1.metric("β̂ (Trend je Jahr)", de(a.trend.b, 3), help="Geschätzter Trend dieser einen Reihe in σ je Jahr; das wahre β steht in der Seitenleiste.")
c2.metric("Likelihood-Quotient", de(a.lr, 2), delta=("über der Schwelle 3,84" if a.detected else "unter der Schwelle 3,84"), delta_color="off",
          help="2·(ℓ mit Trend − ℓ ohne Trend); gegen die χ²-Schwelle 3,84 (Niveau 5 %).")
c3.metric("Neue Höchstwerte (Rekorde)", f"{a.records}", help=f"Zahl der Jahre, die alle früheren übertreffen. Ohne Trend im Mittel H_m = {de(a.records_expected_no_trend, 2)}, mit diesem β {de(a.records_expected_trend, 2)} (exakt).")
c4.metric("Der Test sagt", "Trend" if a.detected else "kein Trend", help="Nur für diese eine Reihe; die Trennschärfe über viele Läufe steht darunter.")
st.plotly_chart(V.fig_series(a.x, settings.beta, a.stat, a.trend), width="stretch", key="series_chart")
if settings.beta == 0:
    st.caption("Die Reihe hat keinen Trend: meldet der Test „Trend“, ist das ein Fehlalarm (Soll 5 % der Läufe)." if a.detected else "Die Reihe hat keinen Trend, und der Test meldet keinen.")
else:
    st.caption("Die Reihe hat einen Trend; der Test hat ihn in dieser einen Reihe erkannt, was über viele Läufe nur mit der Trennschärfe unten geschieht." if a.detected
               else "Die Reihe hat einen Trend, den der Test in dieser einen Reihe nicht erkannt hat: Nicht erkannt heißt nicht ausgeschlossen.")

cells = [REF.power_cell(settings.beta, mm) for mm in C.M_FROZEN] if settings.beta > 0 else [REF.typ1(mm) for mm in C.M_TYP1]
st.plotly_chart(V.fig_power(settings.beta, [c for c in cells if c], settings.m), width="stretch", key="power_chart")
st.caption("Formel (Linie): Φ(√λ − 1,96) + Φ(−√λ − 1,96) mit λ = β²·m(m²−1)/12, asymptotisch und nur für den Test mit bekanntem ξ = 0. Punkte: eingefrorene Simulation (R = 4 000 Läufe je Zelle). "
           "Die senkrechte Linie ist die gewählte Länge.")
rows = []
for b_ in C.STEPS_YEARS80:
    c = REF.years80(b_)
    rows.append({"β (σ je Jahr)": de(b_, 3), "Jahre bis 80 % (Formel)": c["m"], "Formel": pct(c["formel"], 1), f"Simulation (R = {pint(c['R'])} Läufe)": pct(c["hits"] / c["R"], 1)})
t1_rows = [{"Jahre m": c["m"], "Fehlerquote bei β = 0": pct(c["hits"] / c["R"], 1), "Soll": "5 %", "Läufe": pint(c["R"])} for c in (REF.typ1(mm) for mm in C.M_TYP1)]
g1, g2 = st.columns(2)
with g1:
    st.markdown("**Jahre bis 80 % Trennschärfe** (kleinstes m mit Formel ≥ 0,8; Test mit bekanntem ξ = 0)")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
with g2:
    st.markdown("**Fehlerquote erster Art** (β = 0; der Test ist bei kurzen Reihen zu großzügig)")
    st.dataframe(pd.DataFrame(t1_rows), hide_index=True, width="stretch")
st.caption(f"Rekorde in dieser Reihe: {a.records} neue Höchstwerte gegen erwartet {de(a.records_expected_no_trend, 2)} ohne und {de(a.records_expected_trend, 2)} mit diesem Trend (nur Anzeige, kein Befund über Daten). "
           f"Quelle der Tabellen: {REF.meta('power')['quelle']}.")

st.markdown("## ❓ Kernfragen")
st.markdown(
    "1. **Sieht man den Trend?** (hier)\n"
    "2. **Was heißt „100 Jahre“, wenn die Lage wandert?** (Experiment 2)\n"
    "3. **Was liefert eine stationäre Anpassung an die ganze Reihe?** (Experiment 3)\n"
    "4. **Hilft das Trendmodell?** (Experiment 4)"
)
kind, text = EV.verdict_trend(settings.m, settings.beta)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 2: Was heißt „100 Jahre“? -----------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Was heißt „100 Jahre“ bei Trend?")
rt = EV.risk_table(settings.beta, settings.nutz, settings.q)
st.caption(f"Alles exakt (Gumbel mit Lage β·t, β = {de(settings.beta, 2)}): der **heutige** 100-Jahres-Pegel ist {de(M.Y100, 2)} σ über der Lage des heutigen Jahres; hier als Wert von „Jahr 0“ der Zählung gerechnet. "
           "Die Wiederkehrperiode im Jahr t ist der Kehrwert der Überschreitungswahrscheinlichkeit dieses Jahres; als mittlere Wartezeit gilt sie nur, bliebe diese Wahrscheinlichkeit stehen (bei weiter steigendem Trend ist die Wartezeit kürzer), und sie ist keine Frist.")
e1, e2, e3, e4 = st.columns(4)
e1.metric(f"Risiko in {settings.nutz} Jahren (Jahre 0 bis {settings.nutz - 1})", pct(rt.risk, 1), delta=f"ohne Trend {pct(rt.risk_stationary, 1)}", delta_color="off",
          help="Wahrscheinlichkeit, dass der heutige 100-Jahres-Pegel in der Nutzungsdauer mindestens einmal überschritten wird; das heutige Jahr ist eingeschlossen.")
e2.metric(f"Risiko, {settings.nutz} Jahre ab morgen (Jahre 1 bis {settings.nutz})", pct(rt.risk_ahead, 1), help="Gleiche Rechnung für die Jahre 1 bis m: das heutige Jahr zählt nicht mit.")
e3.metric(f"Design Life Level (Risiko {pct(settings.q)})", de(rt.dll, 2), delta=f"{de(rt.dll - rt.dll_stationary, 2)} gegen ohne Trend", delta_color="off",
          help="Oberes Quantil der Verteilung des Maximums über die Nutzungsdauer (Rootzén/Katz 2013, Definition). Das Delta ist Wert minus stationärer Wert.")
e4.metric("Stationärer Pegel (zum Vergleich)", de(rt.dll_stationary, 2))
rp = EV.return_period_rows(settings.beta)
h1, h2 = st.columns([1, 1])
with h1:
    st.plotly_chart(V.fig_return_period(settings.beta), width="stretch", key="return_chart")
with h2:
    st.markdown("**Wiederkehrperiode des heutigen 100-Jahres-Pegels** (exakt)")
    st.dataframe(pd.DataFrame([{"Jahre nach heute": t_, "Wiederkehrperiode (Jahre)": de(v, 1)} for t_, v in rp]), hide_index=True, width="stretch")
with st.spinner(f"Ziehe {C.N_BLOCKS_LIVE} Blöcke zu {settings.nutz} Jahren (Jahr für Jahr) …"):
    blk, secs_b = _blocks(settings)
st.caption(f"Gegenprobe live: {blk['n']} Blöcke zu {settings.nutz} Jahren, Jahr für Jahr gezogen ({de(secs_b, 1)} s auf diesem Rechner, auf der Streamlit Cloud länger). "
           f"Anteil der Blöcke über dem heutigen 100-Jahres-Pegel (Jahre 0 bis {settings.nutz - 1}) {pct(blk['p_heute'], 1)} (exakt {pct(blk['p_heute_exakt'], 1)}, Standardfehler {de(100 * blk['se_heute'], 1)} Punkte); "
           f"über dem Design Life Level {pct(blk['p_dll'], 1)} (exakt {pct(blk['p_dll_exakt'], 1)}, Standardfehler {de(100 * blk['se_dll'], 1)} Punkte).")
fr = []
for b_ in C.BETA_OPTIONS:
    c = REF.blocks(b_)
    fr.append({"β": de(b_, 2), "Risiko exakt (Jahre 0 bis 49)": pct(c["risiko_exakt"], 2), "Simulation (Jahre 0 bis 49)": pct(c["ueber_heute"] / c["blocks"], 2), "Design Life Level (q = 5 %)": de(c["dll"], 3),
               "Blöcke über dem DLL": pct(c["ueber_dll"] / c["blocks"], 2)})
with st.expander(f"Eingefroren: {pint(REF.blocks(0.0)['blocks'])} Blöcke zu 50 Jahren je β gegen die exakten Werte"):
    st.dataframe(pd.DataFrame(fr), hide_index=True, width="stretch")
    st.caption(f"Quelle: {REF.meta('blocks')['quelle']}, Stand {REF.meta('blocks')['stand']}.")
kind, text = EV.verdict_risk(settings.beta, settings.nutz, settings.q)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 3: Stationär anpassen -------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Stationär anpassen heißt: Mittel statt Heute")
tr = a.truth
st.caption("Eine stationäre GEV (ξ frei) an dieselbe Reihe: sie liegt grob beim Pegel der **Mischung** aller Jahre, nicht beim Pegel des letzten Jahres. Wahr sind drei exakte Pegel (erstes Jahr, Mischung, letztes Jahr).")
if a.gev_stat is None:
    st.warning(f"Mit {settings.m} Jahren ist keine stationäre GEV-Anpassung möglich: {a.gev_error}")
else:
    lv = a.gev_stat.level()
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Pegel letztes Jahr (wahr, exakt)", de(tr.last, 2))
    s2.metric("Pegel der Mischung (wahr, exakt)", de(tr.mixture, 2), delta=f"{de(tr.mixture - tr.last, 2)} Modellfehler", delta_color="off", help="Mischung minus letztes Jahr: der Fehler, den die stationäre Annahme auch ohne Rauschen macht.")
    s3.metric("Pegel der stationären GEV", de(lv, 2), delta=f"{de(lv - tr.last, 2)} gegen heute", delta_color="off", help="Geschätzt aus dieser einen Reihe; Delta ist Schätzwert minus Pegel des letzten Jahres.")
    s4.metric("ξ̂ der Anpassung", de(a.gev_stat.xi, 3), help="Mit wachsendem β wird ξ̂ im Mittel negativ (eingefroren, m = 100: −0,17 bei β = 0,04).")
    st.plotly_chart(V.fig_levels(a.x, settings.beta, tr, a.gev_stat), width="stretch", key="levels_chart")
    if a.interval is not None:
        covered = a.interval[0] <= tr.last <= a.interval[1]
        st.caption(f"Delta-Intervall (95 %) dieser einen Anpassung für den Pegel: {de(a.interval[0], 2)} bis {de(a.interval[1], 2)}; es {'deckt' if covered else 'deckt nicht'} den wahren Pegel des letzten Jahres ({de(tr.last, 2)}). "
                   "Eine Reihe ist ein Punkt der Streuung, kein Befund.")
    else:
        st.caption("Für diese Anpassung gibt es kein Delta-Intervall (die numerische Hesse-Matrix ist nicht positiv definit).")
cov_rows = []
for (m_, b_) in ((30, 0.0), (50, 0.02), (100, 0.02), (100, 0.04)):
    c = REF.coverage_cell(m_, b_)
    cov_rows.append({"m (Jahre)": m_, "β": de(b_, 2), "Deckung, letztes Jahr": pct(c["letztes"], 1), "Deckung, mittleres Jahr": pct(c["mittleres"], 1), "Läufe": pint(c["R"]), "Standardfehler (Punkte)": de(c["se_punkte"], 1)})
with st.expander("Eingefroren: Abdeckung des 95-%-Delta-Intervalls der stationären GEV"):
    st.dataframe(pd.DataFrame(cov_rows), hide_index=True, width="stretch")
    st.caption(f"Quelle: {REF.meta('coverage')['quelle']}, Stand {REF.meta('coverage')['stand']}. Schon ohne Trend deckt das Delta-Intervall bei 30 Jahren nur in rund {pct(REF.coverage_cell(30, 0.0)['letztes'])} der Läufe.")
bc = REF.bias_cell()
st.caption(f"Verzerrung der stationären GEV am letzten Jahr bei m = {bc['m']}, β = {de(bc['beta'], 2)}: {de(bc['bias'], 2)} σ (R = {pint(bc['R'])} Läufe, Standardfehler {de(bc['se_bias'], 2)}), "
           f"Streuung {de(bc['sd'], 2)} σ, in {pct(bc['anteil_unter_wahr'])} der Läufe unter dem wahren Pegel; Zerlegung: Modellfehler der Mischung (exakt) plus Rest (Rauschen und Abstand der GEV-Anpassung von der Mischung).")
kind, text = EV.verdict_stationary(settings.m, settings.beta)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 4: Hilft das Trendmodell? -----------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Hilft das Trendmodell?")
st.caption("Drei Anpassungen an dieselben Reihen, Ziel ist der Pegel des letzten Jahres: (a) stationäre GEV (ξ frei), (b) Gumbel mit Lage-Trend (ξ = 0 bekannt, das richtige Modell: nur als Orakel-Zeile), "
           "(c) GEV mit Lage-Trend (ξ frei). Fair ist (c) gegen (a): gleiches freies ξ.")
cell = REF.rmse_cell(settings.m, settings.beta)
if cell:
    st.plotly_chart(V.fig_rmse(cell), width="stretch", key="rmse_chart")
    st.caption(f"Eingefroren: R = {cell['R']} Läufe. Anteil der Läufe, in denen (b) näher am wahren Pegel liegt als (a): {pct(cell['b']['besser_als_a'])}, (c) gegen (a): {pct(cell['c']['besser_als_a'])}. "
               f"ξ̂ im Mittel: (a) {de(cell['a']['xi_mittel'], 3)}, (c) {de(cell['c']['xi_mittel'], 3)}; ξ̂ am Rand −1 in {cell['a']['am_rand']} (a) und {cell['c']['am_rand']} (c) von {cell['R']} Läufen.")
else:
    st.info(f"Für {settings.m} Jahre gibt es keine eingefrorene Zelle (nur 20, 50 und 100 Jahre); die eigenen Läufe unten rechnen live.")
with st.spinner(f"Rechne {settings.runs} Läufe mit m = {settings.m} Jahren (drei Anpassungen je Lauf) …"):
    live, secs_r = _runs(settings)
if live.get("n"):
    st.caption(f"Eigene Läufe: {live['n']} (Zufalls-Seed der Seitenleiste), {de(secs_r, 1)} s auf diesem Rechner, auf der Streamlit Cloud länger. RMSE (a) {de(live['a']['rmse'], 2)}, (b) {de(live['b']['rmse'], 2)}, (c) {de(live['c']['rmse'], 2)} σ; "
               f"Median des Betrags {de(live['a']['med_abs'], 2)} / {de(live['b']['med_abs'], 2)} / {de(live['c']['med_abs'], 2)}. Bei so wenigen Läufen streut das stark.")
tab = []
for c in REF.grid()["rmse"]["daten"]:
    d = c["diff_c_a"]
    tab.append({"m": c["m"], "β": de(c["beta"], 2), "RMSE (a)": de(c["a"]["rmse"], 2), "RMSE (b)": de(c["b"]["rmse"], 2), "RMSE (c)": de(c["c"]["rmse"], 2),
                "Median |Fehler| (a)": de(c["a"]["med_abs"], 2), "Median |Fehler| (c)": de(c["c"]["med_abs"], 2), "(c) − (a)": de(d["diff"], 2), "Standardfehler": de(d["se"], 2), "Urteil": EV.VERDICT_WORDS[d["urteil"]]})
with st.expander("Eingefroren: alle zwölf Zellen (400 Läufe je Zelle)"):
    st.dataframe(pd.DataFrame(tab), hide_index=True, width="stretch")
    st.caption(f"Quelle: {REF.meta('rmse')['quelle']}, Stand {REF.meta('rmse')['stand']}. Die RMSE-Zahlen bei 20 Jahren sind von wenigen Läufen mit ξ̂ am Rand bestimmt (Ausreißer); belastbar ist der Median des Betrags.")
kind, text = EV.verdict_trendmodel(settings.m, settings.beta, live)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 5: Last statt Zeit (Vehikel A) ------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 5: Last statt Zeit (Vehikel A, eingefroren)")
st.caption("Vehikel A (ausgedacht): M/M/1-Wartezeit am Tor, Dienstrate 1, Tag = 256 Kunden (Annahme), Tage unabhängig, Start im Gleichgewicht. Exakt: P(W > x) = ρ·e^{−(1−ρ)x}. Wächst die Last, wirken Skala **und** Lage der Tagesmaxima; "
           "die Gumbel-Näherung der Rechnung (Skala 1/(1−ρ), Lage ln(nθρ)/(1−ρ), θ = (1−ρ)² aus „Spitzen-Cluster“) trifft den 100-Tage-Pegel bis ρ = 0,80 mit höchstens 2,4 % Abweichung (bei ρ = 0,90 liegt sie 8,6 % zu niedrig), Skala und Lage einzeln trifft sie nicht.")
c1rows = REF.vehicle_c1()
st.plotly_chart(V.fig_vehicle_scale_location(c1rows), width="stretch", key="veh_scale")
st.dataframe(pd.DataFrame([{"ρ": de(r["rho"], 2), "Skala (Simulation)": de(r["sigma_dach"], 2), "Skala (Rechnung)": de(r["sigma_rechnung"], 2), "Lage (Simulation)": de(r["mu_dach"], 2), "Lage (Rechnung)": de(r["mu_rechnung"], 2),
                           "Pegel 0,99 (Simulation)": de(r["pegel_simulation"], 2), "Pegel 0,99 (Rechnung)": de(r["pegel_rechnung"], 2), "KS-Abstand der Rechnung": de(r["ks_rechnung"], 3)} for r in sorted(c1rows, key=lambda x: x["rho"])]),
             hide_index=True, width="stretch")
st.caption(f"{pint(c1rows[0]['tage'])} Tage je Last; Pegel in Abfertigungsdauern; Quelle: {REF.meta('vehicle_c1')['quelle']}, Stand {REF.meta('vehicle_c1')['stand']}.")
c2d = REF.vehicle_c2(settings.last_end)
st.plotly_chart(V.fig_vehicle_ratio(c2d), width="stretch", key="veh_ratio")
st.dataframe(pd.DataFrame([{"Modell": n_, "Median Schätzwert / wahr": de(c2d[k]["median"], 3), "Q10 bis Q90": f"{de(c2d[k]['q10'], 2)} bis {de(c2d[k]['q90'], 2)}", "Anteil Unterschätzung": pct(c2d[k]["anteil_unter_1"], 1), "RMSE von ln(Verhältnis)": de(c2d[k]["rmse_ln"], 3)}
                           for n_, k in (("stationäre GEV", "stationaer"), ("GEV mit Lage-Trend", "lage_trend"), ("Gumbel, Trend in Lage und ln σ", "lage_skala_trend"), ("Kontrolle: kein Anstieg, stationäre GEV", "kontrolle_ohne_trend"))]),
             hide_index=True, width="stretch")
st.caption(f"{c2d['R']} Läufe zu je {c2d['tage_je_lauf']} Tagen, Last linear von {de(c2d['rho_von'], 2)} auf {de(c2d['rho_bis'], 2)}; wahr: 0,99-Quantil des Tagesmaximums bei ρ = {de(c2d['rho_bis'], 2)} aus {pint(c2d['wahr_tage'])} Tagen "
           f"({de(c2d['wahr'], 2)}, Standardfehler etwa {de(c2d['wahr_se'], 2)}, bei so wenigen Teilstichproben eher zu klein geschätzt). Die Last wächst hier linear und die Tage sind unabhängig: Annahmen dieser Messung.")
kind, text = EV.verdict_vehicle(settings.last_end, settings.modell)
getattr(st, kind)(text)
wk = REF.weekly()
with st.expander("Vehikel B (Wochenstruktur): nur als Tabelle, periodisch und ohne Trend"):
    st.dataframe(pd.DataFrame([{"Fall": "ganze Woche (Wochenendanteil 2/7), aus der kopierten Gewichtstabelle", "Pegel je Container, p = 10⁻⁵ (Tage)": de(_week_all(), 3)},
                               {"Fall": "nur Werktage", "Pegel je Container, p = 10⁻⁵ (Tage)": de(wk["werktage"], 3)},
                               {"Fall": "nur Wochenende", "Pegel je Container, p = 10⁻⁵ (Tage)": de(wk["wochenende"], 3)}]), hide_index=True, width="stretch")
    st.caption(f"Das Wochenende liegt um den Faktor {de(wk['faktor'], 2)} über den Werktagen (Quadratur und Bisektion aus der Definition der Linie, ohne Zufall; der Faktor ist exp(0,30), der Wochenend-Parameter der Mischung, also eine Eingabe und kein Ergebnis). Das ist eine periodische Nichtstationarität; "
               "ob und wie man sie erkennt oder anpasst, ist hier nicht gemessen. Quelle der Gewichtstabelle: gemeinsame Tabelle der Linie (Demo „Überschreitungen und die GPD“), unverändert kopiert.")

st.markdown("---")

# --- Grenzen ----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(T.ASSUMPTIONS)
st.markdown(T.CONVENTIONS)
st.info(T.VEHIKEL_TEXT)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    # erklaerseite:lesehilfe:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/trend-spitzen-erklaert.json, nicht von Hand ändern)
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind Jahresmaxima $X_t = \beta\,t + G_t$ (Gumbel mit linearem Lage-Trend, $\sigma = 1$, unabhängig). Gesucht sind der Pegel zur Überschreitungswahrscheinlichkeit $p$ im Jahr $t$, das Risiko in $m$ Jahren und die Anpassungen an die Reihe.

| Symbol | Bedeutung |
|---|---|
| $t$, $m$ | Jahr ($t = 0, \dots, m-1$; „heute“ ist das letzte Jahr $m-1$) und Länge der Reihe in Jahren |
| $X_t$, $G_t$ | Jahresmaximum und Standard-Gumbel-Rauschen (Skala $\sigma = 1$) |
| $\beta$, $D$ | Trend: Anstieg der Lage je Jahr in Einheiten $\sigma$; Gesamtanstieg $D = \beta\,(m-1)$ |
| $p$, $x_t(p)$ | Überschreitungswahrscheinlichkeit und Pegel des Jahres $t$ dazu; „100-Jahres-Pegel“ heißt $p = 1/100$ |
| $y_{100}$ | Grundhöhe des 100-Jahres-Pegels ohne Trend, 4,600149 |
| $F_{\mathrm{mix}}$, $x_{\mathrm{mix}}$ | Mischung aller Jahre der Reihe und ihr 0,99-Quantil |
| $S$ | Summe der Gewichte $e^{\beta t}$ über die Nutzungsdauer |
| $q$, $x_q$ | Niveau (5 %) und Design Life Level |
| $T_t$ | Wiederkehrperiode des heutigen Pegels im Jahr $t$ nach heute (keine Frist) |
| $\lambda$ | Nichtzentralitätsparameter der Trennschärfe des Trendtests |
| $\xi$ | Formparameter der GEV ($\xi = 0$ Gumbel) |
| $\rho$, $n$, $\theta$ | Vehikel A: Last, Kunden je Tag, Extremalindex |

[Schritt für Schritt erklärt, mit Herleitung und Literatur](https://sebastianhanisch.net/trend-spitzen-erklaert.html#fachleute)
        """
    )
    # erklaerseite:lesehilfe:end
    st.markdown(T.FORMULAS)

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Extremwerte: Blöcke, Schwelle, Pegel](https://sebastianhanisch.net/konzepte-extremwert.html)."
)
