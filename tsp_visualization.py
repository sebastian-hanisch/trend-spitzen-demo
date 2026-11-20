"""Plotly-Abbildungen (alle Achsen fest, `fixedrange`)."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import tsp_constants as C
import tsp_models as M
import tsp_numeric as N
from tsp_texts import de


def _layout(fig, title=None, height=380, xtitle=None, ytitle=None):
    fig.update_layout(
        title=title, height=height, margin=dict(l=55, r=15, t=45 if title else 15, b=55), legend=dict(orientation="h", y=-0.25),
        xaxis=dict(fixedrange=True, title=xtitle), yaxis=dict(fixedrange=True, title=ytitle),
    )
    return fig


def fig_series(x, beta, f0: M.GumbelFit, f1: M.GumbelFit):
    """Reihe, wahre Lage β·t, angepasste Lage mit und ohne Trend."""
    m = len(x)
    t = np.arange(m)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=x, mode="markers", name=f"{m} Jahresmaxima", marker=dict(color=C.COLOR_PTS, size=6)))
    fig.add_trace(go.Scatter(x=t, y=beta * t, mode="lines", name="wahre Lage β·t", line=dict(color=C.COLOR_TRUE, width=2.5)))
    fig.add_trace(go.Scatter(x=t, y=f1.loc(t), mode="lines", name=f"angepasst mit Trend (β̂ = {de(f1.b, 3)})", line=dict(color=C.COLOR_FIT, width=2.5)))
    fig.add_trace(go.Scatter(x=t, y=np.full(m, f0.a), mode="lines", name="angepasst ohne Trend", line=dict(color=C.COLOR_ALT, width=2.5, dash="dash")))
    return _layout(fig, "Jahresmaxima, wahre und angepasste Lage", 400, "Jahr t (heute = letztes Jahr)", "Jahresmaximum (σ)")


def fig_power(beta, cells, m_now):
    """Trennschärfe über m: Formel (Linie) und eingefrorene Simulation (Punkte mit Wilson-Band); bei β = 0 die Fehlerquote erster Art."""
    fig = go.Figure()
    ms = np.arange(5, 151)
    if beta > 0:
        fig.add_trace(go.Scatter(x=ms, y=[M.power(beta, int(m)) for m in ms], mode="lines", name="Formel", line=dict(color=C.COLOR_TRUE, width=2.5)))
        fig.add_hline(y=0.8, line=dict(color="rgba(120,120,120,0.8)", dash="dot"), annotation_text="80 %", annotation_position="top left")
    else:
        fig.add_trace(go.Scatter(x=ms, y=np.full(ms.size, 0.05), mode="lines", name="Soll (Niveau 5 %)", line=dict(color=C.COLOR_TRUE, width=2.5)))
    pts = sorted(cells, key=lambda c: c["m"])
    if pts:
        y = [c["hits"] / (c["R"] - c["ausgefallen"]) for c in pts]
        lo = [y[i] - N.wilson(c["hits"], c["R"] - c["ausgefallen"])[0] for i, c in enumerate(pts)]
        hi = [N.wilson(c["hits"], c["R"] - c["ausgefallen"])[1] - y[i] for i, c in enumerate(pts)]
        fig.add_trace(go.Scatter(x=[c["m"] for c in pts], y=y, mode="markers", name="Simulation (eingefroren, Wilson-Band)", marker=dict(color=C.COLOR_FIT, size=10),
                                 error_y=dict(type="data", symmetric=False, array=hi, arrayminus=lo)))
    fig.add_vline(x=m_now, line=dict(color="rgba(120,120,120,0.8)", dash="dot"))
    fig.update_yaxes(range=[0, 1.02] if beta > 0 else [0, 0.2])
    title = f"Trennschärfe des Trendtests, β = {de(beta, 2)} σ je Jahr" if beta > 0 else "Fehlerquote erster Art (β = 0)"
    return _layout(fig, title, 360, "Jahre m", "Anteil der Läufe mit Meldung „Trend“")


def fig_return_period(beta):
    """Wiederkehrperiode des heutigen 100-Jahres-Pegels in den Jahren nach heute (exakt)."""
    t = np.arange(0, 101)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=[M.return_period_after(beta, float(v)) for v in t], mode="lines", name="exakt", line=dict(color=C.COLOR_TRUE, width=2.5)))
    fig.add_trace(go.Scatter(x=list(C.LEVEL_YEARS), y=[M.return_period_after(beta, float(v)) for v in C.LEVEL_YEARS], mode="markers", name="Tabelle", marker=dict(color=C.COLOR_FIT, size=9)))
    fig.update_yaxes(range=[0, 105])
    return _layout(fig, "Wiederkehrperiode des heutigen 100-Jahres-Pegels", 340, "Jahre nach heute", "Wiederkehrperiode (Jahre, Mittel)")


def fig_levels(x, beta, truth, gev_fit):
    """Reihe mit den drei wahren Pegeln (erstes Jahr, Mischung, letztes Jahr, exakt), dem wahren Pegel jedes Jahres und dem Pegel der stationären Anpassung."""
    m = len(x)
    t = np.arange(m)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=x, mode="markers", name=f"{m} Jahresmaxima", marker=dict(color=C.COLOR_PTS, size=5)))
    fig.add_trace(go.Scatter(x=t, y=beta * t + M.Y100, mode="lines", name="wahrer 100-Jahres-Pegel des Jahres t", line=dict(color=C.COLOR_TRUE, width=2)))
    for name, val, color, dash in (("Pegel erstes Jahr", truth.first, "rgba(120,120,120,0.9)", "dot"), ("Mischung (stationäres Ziel)", truth.mixture, C.COLOR_ALT, "dash"),
                                   ("Pegel letztes Jahr (heute)", truth.last, C.COLOR_TRUE, "dot")):
        fig.add_trace(go.Scatter(x=[0, m - 1], y=[val, val], mode="lines", name=f"{name} {de(val, 2)}", line=dict(color=color, width=2, dash=dash)))
    if gev_fit is not None:
        lv = gev_fit.level()
        fig.add_trace(go.Scatter(x=[0, m - 1], y=[lv, lv], mode="lines", name=f"stationäre GEV, Pegel {de(lv, 2)}", line=dict(color=C.COLOR_FIT, width=3)))
    return _layout(fig, "Pegel von heute gegen den Pegel der Reihe", 420, "Jahr t", "Wert (σ)")


def fig_rmse(cell):
    """Balken: RMSE und Median des Betrags der drei Anpassungen in einer eingefrorenen Zelle."""
    names = ["stationäre GEV (a)", "Gumbel mit Trend, ξ = 0 bekannt (b)", "GEV mit Trend, ξ frei (c)"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=[cell[k]["rmse"] for k in ("a", "b", "c")], name="RMSE", marker_color=C.COLOR_FIT))
    fig.add_trace(go.Bar(x=names, y=[cell[k]["med_abs"] for k in ("a", "b", "c")], name="Median des Betrags", marker_color=C.COLOR_ALT))
    fig.update_layout(barmode="group")
    return _layout(fig, f"Fehler des Pegels von heute, m = {cell['m']} Jahre, β = {de(cell['beta'], 2)} (R = {cell['R']} Läufe)", 360, None, "Fehler (σ)")


def fig_vehicle_scale_location(rows):
    """Gumbel-ML-Skala und -Lage der Tagesmaxima über der Last, daneben die Näherung der Rechnung."""
    rows = sorted(rows, key=lambda r: r["rho"])
    rho = [r["rho"] for r in rows]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Skala σ", "Lage μ"))
    for col, (k_sim, k_calc) in enumerate((("sigma_dach", "sigma_rechnung"), ("mu_dach", "mu_rechnung")), start=1):
        fig.add_trace(go.Scatter(x=rho, y=[r[k_sim] for r in rows], mode="lines+markers", name="Simulation (Gumbel-ML)", line=dict(color=C.COLOR_FIT), showlegend=(col == 1)), row=1, col=col)
        fig.add_trace(go.Scatter(x=rho, y=[r[k_calc] for r in rows], mode="lines+markers", name="Näherung der Rechnung", line=dict(color=C.COLOR_ALT, dash="dash"), showlegend=(col == 1)), row=1, col=col)
        fig.update_xaxes(fixedrange=True, title_text="Last ρ", row=1, col=col)
        fig.update_yaxes(fixedrange=True, row=1, col=col)
    fig.update_layout(height=340, margin=dict(l=55, r=15, t=45, b=55), legend=dict(orientation="h", y=-0.3))
    return fig


def fig_vehicle_ratio(c2):
    """Pegel-Dach / wahr: Median mit Q10 bis Q90 für die drei Modelle und die Kontrolle ohne Trend."""
    items = [("stationär", "stationaer"), ("Lage-Trend", "lage_trend"), ("Lage- und Skalen-Trend", "lage_skala_trend"), ("Kontrolle: kein Anstieg", "kontrolle_ohne_trend")]
    fig = go.Figure()
    med = [c2[k]["median"] for _n, k in items]
    fig.add_trace(go.Scatter(x=[n for n, _k in items], y=med, mode="markers", name="Median", marker=dict(color=C.COLOR_FIT, size=11),
                             error_y=dict(type="data", symmetric=False, array=[c2[k]["q90"] - c2[k]["median"] for _n, k in items], arrayminus=[c2[k]["median"] - c2[k]["q10"] for _n, k in items])))
    fig.add_hline(y=1.0, line=dict(color=C.COLOR_TRUE, dash="dot"), annotation_text="wahr", annotation_position="top left")
    fig.update_yaxes(range=[0.4, 1.5])
    return _layout(fig, f"Pegel des letzten Tages, geschätzt / wahr (Median, Q10 bis Q90; R = {c2['R']} Läufe)", 360, None, "Verhältnis")
