"""Static HTML dashboard (Plotly, no server) built from the KPI tables and result files.

Deterministic output: fixed div ids, no timestamps, so the committed file can be regenerated
and compared byte for byte. plotly.js is loaded from the official CDN at the version bundled
with the pinned plotly package.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
POLICY_COLOR = {"fixed": SERIES[0], "threshold": SERIES[1], "forecast": SERIES[2]}
POLICY_LABEL = {"fixed": "Fixed, 3x a week", "threshold": "Reactive, reading >= 70%", "forecast": "Forecast-based"}
METHOD_COLOR = {"persistence": SERIES[0], "seasonal_naive": SERIES[1], "mean_rate": SERIES[2], "lightgbm": SERIES[3]}
METHOD_LABEL = {"persistence": "Last value", "seasonal_naive": "Seasonal naive", "mean_rate": "14-day mean rate",
                "lightgbm": "LightGBM"}
TYPE_COLOR = {"residual": SERIES[0], "paper": SERIES[1], "plastic": SERIES[2], "glass": SERIES[3], "unknown": "#898781"}
SCENARIO_LABEL = {"base": "Base", "trnava_variability": "Trnava-like variability", "low_fill_rate": "Low fill rate",
                  "high_fill_rate": "High fill rate", "bigger_truck": "Bigger truck (80 bins)",
                  "facility_in_town": "Unloading point in town", "more_search": "More route search (200 it.)"}

CSS = """
:root { --surface:#fcfcfb; --page:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
        --grid:#e1e0d9; --border:rgba(11,11,11,0.10); }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) { --surface:#1a1a19; --page:#0d0d0d; --ink:#f0efec; --ink2:#c3c2b7;
        --muted:#898781; --grid:#2c2c2a; --border:rgba(255,255,255,0.10); } }
:root[data-theme="dark"] { --surface:#1a1a19; --page:#0d0d0d; --ink:#f0efec; --ink2:#c3c2b7;
        --muted:#898781; --grid:#2c2c2a; --border:rgba(255,255,255,0.10); }
* { box-sizing: border-box; }
body { margin:0; background:var(--page); color:var(--ink);
       font-family: system-ui, -apple-system, "Segoe UI", sans-serif; line-height:1.45; }
main { max-width: 1120px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 1.5rem; margin: 0 0 4px; } h2 { font-size: 1.15rem; margin: 32px 0 8px; }
p.lead, p.note { color: var(--ink2); margin: 4px 0 12px; } p.note { font-size: 0.85rem; }
.tiles { display:grid; grid-template-columns: repeat(auto-fit, minmax(200px,1fr)); gap:12px; }
.tile { background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:12px 14px; }
.tile .label { color:var(--ink2); font-size:0.85rem; } .tile .value { font-size:1.6rem; font-weight:600; }
.tile .sub { color:var(--muted); font-size:0.8rem; }
.card { background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:8px; margin:8px 0; }
.grid2 { display:grid; grid-template-columns: repeat(auto-fit, minmax(320px,1fr)); gap:12px; }
table { border-collapse: collapse; width:100%; font-size:0.85rem; }
th, td { text-align:right; padding:4px 8px; border-bottom:1px solid var(--grid); white-space:nowrap; }
th:first-child, td:first-child { text-align:left; } th { color:var(--ink2); font-weight:600; }
.tablewrap { overflow-x:auto; }
"""

THEME_JS = """
<script>
(function () {
  function dark() {
    var t = document.documentElement.getAttribute('data-theme');
    if (t === 'dark') return true; if (t === 'light') return false;
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  }
  function apply() {
    var d = dark(), ink = d ? '#c3c2b7' : '#52514e', grid = d ? '#2c2c2a' : '#e1e0d9';
    document.querySelectorAll('.js-plotly-plot').forEach(function (el) {
      Plotly.relayout(el, {'font.color': ink, 'xaxis.gridcolor': grid, 'yaxis.gridcolor': grid,
                           'xaxis.linecolor': grid, 'yaxis.linecolor': grid});
    });
  }
  window.addEventListener('load', apply);
  if (window.matchMedia) window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', apply);
})();
</script>
"""


def _layout(fig, title, ytitle=None, height=340):
    fig.update_layout(title=dict(text=title, x=0.01, xanchor="left", font=dict(size=14)), height=height,
                      margin=dict(l=56, r=16, t=48, b=48), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family="system-ui, Segoe UI, sans-serif", size=12,
                                                              color="#52514e"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0.01, font=dict(size=11)),
                      hoverlabel=dict(font_size=12), bargap=0.25, bargroupgap=0.06)
    fig.update_xaxes(gridcolor="#e1e0d9", linecolor="#c3c2b7", zeroline=False)
    fig.update_yaxes(gridcolor="#e1e0d9", linecolor="#c3c2b7", zeroline=False, title_text=ytitle)
    return fig


def _html(fig, div_id):
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id,
                       config={"displaylogo": False, "responsive": True})


def _table(df: pd.DataFrame, fmt: dict) -> str:
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            s = fmt[c].format(v) if c in fmt and pd.notna(v) else ("" if pd.isna(v) else str(v))
            cells.append(f"<td>{html.escape(s)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="tablewrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def build(root: Path, kpi: dict) -> str:
    import plotly.graph_objects as go
    res = root / "results"
    pol = kpi["kpi_policy"]
    base = pol[pol["scenario"] == "base"].set_index("policy")
    parts = []
    tiles = []
    f = base.loc["forecast"]
    for label, val, sub in [
        ("Truck km, forecast vs fixed", f"{f['km_change_pct']:+.1f}%", f"{f['km']:.0f} vs {base.loc['fixed', 'km']:.0f} km over 6 weeks"),
        ("Bin lifts, forecast vs fixed", f"{f['lifts_change_pct']:+.1f}%", f"{f['lifts']:.0f} vs {base.loc['fixed', 'lifts']:.0f} lifts"),
        ("Overflow bin-days, forecast vs fixed", f"{f['overflow_change_pct']:+.1f}%",
         f"{f['overflow_bin_days']:.0f} vs {base.loc['fixed', 'overflow_bin_days']:.0f}"),
        ("CO2 (DESNZ per-km factor)", f"{f['co2_kg'] / 1000:.2f} t", f"fixed schedule: {base.loc['fixed', 'co2_kg'] / 1000:.2f} t"),
    ]:
        tiles.append(f'<div class="tile"><div class="label">{html.escape(label)}</div><div class="value">{html.escape(val)}</div>'
                     f'<div class="sub">{html.escape(sub)}</div></div>')
    parts.append('<div class="tiles">' + "".join(tiles) + "</div>")
    parts.append('<p class="note">Bishan, base scenario, mean of three simulated six-week runs. Fill rates, truck size and '
                 'shift times are assumptions; bin locations, dwelling units and road distances are public data. '
                 'See the README provenance table.</p>')
    # scenarios: km and overflow per policy
    figs = []
    for metric, title, ytitle in [("km", "Truck km per scenario (6 weeks)", "km"),
                                  ("overflow_bin_days", "Overflow bin-days per scenario", "bin-days"),
                                  ("lifts", "Bin lifts per scenario", "lifts")]:
        fig = go.Figure()
        for p in ("fixed", "threshold", "forecast"):
            sub = pol[pol["policy"] == p]
            fig.add_bar(x=[SCENARIO_LABEL.get(s, s) for s in sub["scenario"]], y=sub[metric], name=POLICY_LABEL[p],
                        marker_color=POLICY_COLOR[p], hovertemplate="%{x}<br>" + POLICY_LABEL[p] + ": %{y:.0f}<extra></extra>")
        figs.append(_html(_layout(fig, title, ytitle), f"scen-{metric}"))
    parts.append("<h2>Collect when needed vs fixed schedule</h2>")
    parts.append('<div class="card">' + figs[0] + "</div>")
    parts.append('<div class="grid2"><div class="card">' + figs[1] + '</div><div class="card">' + figs[2] + "</div></div>")
    wk = kpi["kpi_weekly"]
    wk = wk[wk["scenario"] == "base"]
    fig = go.Figure()
    for p in ("fixed", "threshold", "forecast"):
        s = wk[wk["policy"] == p]
        fig.add_scatter(x=s["week"], y=s["overflow_bin_days"], mode="lines+markers", name=POLICY_LABEL[p],
                        line=dict(color=POLICY_COLOR[p], width=2), marker=dict(size=8),
                        hovertemplate="week %{x}: %{y:.1f} overflow bin-days<extra>" + POLICY_LABEL[p] + "</extra>")
    parts.append('<div class="card">' + _html(_layout(fig, "Overflow bin-days per week, base scenario", "bin-days"), "weekly-overflow") + "</div>")
    # forecasting
    fk = kpi["kpi_forecast"]
    fig = go.Figure()
    for m in ("persistence", "seasonal_naive", "mean_rate", "lightgbm"):
        s = fk[fk["method"] == m].sort_values("horizon")
        fig.add_bar(x=[f"{h} day{'s' if h > 1 else ''} ahead" for h in s["horizon"]], y=s["mae"], name=METHOD_LABEL[m],
                    marker_color=METHOD_COLOR[m], hovertemplate="%{x}<br>" + METHOD_LABEL[m] + ": MAE %{y:.2f} pp<extra></extra>")
    parts.append("<h2>Fill-level forecasting (Trnava open data)</h2>")
    fc = json.loads((res / "forecast_backtest.json").read_text(encoding="utf-8"))
    parts.append(f'<p class="lead">{fc["n_forecasts"]:,} forecasts for {fc["n_bins"]} bins, rolling origins '
                 f'{html.escape(fc["origins"][0])} to {html.escape(fc["origins"][-1])}; error in percentage points of fill, '
                 'only windows without a collection.</p>')
    parts.append('<div class="card">' + _html(_layout(fig, "Mean absolute error by horizon", "MAE (pp)"), "forecast-mae") + "</div>")
    ob = kpi["kpi_bins_observed"]
    parts.append('<div class="card">' + _table(
        ob.rename(columns={"waste_type": "waste type", "bin_days": "bin-days", "mean_level_pct": "mean level %",
                           "mean_daily_increment_pp": "mean daily rise (pp)", "collections_per_bin_week": "collections / bin / week",
                           "share_days_max_ge_90": "days >= 90% full", "stale_share": "stale sensor days"}),
        {"mean level %": "{:.1f}", "mean daily rise (pp)": "{:.1f}", "collections / bin / week": "{:.2f}",
         "days >= 90% full": "{:.1%}", "stale sensor days": "{:.1%}", "bin-days": "{:,.0f}"}) + "</div>")
    bins = pd.read_csv(root / "data" / "inputs" / "fill_bins.csv")
    fig = go.Figure()
    for t in ("residual", "paper", "plastic", "glass"):
        s = bins[bins["waste_type"] == t].drop_duplicates(["lat", "lon"])
        fig.add_scatter(x=s["lon"], y=s["lat"], mode="markers", name=t, marker=dict(color=TYPE_COLOR[t], size=8,
                        line=dict(color="#fcfcfb", width=1)), hovertemplate=t + " bin station<extra></extra>")
    fig.update_yaxes(scaleanchor="x", scaleratio=1.0 / 0.665)
    parts.append('<div class="card">' + _html(_layout(fig, "Trnava smart-bin stations (one marker per station and type)",
                                                      "latitude", 420), "trnava-map") + "</div>")
    # routing
    rt = kpi["kpi_routing"]
    parts.append("<h2>Routing benchmark (Kim et al. 2006 instances)</h2>")
    if len(rt):
        parts.append('<div class="card">' + _table(
            rt[["instance", "customers", "best", "mean", "reference", "gap_best_pct", "gap_mean_pct", "reference_source"]].rename(
                columns={"gap_best_pct": "gap best %", "gap_mean_pct": "gap mean %", "reference_source": "reference source"}),
            {"best": "{:.2f}", "mean": "{:.2f}", "reference": "{:.2f}", "gap best %": "{:+.2f}", "gap mean %": "{:+.2f}"}) + "</div>")
        parts.append('<p class="note">Distances in miles. Reference = lowest published value read in an accessible source; '
                     'see results/routing_benchmark.json.</p>')
    else:
        parts.append('<p class="note">Benchmark results not available (instances are fetched, not committed).</p>')
    # edge
    em = res / "edge_model.json"
    if em.exists():
        e = json.loads(em.read_text(encoding="utf-8"))
        rows = []
        for name in ("fp32", "int8"):
            m = e["models"][name]
            rows.append({"model": name.upper(), "accuracy": m["accuracy"], "macro F1": m["macro_f1"], "size (MB)": m["bytes"] / 1e6,
                         "p50 ms (1 thread)": m["latency_1_thread"]["p50_ms"], "p95 ms (1 thread)": m["latency_1_thread"]["p95_ms"],
                         "p50 ms (4 threads)": m["latency_4_threads"]["p50_ms"]})
        parts.append("<h2>Waste-image classifier, FP32 vs INT8</h2>")
        parts.append('<div class="card">' + _table(pd.DataFrame(rows), {"accuracy": "{:.1%}", "macro F1": "{:.3f}", "size (MB)": "{:.2f}",
                     "p50 ms (1 thread)": "{:.2f}", "p95 ms (1 thread)": "{:.2f}", "p50 ms (4 threads)": "{:.2f}"}) + "</div>")
        parts.append(f'<p class="note">TrashNet test split ({e["test_images"]} images). Latency measured with ONNX Runtime '
                     f'{html.escape(e["onnxruntime"])} on {html.escape(e["cpu"])}; not measured on edge hardware.</p>')
    body = "\n".join(parts)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Waste Collection Dashboard</title>
<style>{CSS}</style>
<script src="https://cdn.plot.ly/plotly-{_plotlyjs_version()}.min.js" charset="utf-8"></script>
</head>
<body>
<main>
<h1>Waste collection desk study</h1>
<p class="lead">Personal desk study: routing with unloading trips, fill-level forecasting, a Singapore HDB-town
simulation and an INT8 edge classifier. Generated from the files in <code>results/</code>; no server needed.</p>
{body}
</main>
{THEME_JS}
</body>
</html>
"""


def _plotlyjs_version() -> str:
    from plotly.offline import get_plotlyjs_version
    return get_plotlyjs_version()
