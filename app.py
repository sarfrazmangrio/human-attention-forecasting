import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime

st.set_page_config(
    page_title="Human Attention Forecasting",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500&display=swap" rel="stylesheet">
<style>
* { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; }
.stApp { background: #0a0a12; color: #e0e0f0; }
section[data-testid="stSidebar"] {
    background: #0f0f1a;
    border-right: 1px solid #1a1a2e;
}
.stTabs [data-baseweb="tab-list"] { background: transparent; border-bottom: 1px solid #1a1a2e; }
.stTabs [data-baseweb="tab"] { color: #6060a0; font-size: 13px; font-weight: 500; }
.stTabs [aria-selected="true"] { color: #a0a0ff !important; }
div[data-testid="stMetric"] {
    background: #0f0f1a;
    border: 1px solid #1a1a2e;
    border-radius: 12px;
    padding: 16px 20px;
}
.big-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 44px;
    font-weight: 700;
    letter-spacing: -1.5px;
    line-height: 1.1;
    color: #e8e8ff;
    margin-bottom: 8px;
}
.signal-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 24px;
    font-weight: 600;
    color: #c8c8f0;
    margin: 8px 0 24px 0;
}
.tag {
    display: inline-block;
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.08em;
    color: #8080c0;
    margin-right: 8px;
    text-transform: uppercase;
}
.mcard {
    background: #0f0f1a;
    border: 1px solid #1a1a2e;
    border-radius: 14px;
    padding: 22px 24px;
    min-height: 110px;
}
.mcard-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #505080;
    margin-bottom: 10px;
}
.mcard-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 28px;
    font-weight: 700;
    line-height: 1;
}
.mcard-sub { font-size: 11px; color: #505080; margin-top: 8px; }
.brief-box {
    background: #0f0f1a;
    border: 1px solid #1a1a2e;
    border-left: 3px solid #6366f1;
    border-radius: 14px;
    padding: 28px 32px;
    line-height: 1.9;
    color: #b0b0d0;
    font-size: 15px;
}
.brief-box strong { color: #e0e0ff; }
.section-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #404070;
    margin: 28px 0 14px 0;
}
</style>
""", unsafe_allow_html=True)

DATA_DIR = Path(".")

PLATFORMS = ["YouTube", "TikTok", "Instagram", "LinkedIn", "Reddit"]
CATEGORIES = [
    "Artificial Intelligence and Technology",
    "Personal Finance",
    "Mental Health and Self Improvement",
    "Sports",
    "Gaming",
    "News and Politics",
    "Entertainment and Media",
    "Science and Education",
    "Lifestyle and Advice",
    "Culture Food and Travel"
]

PLATFORM_COLORS = {
    "YouTube": "#FF4444", "TikTok": "#40d9d0",
    "Instagram": "#E1306C", "LinkedIn": "#0A8FC4", "Reddit": "#FF6314"
}
CATEGORY_COLORS = [
    "#6366f1","#8b5cf6","#a78bfa","#c084fc",
    "#06b6d4","#0ea5e9","#22c55e","#f59e0b","#ef4444","#ec4899"
]


@st.cache_data
def load_data():
    dfs = {}

    for key, fname in {
        "summary"  : "future_summary.csv",
        "backtest" : "backtest_model_results.csv",
        "best"     : "best_models.csv",
        "hist"     : "platform_trends.csv",
    }.items():
        try:
            df = pd.read_csv(DATA_DIR / fname)
            dfs[key] = df
        except Exception:
            dfs[key] = pd.DataFrame()

    for key, fname in {
        "future_plat" : "platform_future_forecasts.csv",
        "future_cat"  : "category_future_forecasts.csv",
        "future_all"  : "future_forecasts.csv",
    }.items():
        try:
            df = pd.read_csv(DATA_DIR / fname)
            for col in df.columns:
                if any(x in col.lower() for x in ["date","ds","week","time"]):
                    df[col] = pd.to_datetime(df[col], errors="coerce")
            dfs[key] = df
        except Exception:
            dfs[key] = pd.DataFrame()

    try:
        hist = pd.read_csv(DATA_DIR / "platform_trends.csv",
                           index_col=0, parse_dates=True)
        dfs["hist_indexed"] = hist
    except Exception:
        dfs["hist_indexed"] = pd.DataFrame()

    return dfs


data = load_data()


def find_col(df, *hints):
    for hint in hints:
        for col in df.columns:
            if hint.lower() in col.lower():
                return col
    return None


def get_summary(target):
    df = data["summary"]
    if df.empty:
        return {}
    rows = df[df["Target"].astype(str) == str(target)]
    return rows.iloc[0].to_dict() if not rows.empty else {}


def get_future(target):
    for key in ["future_all", "future_plat", "future_cat"]:
        df = data.get(key, pd.DataFrame())
        if df.empty:
            continue
        tc = find_col(df, "target", "signal", "name")
        if tc and str(target) in df[tc].astype(str).values:
            return df[df[tc].astype(str) == str(target)].copy()
    return pd.DataFrame()


def get_backtest(target):
    for key in ["backtest", "best"]:
        df = data.get(key, pd.DataFrame())
        if df.empty:
            continue
        tc = find_col(df, "target", "signal", "name")
        if tc:
            sub = df[df[tc].astype(str) == str(target)].copy()
            if not sub.empty:
                return sub
    return pd.DataFrame()


def get_hist(target):
    df = data["hist_indexed"]
    if df.empty or target not in df.columns:
        return pd.Series(dtype=float)
    return df[target].dropna()


def parse_summary(row):
    if not row:
        return None, None, None, None, None

    current      = row.get("Forecast start")
    end_val      = row.get("Forecast end")
    exp_change   = row.get("Expected change")
    exp_pct      = row.get("Expected percent change")
    best_model   = row.get("Selected model")

    try: current    = round(float(current), 1)
    except Exception: current = None
    try: end_val    = round(float(end_val), 1)
    except Exception: end_val = None
    try: exp_change = round(float(exp_change), 2)
    except Exception: exp_change = None
    try: exp_pct    = round(float(exp_pct), 1)
    except Exception: exp_pct = None

    if exp_change is not None:
        direction = "Rising" if exp_change > 0 else "Declining"
    else:
        direction = "N/A"

    return current, end_val, exp_change, direction, best_model


def build_brief(target, row, bdf):
    current, end_val, exp_change, direction, best_model = parse_summary(row)
    is_rising = direction == "Rising"

    c_str   = f"{current:.1f}"    if current    is not None else "N/A"
    e_str   = f"{end_val:.1f}"    if end_val    is not None else "N/A"
    ch_str  = f"{exp_change:+.2f}" if exp_change is not None else "N/A"

    mac = find_col(bdf, "mae")
    bm  = best_model or "the best performing model"
    mae_str = ""
    if not bdf.empty and mac:
        mc = find_col(bdf, "model")
        if mc:
            best_row = bdf.sort_values(mac).iloc[0]
            bm       = best_row[mc]
            mae_str  = f" with a mean absolute error of <strong>{float(best_row[mac]):.2f}</strong>"

    rec = (
        "Increase investment in this signal before the upward trend becomes visible to competitors."
        if is_rising else
        "Monitor closely and consider reallocating budget toward rising signals until the trend reverses."
    )

    parts = [
        f"The attention level at the start of the forecast period was "
        f"<strong>{c_str} out of 100</strong>.",
        f"By the end of the six-month window it is projected to reach "
        f"<strong>{e_str} out of 100</strong>, an expected change of "
        f"<strong>{ch_str} points</strong>.",
        f"The overall forecast direction is <strong>{direction.lower()}</strong>.",
        f"The best performing model for this signal was <strong>{bm}</strong>{mae_str} "
        f"on the 26-week backtest.",
        f"<br><strong>Recommendation:</strong> {rec}",
    ]
    return " ".join(parts)


def forecast_chart(target, color):
    hist   = get_hist(target)
    future = get_future(target)
    fig    = go.Figure()

    if not hist.empty:
        h52 = hist.iloc[-52:]
        fig.add_trace(go.Scatter(
            x=h52.index, y=h52.values, name="Historical",
            line=dict(color=color, width=2.2),
            hovertemplate="%{y:.1f}<extra></extra>"
        ))

    if not future.empty:
        date_c = find_col(future, "date", "ds", "week", "time")
        val_c  = find_col(future, "forecast", "yhat", "pred", "value", "mean", "end")
        lo_c   = find_col(future, "lower", "lo", "yhat_lower")
        hi_c   = find_col(future, "upper", "hi", "yhat_upper")

        if date_c and val_c:
            fig.add_trace(go.Scatter(
                x=future[date_c], y=future[val_c],
                name="6-month forecast",
                line=dict(color=color, dash="dash", width=2.5),
                hovertemplate="%{y:.1f}<extra></extra>"
            ))
            if lo_c and hi_c:
                h = color.lstrip("#")
                if len(h) == 6:
                    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
                    fc = f"rgba({r},{g},{b},0.10)"
                else:
                    fc = "rgba(99,102,241,0.10)"
                fig.add_trace(go.Scatter(x=future[date_c], y=future[hi_c],
                    fill=None, mode="lines", line=dict(width=0), showlegend=False))
                fig.add_trace(go.Scatter(x=future[date_c], y=future[lo_c],
                    fill="tonexty", mode="lines", line=dict(width=0),
                    name="Confidence range", fillcolor=fc))

        if not hist.empty:
            fig.add_vline(x=hist.index[-1], line_dash="dot",
                line_color="rgba(255,255,255,0.15)",
                annotation_text="Forecast starts here",
                annotation_font_color="rgba(255,255,255,0.3)",
                annotation_position="top left")

    fig.update_layout(
        yaxis_title="Search Interest (0–100)", hovermode="x unified",
        height=400, margin=dict(l=0, r=0, t=20, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#707090", family="Inter"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
        xaxis=dict(gridcolor="#1a1a2e"), yaxis=dict(gridcolor="#1a1a2e"),
    )
    return fig


def model_chart(target):
    bdf = get_backtest(target)
    if bdf.empty:
        return None, bdf
    mc  = find_col(bdf, "model")
    mac = find_col(bdf, "mae")
    rmc = find_col(bdf, "rmse")
    if not mc or not mac:
        return None, bdf

    bdf = bdf.sort_values(mac)
    cols = 2 if rmc else 1
    titles = ["MAE — lower is better", "RMSE — lower is better"] if rmc else ["MAE — lower is better"]
    fig = make_subplots(rows=1, cols=cols, subplot_titles=titles, horizontal_spacing=0.14)
    fig.add_trace(go.Bar(x=bdf[mac], y=bdf[mc], orientation="h",
        marker=dict(color="#6366f1", opacity=0.85), name="MAE",
        hovertemplate="%{x:.2f}<extra></extra>"), row=1, col=1)
    if rmc:
        fig.add_trace(go.Bar(x=bdf[rmc], y=bdf[mc], orientation="h",
            marker=dict(color="#f59e0b", opacity=0.85), name="RMSE",
            hovertemplate="%{x:.2f}<extra></extra>"), row=1, col=2)

    fig.update_layout(height=240, showlegend=False,
        margin=dict(l=0, r=0, t=36, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#707090", family="Inter"),
        xaxis=dict(gridcolor="#1a1a2e"), yaxis=dict(gridcolor="#1a1a2e"))
    if rmc:
        fig.update_layout(xaxis2=dict(gridcolor="#1a1a2e"),
                          yaxis2=dict(gridcolor="#1a1a2e"))
    fig.update_annotations(font_color="#505080", font_size=11)
    return fig, bdf


def all_platforms_chart():
    df = data["hist_indexed"]
    if df.empty:
        return None
    fig = go.Figure()
    for p in [pl for pl in PLATFORMS if pl in df.columns]:
        s = df[p].iloc[-52:]
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=p,
            line=dict(color=PLATFORM_COLORS.get(p, "#6366f1"), width=1.8),
            hovertemplate=f"{p}: %{{y:.1f}}<extra></extra>"))
    fig.update_layout(height=320, hovermode="x unified",
        margin=dict(l=0, r=0, t=12, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#707090", family="Inter"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="#1a1a2e"),
        yaxis=dict(gridcolor="#1a1a2e", title="Search Interest (0–100)"))
    return fig


def all_signals_bar():
    df = data["summary"]
    if df.empty or "Target" not in df.columns:
        return None
    ec = "Expected change"
    fe = "Forecast end"
    if fe not in df.columns:
        return None
    df = df.copy()
    if ec in df.columns:
        df["bar_color"] = df[ec].apply(
            lambda x: "#22c55e" if float(x) > 0 else "#ef4444"
        )
    else:
        df["bar_color"] = "#6366f1"
    df_s = df.sort_values(fe, ascending=True)
    fig = go.Figure(go.Bar(
        x=df_s[fe], y=df_s["Target"], orientation="h",
        marker_color=df_s["bar_color"],
        hovertemplate="%{y}: %{x:.1f}<extra></extra>"
    ))
    fig.update_layout(height=max(320, len(df)*28),
        margin=dict(l=0, r=0, t=12, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#707090", family="Inter"),
        xaxis=dict(gridcolor="#1a1a2e", title="Forecasted Attention at 6 Months (0–100)"),
        yaxis=dict(gridcolor="#1a1a2e"))
    return fig


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='font-family:Space Grotesk,sans-serif'>"
        "<span style='font-size:22px;font-weight:700;color:#e0e0ff'>📡 Attention</span><br>"
        "<span style='font-size:22px;font-weight:700;color:#6366f1'>Forecasting</span><br>"
        "<span style='font-size:11px;color:#404070;display:block;margin-top:4px'>"
        "Sarfraz Hussain</span></div>",
        unsafe_allow_html=True
    )
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    st.markdown(
        "<p style='font-size:10px;font-weight:600;letter-spacing:0.14em;"
        "text-transform:uppercase;color:#404070;margin-bottom:6px'>Signal Type</p>",
        unsafe_allow_html=True
    )
    signal_type = st.radio("Signal Type", ["Platform", "Content Category"],
                           label_visibility="collapsed")
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    if signal_type == "Platform":
        st.markdown(
            "<p style='font-size:10px;font-weight:600;letter-spacing:0.14em;"
            "text-transform:uppercase;color:#404070;margin-bottom:6px'>Platform</p>",
            unsafe_allow_html=True
        )
        selected = st.selectbox("Platform", PLATFORMS, label_visibility="collapsed")
        color    = PLATFORM_COLORS.get(selected, "#6366f1")
    else:
        st.markdown(
            "<p style='font-size:10px;font-weight:600;letter-spacing:0.14em;"
            "text-transform:uppercase;color:#404070;margin-bottom:6px'>Category</p>",
            unsafe_allow_html=True
        )
        selected = st.selectbox("Category", CATEGORIES, label_visibility="collapsed")
        color    = CATEGORY_COLORS[CATEGORIES.index(selected) % len(CATEGORY_COLORS)]

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:11px;color:#404070;line-height:2'>"
        "Data: Google Trends · Reddit<br>"
        "Models: ARIMA · Prophet · LSTM · TFT<br>"
        f"{datetime.today().strftime('%B %d, %Y')}</div>",
        unsafe_allow_html=True
    )


# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='big-title'>Human Attention<br>Forecasting</div>",
    unsafe_allow_html=True
)

row   = get_summary(selected)
bdf_r = get_backtest(selected)

current, end_val, exp_change, direction, best_model = parse_summary(row)

is_rising = direction == "Rising"
dir_color = "#22c55e" if is_rising else ("#ef4444" if direction == "Declining" else "#6366f1")

st.markdown(
    f"<div style='margin:12px 0 4px'>"
    f"<span class='tag'>{signal_type}</span>"
    f"<span class='tag' style='color:{dir_color};border-color:{dir_color}55'>"
    f"{direction}</span></div>"
    f"<div class='signal-name'>{selected}</div>",
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)

def mcard(col, label, value, sub, vcolor="#a0a0ff"):
    col.markdown(
        f"<div class='mcard'>"
        f"<div class='mcard-label'>{label}</div>"
        f"<div class='mcard-value' style='color:{vcolor}'>{value}</div>"
        f"<div class='mcard-sub'>{sub}</div>"
        f"</div>",
        unsafe_allow_html=True
    )

c_str  = f"{current:.1f}/100"  if current   is not None else "N/A"
e_str  = f"{end_val:.1f}/100"  if end_val   is not None else "N/A"
ch_str = f"{exp_change:+.2f}" if exp_change is not None else "N/A"
m_col  = "#22c55e" if (exp_change or 0) > 0 else "#ef4444"

mcard(c1, "Attention at Forecast Start", c_str,  "starting point")
mcard(c2, "Attention at 6 Months",       e_str,  "forecast endpoint", "#a0a0ff")
mcard(c3, "Forecast Direction",          direction, "next 6 months",  dir_color)
mcard(c4, "Expected Change",             ch_str, "absolute shift",   m_col)

st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📈  Forecast", "🏆  Model Comparison", "💡  Intelligence Brief", "🌐  Overview"]
)

with tab1:
    st.markdown(
        "<p style='color:#505080;font-size:13px;margin-bottom:12px'>"
        "Historical signal (last 52 weeks) followed by the true six-month forward forecast. "
        "The dotted line marks where the data ends and the prediction begins.</p>",
        unsafe_allow_html=True
    )
    st.plotly_chart(forecast_chart(selected, color), use_container_width=True)

    hist_s = get_hist(selected)
    if not hist_s.empty:
        d1, d2, d3 = st.columns(3)
        d1.metric("Recent 8-Week Average", f"{hist_s.iloc[-8:].mean():.1f}")
        d2.metric("Historical Average",    f"{hist_s.mean():.1f}")
        d3.metric("Historical Peak",       f"{hist_s.max():.1f}")

with tab2:
    st.markdown(
        "<p style='color:#505080;font-size:13px;margin-bottom:12px'>"
        "All models evaluated on the same 26-week held-out test period. "
        "Lower MAE and RMSE indicate better performance.</p>",
        unsafe_allow_html=True
    )
    mc_result = model_chart(selected)
    if mc_result[0] is not None:
        st.plotly_chart(mc_result[0], use_container_width=True)
        bdf_show = mc_result[1]
        if not bdf_show.empty:
            mc2  = find_col(bdf_show, "model")
            mac2 = find_col(bdf_show, "mae")
            rmc2 = find_col(bdf_show, "rmse")
            if mc2 and mac2:
                show = [mc2, mac2] + ([rmc2] if rmc2 else [])
                pretty = bdf_show[show].sort_values(mac2).reset_index(drop=True)
                pretty.columns = ["Model","MAE"] + (["RMSE"] if rmc2 else [])
                st.dataframe(
                    pretty.style.format({c:"{:.2f}" for c in ["MAE","RMSE"] if c in pretty.columns}),
                    use_container_width=True, hide_index=True
                )
    else:
        st.info("Model comparison data not available for this selection.")

with tab3:
    st.markdown(
        "<p style='color:#505080;font-size:13px;margin-bottom:12px'>"
        "Generated from computed forecast values for this signal. "
        "Every number comes directly from the model outputs.</p>",
        unsafe_allow_html=True
    )
    if row:
        brief_html = build_brief(selected, row, bdf_r)
        st.markdown(
            f"<div class='brief-box'>{brief_html}</div>",
            unsafe_allow_html=True
        )
    else:
        st.info("Forecast summary data not available for this selection.")

with tab4:
    st.markdown(
        "<p style='color:#505080;font-size:13px;margin-bottom:12px'>"
        "Six-month forecasted attention endpoint across all tracked signals. "
        "Green bars are forecast to rise, red to decline.</p>",
        unsafe_allow_html=True
    )
    sc = all_signals_bar()
    if sc:
        st.plotly_chart(sc, use_container_width=True)

    if signal_type == "Platform":
        st.markdown(
            "<p class='section-label'>All Platforms — Last 52 Weeks</p>",
            unsafe_allow_html=True
        )
        apc = all_platforms_chart()
        if apc:
            st.plotly_chart(apc, use_container_width=True)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown(
    "<div style='margin-top:40px;padding-top:20px;border-top:1px solid #1a1a2e;"
    "display:flex;justify-content:space-between;color:#303050;font-size:11px'>"
    "<span>Human Attention Forecasting</span>"
    "<span>Generative AI and Strategic Forecasting · Sarfraz Hussain</span>"
    "</div>",
    unsafe_allow_html=True
)
