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
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
}

.stApp {
    background: #0d0d14;
    color: #e8e8f0;
}

section[data-testid="stSidebar"] {
    background: #13131f;
    border-right: 1px solid #1e1e30;
}

.metric-row {
    display: flex;
    gap: 16px;
    margin-bottom: 24px;
}

.metric-card {
    background: #13131f;
    border: 1px solid #1e1e30;
    border-radius: 12px;
    padding: 20px 24px;
    flex: 1;
    transition: border-color 0.2s;
}

.metric-card:hover {
    border-color: #3d3d6b;
}

.metric-label {
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #6b6b8a;
    margin-bottom: 8px;
}

.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 28px;
    font-weight: 700;
    color: #e8e8f0;
    line-height: 1;
}

.metric-sub {
    font-size: 12px;
    color: #6b6b8a;
    margin-top: 6px;
}

.rising   { border-top: 2px solid #22c55e; }
.declining { border-top: 2px solid #ef4444; }
.neutral  { border-top: 2px solid #6366f1; }

.brief-box {
    background: #13131f;
    border: 1px solid #1e1e30;
    border-left: 3px solid #6366f1;
    border-radius: 12px;
    padding: 28px 32px;
    margin-top: 8px;
    line-height: 1.8;
    color: #c8c8d8;
}

.brief-box strong {
    color: #e8e8f0;
}

.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #6b6b8a;
    margin: 32px 0 16px 0;
}

.tag {
    display: inline-block;
    background: #1e1e30;
    border: 1px solid #2e2e4a;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 11px;
    color: #9090b0;
    margin-right: 6px;
    font-weight: 500;
}

.stSelectbox label, .stRadio label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #6b6b8a !important;
}

div[data-testid="stMetric"] {
    background: #13131f;
    border: 1px solid #1e1e30;
    border-radius: 12px;
    padding: 16px 20px;
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
    "YouTube"   : "#FF4444",
    "TikTok"    : "#69C9D0",
    "Instagram" : "#E1306C",
    "LinkedIn"  : "#0A8FC4",
    "Reddit"    : "#FF6534"
}

CATEGORY_COLORS = [
    "#6366f1", "#8b5cf6", "#a78bfa", "#c4b5fd",
    "#06b6d4", "#0891b2", "#0e7490", "#22c55e",
    "#f59e0b", "#ef4444"
]


@st.cache_data
def load_data():
    dfs = {}
    files = {
        "future"   : "future_forecasts.csv",
        "summary"  : "future_summary.csv",
        "backtest" : "backtest_model_results.csv",
        "model"    : "model_summary.csv",
        "trends"   : "platform_trends.csv",
    }
    for key, fname in files.items():
        try:
            df = pd.read_csv(DATA_DIR / fname)
            if key in ("future", "trends"):
                for col in df.columns:
                    if "date" in col.lower() or col.lower() == "ds":
                        df[col] = pd.to_datetime(df[col], errors="coerce")
            dfs[key] = df
        except Exception:
            dfs[key] = pd.DataFrame()

    try:
        hist = pd.read_csv(DATA_DIR / "platform_trends.csv", index_col=0, parse_dates=True)
        dfs["hist"] = hist
    except Exception:
        dfs["hist"] = pd.DataFrame()

    return dfs


data = load_data()


def find_col(df, *hints):
    for hint in hints:
        for col in df.columns:
            if hint.lower() in col.lower():
                return col
    return None


def get_future(target):
    df = data["future"]
    if df.empty:
        return pd.DataFrame()
    tc = find_col(df, "target", "signal", "name")
    if tc is None:
        return pd.DataFrame()
    return df[df[tc] == target].copy()


def get_summary(target):
    df = data["summary"]
    if df.empty:
        return {}
    tc = find_col(df, "target", "signal", "name")
    if tc is None:
        return {}
    rows = df[df[tc] == target]
    return rows.iloc[0].to_dict() if not rows.empty else {}


def get_models(target):
    df = data["model"] if not data["model"].empty else data["backtest"]
    if df.empty:
        return pd.DataFrame()
    tc = find_col(df, "target", "signal", "name")
    if tc is None:
        return df
    return df[df[tc] == target].copy()


def get_hist(target):
    df = data["hist"]
    if df.empty or target not in df.columns:
        return pd.Series(dtype=float)
    return df[target].dropna()


def safe(d, *hints):
    for hint in hints:
        for k in d:
            if hint.lower() in k.lower():
                try:
                    v = d[k]
                    return round(float(v), 2)
                except Exception:
                    return str(d[k])
    return None


def build_brief(target, row, model_df):
    current      = safe(row, "current", "last", "latest")
    direction    = safe(row, "direction", "trend")
    momentum     = safe(row, "momentum")
    avg_forecast = safe(row, "avg_forecast", "avg", "mean_forecast", "forecast")

    direction_str = str(direction) if direction is not None else "stable"
    is_rising     = "ris" in direction_str.lower()

    c_str   = f"{current:.1f}"   if isinstance(current,      float) else str(current)   if current   else "N/A"
    avg_str = f"{avg_forecast:.1f}" if isinstance(avg_forecast, float) else str(avg_forecast) if avg_forecast else "N/A"
    mom_str = f"{momentum:+.2f}" if isinstance(momentum,     float) else str(momentum)  if momentum  else "N/A"

    best_model = None
    best_mae   = None
    if not model_df.empty:
        mc  = find_col(model_df, "model")
        mac = find_col(model_df, "mae")
        if mc and mac:
            best_row   = model_df.sort_values(mac).iloc[0]
            best_model = best_row[mc]
            best_mae   = round(float(best_row[mac]), 2)

    change = "increased" if is_rising else "decreased"
    rec    = (
        "Increase investment in this signal before the upward trend becomes visible to competitors."
        if is_rising else
        "Monitor closely and consider reallocating budget toward rising signals until this trend reverses."
    )

    parts = [
        f"Current attention level is <strong>{c_str} out of 100</strong>.",
        f"The eight-week momentum score is <strong>{mom_str}</strong>, meaning attention has {change} recently.",
        f"The six-month forecast projects a <strong>{direction_str.lower()}</strong> trend "
        f"with an average forecasted value of <strong>{avg_str} out of 100</strong>.",
    ]
    if best_model and best_mae is not None:
        parts.append(
            f"The best performing model for this signal was <strong>{best_model}</strong>, "
            f"achieving a mean absolute error of <strong>{best_mae}</strong> points on the 26-week backtest."
        )
    parts.append(f"<br><strong>Recommendation:</strong> {rec}")

    return " ".join(parts)


def forecast_chart(target, color):
    hist   = get_hist(target)
    future = get_future(target)
    fig    = go.Figure()

    if not hist.empty:
        h52 = hist.iloc[-52:]
        fig.add_trace(go.Scatter(
            x=h52.index, y=h52.values,
            name="Historical",
            line=dict(color=color, width=2),
            hovertemplate="%{y:.1f}<extra></extra>"
        ))

    if not future.empty:
        date_c = find_col(future, "date", "ds", "week")
        val_c  = find_col(future, "forecast", "yhat", "pred", "value")
        lo_c   = find_col(future, "lower", "lo", "yhat_lower")
        hi_c   = find_col(future, "upper", "hi", "yhat_upper")

        if date_c and val_c:
            fig.add_trace(go.Scatter(
                x=future[date_c], y=future[val_c],
                name="6-month forecast",
                line=dict(color=color, dash="dash", width=2.5),
                hovertemplate="%{y:.1f}<extra></extra>"
            ))

            if hi_c and lo_c:
                rgba = color.lstrip("#")
                r, g, b = int(rgba[0:2], 16), int(rgba[2:4], 16), int(rgba[4:6], 16)
                fill_color = f"rgba({r},{g},{b},0.12)"

                fig.add_trace(go.Scatter(
                    x=future[date_c], y=future[hi_c],
                    fill=None, mode="lines",
                    line=dict(width=0), showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=future[date_c], y=future[lo_c],
                    fill="tonexty", mode="lines",
                    line=dict(width=0),
                    name="Confidence range",
                    fillcolor=fill_color
                ))

        if not hist.empty:
            fig.add_vline(
                x=hist.index[-1],
                line_dash="dot",
                line_color="rgba(255,255,255,0.2)",
                annotation_text="Forecast begins",
                annotation_font_color="rgba(255,255,255,0.4)",
                annotation_position="top left"
            )

    fig.update_layout(
        yaxis_title="Search Interest (0–100)",
        hovermode="x unified",
        height=400,
        margin=dict(l=0, r=0, t=32, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#9090b0", family="DM Sans"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="#1e1e30", showgrid=True),
        yaxis=dict(gridcolor="#1e1e30", showgrid=True),
    )
    return fig


def model_chart(target):
    mdf = get_models(target)
    if mdf.empty:
        return None

    mc  = find_col(mdf, "model")
    mac = find_col(mdf, "mae")
    rmc = find_col(mdf, "rmse")
    if not mc or not mac:
        return None

    mdf = mdf.sort_values(mac)
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["MAE — lower is better", "RMSE — lower is better"],
        horizontal_spacing=0.12
    )
    fig.add_trace(go.Bar(
        x=mdf[mac], y=mdf[mc], orientation="h",
        marker=dict(color="#6366f1", opacity=0.85),
        name="MAE", hovertemplate="%{x:.2f}<extra></extra>"
    ), row=1, col=1)

    if rmc:
        fig.add_trace(go.Bar(
            x=mdf[rmc], y=mdf[mc], orientation="h",
            marker=dict(color="#f59e0b", opacity=0.85),
            name="RMSE", hovertemplate="%{x:.2f}<extra></extra>"
        ), row=1, col=2)

    fig.update_layout(
        height=240,
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#9090b0", family="DM Sans"),
        xaxis=dict(gridcolor="#1e1e30"),
        yaxis=dict(gridcolor="#1e1e30"),
        xaxis2=dict(gridcolor="#1e1e30"),
        yaxis2=dict(gridcolor="#1e1e30"),
    )
    fig.update_annotations(font_color="#6b6b8a", font_size=12)
    return fig


def all_platforms_chart():
    df = data["hist"]
    if df.empty:
        return None
    fig = go.Figure()
    cols  = [p for p in PLATFORMS if p in df.columns]
    clrs  = [PLATFORM_COLORS.get(p, "#6366f1") for p in cols]
    for plat, color in zip(cols, clrs):
        s = df[plat].iloc[-52:]
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, name=plat,
            line=dict(color=color, width=1.8),
            hovertemplate=f"{plat}: %{{y:.1f}}<extra></extra>"
        ))
    fig.update_layout(
        height=340,
        hovermode="x unified",
        margin=dict(l=0, r=0, t=16, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#9090b0", family="DM Sans"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="#1e1e30"),
        yaxis=dict(gridcolor="#1e1e30", title="Search Interest (0–100)"),
    )
    return fig


def future_summary_chart():
    df = data["summary"]
    if df.empty:
        return None

    tc  = find_col(df, "target", "signal")
    avc = find_col(df, "avg_forecast", "avg", "mean")
    dc  = find_col(df, "direction")
    if not tc or not avc:
        return None

    df = df.copy()
    df["color"] = df[dc].apply(
        lambda x: "#22c55e" if "ris" in str(x).lower() else "#ef4444"
    ) if dc else "#6366f1"

    df_sorted = df.sort_values(avc, ascending=True)

    fig = go.Figure(go.Bar(
        x=df_sorted[avc],
        y=df_sorted[tc],
        orientation="h",
        marker_color=df_sorted["color"] if "color" in df_sorted else "#6366f1",
        hovertemplate="%{y}: %{x:.1f}<extra></extra>"
    ))
    fig.update_layout(
        height=max(300, len(df) * 30),
        margin=dict(l=0, r=0, t=16, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#9090b0", family="DM Sans"),
        xaxis=dict(gridcolor="#1e1e30", title="Avg Forecasted Attention (0–100)"),
        yaxis=dict(gridcolor="#1e1e30"),
    )
    return fig


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<p style='font-family:Syne;font-size:20px;font-weight:800;"
        "color:#e8e8f0;margin-bottom:4px'>📡 Attention</p>"
        "<p style='font-family:Syne;font-size:20px;font-weight:800;"
        "color:#6366f1;margin-top:0;margin-bottom:2px'>Forecasting</p>"
        "<p style='font-size:11px;color:#6b6b8a;margin-top:0;margin-bottom:24px'>"
        "Sarfraz Hussain</p>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p style='font-size:11px;font-weight:600;letter-spacing:0.12em;"
        "text-transform:uppercase;color:#6b6b8a;margin-bottom:8px'>"
        "Signal Type</p>",
        unsafe_allow_html=True
    )
    signal_type = st.radio(
        "Signal Type",
        ["Platform", "Content Category"],
        label_visibility="collapsed"
    )

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    if signal_type == "Platform":
        st.markdown(
            "<p style='font-size:11px;font-weight:600;letter-spacing:0.12em;"
            "text-transform:uppercase;color:#6b6b8a;margin-bottom:8px'>"
            "Platform</p>",
            unsafe_allow_html=True
        )
        selected = st.selectbox("Platform", PLATFORMS, label_visibility="collapsed")
        color    = PLATFORM_COLORS.get(selected, "#6366f1")
    else:
        st.markdown(
            "<p style='font-size:11px;font-weight:600;letter-spacing:0.12em;"
            "text-transform:uppercase;color:#6b6b8a;margin-bottom:8px'>"
            "Content Category</p>",
            unsafe_allow_html=True
        )
        selected = st.selectbox("Category", CATEGORIES, label_visibility="collapsed")
        idx      = CATEGORIES.index(selected)
        color    = CATEGORY_COLORS[idx % len(CATEGORY_COLORS)]

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:11px;color:#4a4a6a'>Data: Google Trends + Reddit</p>"
        "<p style='font-size:11px;color:#4a4a6a'>Models: ARIMA · Prophet · LSTM · TFT</p>"
        f"<p style='font-size:11px;color:#4a4a6a'>{datetime.today().strftime('%B %d, %Y')}</p>",
        unsafe_allow_html=True
    )


# ─── MAIN ─────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='font-family:Syne;font-size:32px;font-weight:800;"
    "color:#e8e8f0;margin-bottom:4px'>Human Attention Forecasting</h1>",
    unsafe_allow_html=True
)

row    = get_summary(selected)
mdf    = get_models(selected)
hist_s = get_hist(selected)

current      = safe(row, "current", "last", "latest")
direction    = safe(row, "direction", "trend")
momentum     = safe(row, "momentum")
avg_forecast = safe(row, "avg_forecast", "avg", "mean_forecast", "forecast")

direction_str = str(direction) if direction is not None else "N/A"
is_rising     = "ris" in direction_str.lower()
dir_color     = "#22c55e" if is_rising else "#ef4444"

tag_html  = f'<span class="tag">{signal_type}</span>'
tag_html += f'<span class="tag" style="color:{dir_color};border-color:{dir_color}40">{direction_str}</span>'
st.markdown(
    f"<div style='margin-bottom:4px'>{tag_html}</div>"
    f"<h2 style='font-family:Syne;font-size:26px;font-weight:700;"
    f"color:#e8e8f0;margin-top:0;margin-bottom:24px'>{selected}</h2>",
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)

def metric_card(col, label, value, sub="", accent="#6366f1"):
    card_class = "rising" if "ris" in str(value).lower() else "declining" if "dec" in str(value).lower() else "neutral"
    col.markdown(
        f"<div class='metric-card {card_class}'>"
        f"<div class='metric-label'>{label}</div>"
        f"<div class='metric-value' style='color:{accent}'>{value}</div>"
        f"<div class='metric-sub'>{sub}</div>"
        f"</div>",
        unsafe_allow_html=True
    )

c_str   = f"{current:.1f}"      if isinstance(current,      float) else "N/A"
avg_str = f"{avg_forecast:.1f}" if isinstance(avg_forecast, float) else "N/A"
mom_str = f"{momentum:+.2f}"    if isinstance(momentum,     float) else "N/A"

metric_card(c1, "Current Attention",    f"{c_str}/100",   "out of 100")
metric_card(c2, "6-Month Avg Forecast", f"{avg_str}/100", "projected")
metric_card(c3, "Forecast Direction",   direction_str,    "next 6 months", dir_color)
metric_card(c4, "8-Week Momentum",      mom_str,          "trend strength",
            "#22c55e" if isinstance(momentum, float) and momentum > 0 else "#ef4444")

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈  Forecast", "🏆  Model Comparison", "💡  Intelligence Brief", "🌐  Overview"
])

with tab1:
    st.markdown(
        "<p style='color:#6b6b8a;font-size:13px;margin-bottom:16px'>"
        "Historical signal (last 52 weeks) followed by the true six-month forward forecast. "
        "The dotted line marks where the data ends and the prediction begins.</p>",
        unsafe_allow_html=True
    )
    fig = forecast_chart(selected, color)
    st.plotly_chart(fig, use_container_width=True)

    if not hist_s.empty:
        recent_avg   = round(float(hist_s.iloc[-8:].mean()),  1)
        all_time_avg = round(float(hist_s.mean()),             1)
        all_time_max = round(float(hist_s.max()),              1)
        d1, d2, d3   = st.columns(3)
        d1.metric("Recent 8-Week Average", f"{recent_avg}")
        d2.metric("All-Time Average",      f"{all_time_avg}")
        d3.metric("All-Time Peak",         f"{all_time_max}")

with tab2:
    st.markdown(
        "<p style='color:#6b6b8a;font-size:13px;margin-bottom:16px'>"
        "All four models evaluated on the same 26-week held-out test period. "
        "Lower MAE and RMSE indicate better performance.</p>",
        unsafe_allow_html=True
    )
    mc = model_chart(selected)
    if mc:
        st.plotly_chart(mc, use_container_width=True)

        if not mdf.empty:
            mac = find_col(mdf, "mae")
            rmc = find_col(mdf, "rmse")
            mc2 = find_col(mdf, "model")
            if mc2 and mac:
                display_cols = [mc2, mac] + ([rmc] if rmc else [])
                pretty = mdf[display_cols].sort_values(mac).reset_index(drop=True)
                pretty.columns = ["Model", "MAE"] + (["RMSE"] if rmc else [])
                st.dataframe(
                    pretty.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}"}),
                    use_container_width=True, hide_index=True
                )
    else:
        st.info("Model comparison data not available for this selection.")

with tab3:
    st.markdown(
        "<p style='color:#6b6b8a;font-size:13px;margin-bottom:16px'>"
        "Generated from computed forecast values — current level, momentum, "
        "direction, and model accuracy. All numbers come from the models.</p>",
        unsafe_allow_html=True
    )
    if row:
        brief_html = build_brief(selected, row, mdf)
        st.markdown(
            f"<div class='brief-box'>{brief_html}</div>",
            unsafe_allow_html=True
        )
    else:
        st.info("Forecast summary data not available for this selection.")

with tab4:
    st.markdown(
        "<p style='color:#6b6b8a;font-size:13px;margin-bottom:20px'>"
        "Six-month average forecasted attention across all tracked signals, "
        "ranked by predicted value. Green bars are forecast to rise, red to decline.</p>",
        unsafe_allow_html=True
    )
    sc = future_summary_chart()
    if sc:
        st.plotly_chart(sc, use_container_width=True)

    if signal_type == "Platform":
        st.markdown(
            "<p class='section-title'>All Platforms — Last 52 Weeks</p>",
            unsafe_allow_html=True
        )
        apc = all_platforms_chart()
        if apc:
            st.plotly_chart(apc, use_container_width=True)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown(
    "<div style='margin-top:48px;padding-top:20px;border-top:1px solid #1e1e30;"
    "color:#4a4a6a;font-size:11px;display:flex;justify-content:space-between'>"
    "<span>Human Attention Forecasting — Final Project</span>"
    "<span>Generative AI and Strategic Forecasting · Sarfraz Hussain</span>"
    "</div>",
    unsafe_allow_html=True
)
