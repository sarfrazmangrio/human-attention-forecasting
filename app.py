
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Human Attention Forecasting", page_icon="📈", layout="wide")

DATA_DIR = Path(".")

@st.cache_data
def load_data():
    model_data = pd.read_csv(DATA_DIR / "modeling_attention_data.csv", index_col=0, parse_dates=True)
    future_forecasts = pd.read_csv(DATA_DIR / "future_forecasts.csv", parse_dates=["Date"])
    future_summary = pd.read_csv(DATA_DIR / "future_summary.csv")
    results_df = pd.read_csv(DATA_DIR / "backtest_model_results.csv")
    model_summary = pd.read_csv(DATA_DIR / "model_summary.csv")
    source_status = pd.read_csv(DATA_DIR / "source_status.csv")
    brief = (DATA_DIR / "attention_intelligence_brief.txt").read_text(encoding="utf-8")
    return model_data, future_forecasts, future_summary, results_df, model_summary, source_status, brief

model_data, future_forecasts, future_summary, results_df, model_summary, source_status, brief = load_data()

st.title("Human Attention Forecasting")
st.caption("Forecasting platforms and content categories using public attention signals")

st.write("This dashboard uses the forecast outputs exported from the final notebook.")

targets = future_summary["Target"].tolist()
selected_target = st.sidebar.selectbox("Select target", targets)

summary_row = future_summary[future_summary["Target"] == selected_target].iloc[0]
target_future = future_forecasts[future_forecasts["Target"] == selected_target]
target_results = results_df[results_df["Target"] == selected_target].sort_values("MAE")

col1, col2, col3 = st.columns(3)
col1.metric("Target group", summary_row["Group"])
col2.metric("Selected model", summary_row["Selected model"])
col3.metric("Expected change", f"{summary_row['Expected percent change']:.2f}%")

fig = go.Figure()
fig.add_trace(go.Scatter(x=model_data.index, y=model_data[selected_target], mode="lines", name="Historical"))
fig.add_trace(go.Scatter(x=target_future["Date"], y=target_future["Forecast"], mode="lines+markers", name="True future forecast"))
fig.update_layout(title=f"True 26 week future forecast for {selected_target}", xaxis_title="Date", yaxis_title="Attention score", template="plotly_white", hovermode="x unified", height=520)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Future forecast summary")
st.dataframe(future_summary.sort_values(["Group", "Expected percent change"], ascending=[True, False]), use_container_width=True)

st.subheader("Backtest model comparison")
st.dataframe(target_results[["Model", "MAE", "RMSE", "MAPE"]].round(3), use_container_width=True)

st.subheader("Average model performance")
st.dataframe(model_summary.round(3), use_container_width=True)

st.subheader("Data source status")
st.dataframe(source_status, use_container_width=True)

st.subheader("Attention Intelligence Brief")
st.text(brief)

st.markdown("### Limitation")
st.write("Google Trends, Wikipedia, and Reddit engagement are proxy signals. They support decision making, but they do not guarantee future behavior.")
