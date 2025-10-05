import streamlit as st
import pandas as pd

st.set_page_config(page_title="GatorAI Dashboard", layout="wide")

st.title("GatorAI Dashboard Prototype")

st.markdown("Load data, run a simple backtest, and visualize results.")

uploaded = st.file_uploader("Upload returns CSV (wide format)", type=["csv"])

if uploaded is not None:
	ret = pd.read_csv(uploaded, index_col=0, parse_dates=True)
	st.write("Preview:", ret.head())
	st.line_chart(ret.cumsum())
else:
	st.info("Upload a CSV of returns to preview cumulative returns.")
