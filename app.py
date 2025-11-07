import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

st.set_page_config(page_title="Portfolio Tracker", layout="wide")

# -------------------- Helpers --------------------
@st.cache_data(ttl=60)
def get_last_price(ticker: str) -> float:
    try:
        t = yf.Ticker(ticker)
        fi = getattr(t, "fast_info", None)
        if fi and "lastPrice" in fi and fi["lastPrice"] is not None:
            return float(fi["lastPrice"])
        hist = t.history(period="1d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return np.nan

@st.cache_data(ttl=3600)
def get_ticker_currency(ticker: str) -> str:
    try:
        t = yf.Ticker(ticker)
        fi = getattr(t, "fast_info", None)
        if fi and "currency" in fi and fi["currency"]:
            return fi["currency"]
        info = getattr(t, "info", {}) or {}
        if "currency" in info and info["currency"]:
            return info["currency"]
    except Exception:
        pass
    if ticker.endswith("=X") and len(ticker) >= 6:  # e.g. EURUSD=X
        return ticker[-4:-1]
    return "USD"

@st.cache_data(ttl=300)
def get_fx_rate_to_base(asset_ccy: str, base_ccy: str) -> float:
    if asset_ccy == base_ccy:
        return 1.0
    direct = f"{asset_ccy}{base_ccy}=X"
    r = get_last_price(direct)
    if not np.isnan(r):
        return float(r)
    inverse = f"{base_ccy}{asset_ccy}=X"
    r_inv = get_last_price(inverse)
    if not np.isnan(r_inv) and r_inv != 0:
        return 1.0 / float(r_inv)
    return np.nan

def calc_metrics(df: pd.DataFrame, base_ccy: str) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["price"] = df["ticker"].apply(get_last_price)
    df["asset_ccy"] = df["ticker"].apply(get_ticker_currency)
    df["fx_to_base"] = [get_fx_rate_to_base(ccy, base_ccy) for ccy in df["asset_ccy"]]
    df["value_base"] = df["quantity"] * df["price"] * df["fx_to_base"]

    if "cost_basis" not in df.columns:
        df["cost_basis"] = np.nan

    df["pl_abs_base"] = (df["price"] - df["cost_basis"]) * df["quantity"] * df["fx_to_base"]
    df["pl_pct"] = np.where(
        df["cost_basis"].notna() & (df["cost_basis"] != 0),
        (df["price"] - df["cost_basis"]) / df["cost_basis"] * 100.0,
        np.nan,
    )
    total = df["value_base"].sum(skipna=True)
    df["weight_pct"] = np.where(total > 0, df["value_base"] / total * 100.0, 0.0)

    cols = ["ticker","quantity","asset_ccy","price","fx_to_base","value_base",
            "cost_basis","pl_abs_base","pl_pct","weight_pct"]
    return df[cols]

def empty_portfolio():
    return pd.DataFrame({"ticker": [], "quantity": [], "cost_basis": []})

# -------------------- Sidebar --------------------
st.sidebar.title("Settings")
base_currency = st.sidebar.selectbox("Base currency", ["EUR","USD","GBP"], index=0)

uploaded = st.sidebar.file_uploader("Load portfolio (CSV)", type=["csv"])
if uploaded is not None:
    try:
        st.session_state["portfolio"] = pd.read_csv(uploaded)
        st.sidebar.success("Portfolio loaded.")
    except Exception as e:
        st.sidebar.error(f"Could not read CSV: {e}")

if "portfolio" not in st.session_state:
    st.session_state["portfolio"] = empty_portfolio()

st.sidebar.markdown("---")
st.sidebar.write("**Add position**")
with st.sidebar.form("add_pos"):
    new_ticker = st.text_input("Ticker (e.g., AAPL, ENI.MI, EURUSD=X)").strip().upper()
    new_qty = st.number_input("Quantity / Notional", value=0.0, step=1.0, format="%.4f")
    new_cb = st.number_input("Cost basis (optional)", value=0.0, step=0.01, format="%.4f")
    add_btn = st.form_submit_button("Add")
    if add_btn:
        if new_ticker and new_qty != 0:
            row = {"ticker": new_ticker, "quantity": new_qty}
            if new_cb > 0:
                row["cost_basis"] = new_cb
            st.session_state["portfolio"] = pd.concat(
                [st.session_state["portfolio"], pd.DataFrame([row])],
                ignore_index=True
            )
            st.success(f"Added {new_ticker}")
        else:
            st.warning("Please provide a ticker and a non-zero quantity.")

# -------------------- Main --------------------
st.title("📊 Portfolio Tracker (Yahoo Finance)")
st.caption(f"Base currency: {base_currency} • Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Delete rows UI
if not st.session_state["portfolio"].empty:
    to_delete = st.multiselect(
        "Select rows to delete",
        options=list(range(len(st.session_state["portfolio"]))),
        format_func=lambda i: f'{i}: {st.session_state["portfolio"].iloc[i]["ticker"]}'
    )
    if st.button("Delete selected"):
        st.session_state["portfolio"].drop(index=to_delete, inplace=True)
        st.session_state["portfolio"].reset_index(drop=True, inplace=True)
        st.success("Deleted selected rows.")

df = st.session_state["portfolio"].copy()
metrics = calc_metrics(df, base_currency) if not df.empty else df

if metrics.empty:
    st.info("Add your first position from the sidebar.")
else:
    shown = metrics.copy()
    for col in ["price","fx_to_base","value_base","cost_basis","pl_abs_base"]:
        if col in shown:
            shown[col] = shown[col].astype(float).round(4)
    for col in ["pl_pct","weight_pct"]:
        if col in shown:
            shown[col] = shown[col].astype(float).round(2)

    st.dataframe(shown, use_container_width=True)

    total_val = float(shown["value_base"].sum(skipna=True))
    total_pl  = float(shown["pl_abs_base"].sum(skipna=True)) if "pl_abs_base" in shown else np.nan
    st.markdown(f"**Total value ({base_currency})**: {total_val:,.2f}")
    if not np.isnan(total_pl):
        st.markdown(f"**Total P/L ({base_currency})**: {total_pl:,.2f}")
# ---------- Pie chart: Portfolio allocation ----------
import matplotlib.pyplot as plt

# Mostra il grafico solo se ci sono dati nella tabella
if 'shown' in locals() and not shown.empty:
    alloc = shown[["ticker", "value_base"]].dropna()
    if not alloc.empty and alloc["value_base"].sum() > 0:
        # Aggrega le posizioni piccole in "Other" (<3%)
        alloc = alloc.groupby("ticker", as_index=False)["value_base"].sum()
        total_val_for_pie = alloc["value_base"].sum()
        alloc["weight_pct"] = alloc["value_base"] / total_val_for_pie * 100

        major = alloc[alloc["weight_pct"] >= 3].copy()
        minor = alloc[alloc["weight_pct"] < 3].copy()
        if not minor.empty:
            other_row = pd.DataFrame([{
                "ticker": "Other",
                "value_base": minor["value_base"].sum(),
                "weight_pct": minor["weight_pct"].sum()
            }])
            alloc_plot = pd.concat([major, other_row], ignore_index=True)
        else:
            alloc_plot = major

        labels = [f'{t} ({w:.1f}%)' for t, w in zip(alloc_plot["ticker"], alloc_plot["weight_pct"])]
        fig, ax = plt.subplots()
        ax.pie(alloc_plot["value_base"], labels=labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.subheader("Allocation by position")
        st.pyplot(fig)
else:
    st.info("Add positions to see the allocation chart.")

csv = st.session_state["portfolio"].to_csv(index=False).encode("utf-8")
st.download_button("Download portfolio CSV", data=csv, file_name="portfolio.csv", mime="text/csv")
