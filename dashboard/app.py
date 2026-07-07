"""NIFTY AI Agent — Streamlit dashboard.

Run with:
    streamlit run dashboard/app.py

Save the URL to your iPhone home screen for a personal signal dashboard.
"""

import sys
from pathlib import Path

# Allow imports from project root when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session

from nifty_ai_agent.config import get_settings
from nifty_ai_agent.database.models import MarketDataRecord, SignalRecord
from nifty_ai_agent.indicators.ema import compute_ema
from nifty_ai_agent.indicators.rsi import compute_rsi
from nifty_ai_agent.indicators.macd import compute_macd
from nifty_ai_agent.strategies.base import SignalType

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NIFTY AI Signal Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS for mobile-friendly look ───────────────────────────────────────
st.markdown(
    """
    <style>
    .signal-box {
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        font-size: 1.4rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1rem;
    }
    .signal-buy-ce  { background: #d4edda; color: #155724; border: 2px solid #28a745; }
    .signal-buy-pe  { background: #f8d7da; color: #721c24; border: 2px solid #dc3545; }
    .signal-hold    { background: #fff3cd; color: #856404; border: 2px solid #ffc107; }
    .metric-card    { background: #f8f9fa; border-radius: 8px; padding: 0.8rem; text-align: center; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── DB helpers ─────────────────────────────────────────────────────────────────

@st.cache_resource
def _get_engine():
    try:
        settings = get_settings()
        db_url = settings.database_url
    except Exception:
        db_url = "sqlite:///nifty_ai_agent.db"
    return create_engine(db_url, connect_args={"check_same_thread": False})


@st.cache_data(ttl=60)
def load_signals(limit: int = 50) -> pd.DataFrame:
    engine = _get_engine()
    with Session(engine) as session:
        stmt = (
            select(SignalRecord)
            .order_by(SignalRecord.datetime.desc())
            .limit(limit)
        )
        rows = session.scalars(stmt).all()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "datetime": r.datetime,
                "signal": r.signal,
                "confidence": r.confidence,
                "strategy": r.strategy,
                "entry": r.entry_price,
                "sl": r.stop_loss,
                "target": r.target,
                "rr": r.risk_reward,
                "reason": r.reason,
                "ai_explanation": r.ai_explanation or "",
                "status": r.status,
            }
            for r in rows
        ]
    )


@st.cache_data(ttl=60)
def load_market_data(limit: int = 120) -> pd.DataFrame:
    engine = _get_engine()
    with Session(engine) as session:
        stmt = (
            select(MarketDataRecord)
            .order_by(MarketDataRecord.datetime.desc())
            .limit(limit)
        )
        rows = session.scalars(stmt).all()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(
        [
            {
                "datetime": r.datetime,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
            }
            for r in rows
        ]
    ).sort_values("datetime").set_index("datetime")
    return df


# ── Main layout ────────────────────────────────────────────────────────────────

def render_header():
    col_title, col_refresh = st.columns([6, 1])
    with col_title:
        st.title("📈 NIFTY AI Signal Agent")
        st.caption("Weekly expiry options · EMA Crossover + VWAP Breakout strategies · Auto-refreshes every 60 s")
    with col_refresh:
        st.write("")
        if st.button("↻ Refresh"):
            st.cache_data.clear()
            st.rerun()


def render_latest_signals(signals_df: pd.DataFrame):
    """Show the latest prediction from every strategy, side by side."""
    st.subheader("Latest Predictions")
    if signals_df.empty:
        st.info("No signals yet. Start the agent with `python main.py`.")
        return

    css_class = {
        "BUY_CE": "signal-buy-ce",
        "BUY_PE": "signal-buy-pe",
        "HOLD": "signal-hold",
    }
    label = {
        "BUY_CE": "📈 BUY CALL (CE)",
        "BUY_PE": "📉 BUY PUT (PE)",
        "HOLD": "⏸ HOLD — No Trade",
    }

    # One column per strategy, most-recently-seen order preserved.
    strategies: list[str] = []
    for s in signals_df["strategy"]:
        if s not in strategies:
            strategies.append(s)

    cols = st.columns(len(strategies))
    for col, strat in zip(cols, strategies):
        latest = signals_df[signals_df["strategy"] == strat].iloc[0]
        signal_val = latest["signal"]
        with col:
            st.markdown(
                f'<div class="signal-box {css_class.get(signal_val, "signal-hold")}">'
                f'{label.get(signal_val, signal_val)}<br>'
                f'<span style="font-size:0.95rem">{latest["confidence"]}% confidence</span></div>',
                unsafe_allow_html=True,
            )
            ts = pd.Timestamp(latest["datetime"])
            st.caption(f"**{strat}** &nbsp;·&nbsp; {ts.strftime('%d %b %Y  %H:%M')} IST")

            entry, sl, target, rr = latest["entry"], latest["sl"], latest["target"], latest["rr"]
            if pd.isna(entry) or entry == 0:
                st.caption("No risk parameters — HOLD signal.")
            else:
                st.write(f"Entry **{entry:,.0f}** · SL **{sl:,.0f}** · Target **{target:,.0f}** · RR 1:{rr}")

            explanation = latest.get("ai_explanation", "")
            reason = latest.get("reason", "")
            if explanation:
                st.info(explanation)
            elif reason:
                st.info(reason)


def render_chart(market_df: pd.DataFrame):
    if market_df.empty:
        st.info("No market data in database yet.")
        return

    st.subheader("NIFTY Price Chart")

    df = compute_ema(market_df, periods=[20, 50])
    df = compute_rsi(df)
    df = compute_macd(df)

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2],
        vertical_spacing=0.03,
        subplot_titles=("NIFTY 50", "RSI (14)", "MACD"),
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"], high=df["high"],
            low=df["low"], close=df["close"],
            name="NIFTY",
            increasing_line_color="#28a745",
            decreasing_line_color="#dc3545",
        ),
        row=1, col=1,
    )
    # EMA lines
    fig.add_trace(
        go.Scatter(x=df.index, y=df["ema_20"], name="EMA 20",
                   line=dict(color="#1f77b4", width=1.5)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["ema_50"], name="EMA 50",
                   line=dict(color="#ff7f0e", width=1.5)),
        row=1, col=1,
    )

    # RSI
    fig.add_trace(
        go.Scatter(x=df.index, y=df["rsi"], name="RSI",
                   line=dict(color="#9467bd", width=1.5)),
        row=2, col=1,
    )
    fig.add_hline(y=60, line_dash="dot", line_color="green", row=2, col=1)
    fig.add_hline(y=40, line_dash="dot", line_color="red", row=2, col=1)

    # MACD
    fig.add_trace(
        go.Scatter(x=df.index, y=df["macd"], name="MACD",
                   line=dict(color="#17becf", width=1.5)),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["macd_signal"], name="Signal",
                   line=dict(color="#e377c2", width=1.2, dash="dash")),
        row=3, col=1,
    )
    fig.add_trace(
        go.Bar(x=df.index, y=df["macd_histogram"], name="Histogram",
               marker_color="#8c564b", opacity=0.4),
        row=3, col=1,
    )

    fig.update_layout(
        height=600,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="white",
        plot_bgcolor="#fafafa",
    )
    fig.update_yaxes(gridcolor="#eeeeee")
    st.plotly_chart(fig, width="stretch")


def render_signal_history(signals_df: pd.DataFrame):
    if signals_df.empty:
        return
    st.subheader("Signal History")

    strategies = sorted(signals_df["strategy"].unique().tolist())
    selected = st.multiselect("Strategy", strategies, default=strategies)
    filtered = signals_df[signals_df["strategy"].isin(selected)] if selected else signals_df

    def _color_signal(val: str) -> str:
        colors = {"BUY_CE": "background-color:#d4edda", "BUY_PE": "background-color:#f8d7da"}
        return colors.get(val, "")

    display = filtered[["datetime", "strategy", "signal", "confidence", "entry", "sl", "target", "rr"]].copy()
    display["datetime"] = pd.to_datetime(display["datetime"]).dt.strftime("%d %b %H:%M")
    display.columns = ["Time", "Strategy", "Signal", "Conf %", "Entry", "SL", "Target", "RR"]

    st.dataframe(
        display.style.map(_color_signal, subset=["Signal"]),
        width="stretch",
        hide_index=True,
    )


def render_indicator_snapshot(market_df: pd.DataFrame):
    if market_df.empty:
        return
    df = compute_ema(market_df, periods=[20, 50])
    df = compute_rsi(df)
    df = compute_macd(df)
    last = df.dropna(subset=["ema_20", "ema_50", "rsi"]).iloc[-1]

    st.subheader("Current Indicators")
    cols = st.columns(5)
    ind = [
        ("EMA 20", f"{last['ema_20']:,.0f}"),
        ("EMA 50", f"{last['ema_50']:,.0f}"),
        ("RSI (14)", f"{last['rsi']:.1f}"),
        ("MACD", f"{last.get('macd', 0):.1f}"),
        ("Close", f"{last['close']:,.0f}"),
    ]
    for col, (label, value) in zip(cols, ind):
        col.metric(label=label, value=value)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    render_header()
    st.divider()

    signals_df = load_signals()
    market_df = load_market_data()

    render_latest_signals(signals_df)

    st.divider()

    # Chart + indicators
    render_indicator_snapshot(market_df)
    st.write("")
    render_chart(market_df)

    st.divider()

    # Signal history table
    render_signal_history(signals_df)

    # Auto-refresh via meta tag
    st.markdown(
        '<meta http-equiv="refresh" content="60">',
        unsafe_allow_html=True,
    )


main()
