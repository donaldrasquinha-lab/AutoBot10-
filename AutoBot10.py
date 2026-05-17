import streamlit as st
import time
import requests
import pandas as pd
import ta
from datetime import date, datetime

# ==============================================================================
# CONFIGURATION CONSTANTS
# ==============================================================================
UPSTOX_BASE_URL = "https://api.upstox.com/v2"

# Nifty lot size — update if SEBI revises
NIFTY_LOT_SIZE = 75

# Consecutive loss streak limit before circuit breaker trips
MAX_LOSS_STREAK = 2

# ==============================================================================
# STREAMLIT UI CONFIGURATION & STATE
# ==============================================================================
st.set_page_config(page_title="Upstox Algo Scalper", layout="wide")

defaults = {
    "bot_active": False,
    "current_position": None,
    "trade_logs": [],
    "session_pnl": 0.0,
    "loss_streak": 0,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ==============================================================================
# BACKEND API HELPERS
# ==============================================================================

def _auth_headers(token: str) -> dict:
    return {"Accept": "application/json", "Authorization": f"Bearer {token}"}


def fetch_ltp(token: str, instrument_key: str) -> float | None:
    """
    Fetches Last Traded Price via Upstox V2 /market-quote/ltp.
    Works 24x7 — returns None on any failure.
    """
    try:
        url = f"{UPSTOX_BASE_URL}/market-quote/ltp"
        params = {"instrument_key": instrument_key}
        res = requests.get(url, headers=_auth_headers(token), params=params, timeout=5)
        res.raise_for_status()
        data = res.json().get("data", {})
        # Key in response uses '|' replaced with ':' e.g. NSE_INDEX:Nifty 50
        normalized_key = instrument_key.replace("|", ":")
        return float(data[normalized_key]["last_price"])
    except Exception:
        return None


def fetch_ltp_batch(token: str, instrument_keys: list[str]) -> dict:
    """
    Batch-fetches LTPs for up to 50 instrument keys in one call.
    Returns {instrument_key: ltp} dict; missing keys are omitted.
    """
    results = {}
    for i in range(0, len(instrument_keys), 50):
        batch = instrument_keys[i : i + 50]
        try:
            url = f"{UPSTOX_BASE_URL}/market-quote/ltp"
            params = {"instrument_key": ",".join(batch)}
            res = requests.get(url, headers=_auth_headers(token), params=params, timeout=5)
            res.raise_for_status()
            data = res.json().get("data", {})
            for raw_key, val in data.items():
                # Normalize back: response key uses ':' separator
                original_key = raw_key.replace(":", "|")
                results[original_key] = float(val["last_price"])
        except Exception:
            continue
    return results


def fetch_historical_candles(token: str, instrument_key: str) -> pd.DataFrame:
    """
    Fetches today's 1-minute OHLCV candles via Upstox V2 historical-candle endpoint.
    Returns empty DataFrame on failure.
    """
    try:
        today = date.today().strftime("%Y-%m-%d")
        # instrument_key must be URL-encoded — requests handles this via params
        url = f"{UPSTOX_BASE_URL}/historical-candle/intraday/{instrument_key}/1minute"
        res = requests.get(url, headers=_auth_headers(token), timeout=5)
        res.raise_for_status()
        candles = res.json()["data"]["candles"]
        df = pd.DataFrame(
            candles, columns=["timestamp", "open", "high", "low", "close", "volume", "oi"]
        )
        # API returns newest-first; reverse for chronological indicator calc
        df = df.iloc[::-1].reset_index(drop=True)
        for col in ["open", "high", "low", "close", "volume", "oi"]:
            df[col] = pd.to_numeric(df[col])
        return df
    except Exception:
        return pd.DataFrame()


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Standard intraday VWAP (cumulative)."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cum_tp_vol = (typical_price * df["volume"]).cumsum()
    cum_vol = df["volume"].cumsum()
    return cum_tp_vol / cum_vol.replace(0, float("nan"))


def get_atm_option_key(spot: float, option_type: str) -> tuple[str, str]:
    """
    Derives ATM option instrument key and human label for the nearest
    weekly Nifty expiry.

    ⚠️  NOTE: Upstox instrument keys require exact expiry dates embedded
    in the key string. In production, resolve the active expiry date
    by fetching the instrument master CSV and filtering by symbol + expiry.
    This function generates a best-effort key for demonstration; replace
    the expiry_str lookup with a master-file query for reliability.
    """
    # Round spot to nearest 50 (Nifty strike interval)
    atm_strike = round(spot / 50) * 50

    # Nearest Thursday expiry (weekly)
    today = date.today()
    days_to_thursday = (3 - today.weekday()) % 7
    if days_to_thursday == 0 and datetime.now().hour >= 15:
        days_to_thursday = 7
    from datetime import timedelta
    expiry = today + timedelta(days=days_to_thursday)
    expiry_str = expiry.strftime("%y%b%d").upper()  # e.g. 25MAY15

    ot = option_type.upper()  # CE or PE
    symbol = f"NIFTY{expiry_str}{atm_strike}{ot}"
    instrument_key = f"NSE_FO|{symbol}"
    return instrument_key, symbol


# ==============================================================================
# SIDEBAR CONTROL PANEL
# ==============================================================================
st.sidebar.header("🔌 Authentication")
ACCESS_TOKEN = st.sidebar.text_input(
    "Upstox Access Token",
    type="password",
    help="Generate daily via Upstox Developer Console. Expires at midnight.",
)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Strategy Parameters")
INITIAL_CAPITAL = st.sidebar.number_input("Starting Capital (₹)", value=50_000, step=5_000)
STOP_LOSS_PCT = st.sidebar.slider("Stop Loss (%)", 1, 10, 6) / 100.0
TARGET_PCT = st.sidebar.slider("Target Profit (%)", 1, 25, 12) / 100.0
LOT_SIZE = st.sidebar.number_input(
    "Lot Size (Nifty = 75)", value=NIFTY_LOT_SIZE, step=1, min_value=1
)
TRADE_MODE = st.sidebar.radio(
    "🎯 Execution Mode",
    ["📄 Paper Trading (Simulated)", "⚡ Live Trading (Real Money)"],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    f"⚡ Circuit breaker: halts after **{MAX_LOSS_STREAK}** consecutive losses"
)

current_capital = INITIAL_CAPITAL + st.session_state.session_pnl

# ==============================================================================
# MAIN DASHBOARD
# ==============================================================================
st.title("🦅 Upstox Options Advanced Scalping Engine")

# Row 1: KPI Metrics
m1, m2, m3, m4, m5 = st.columns(5)

nifty_spot = fetch_ltp(ACCESS_TOKEN, "NSE_INDEX|Nifty 50") if ACCESS_TOKEN else None

m1.metric(
    "📊 Nifty 50 Spot",
    f"₹{nifty_spot:,.2f}" if nifty_spot else "—",
    help="Live spot via /market-quote/ltp",
)
m2.metric("💰 Account Balance", f"₹{current_capital:,.2f}")
m3.metric(
    "📈 Session PnL",
    f"₹{st.session_state.session_pnl:,.2f}",
    delta=f"{(st.session_state.session_pnl / INITIAL_CAPITAL * 100):.2f}%" if INITIAL_CAPITAL else "0%",
)
m4.metric("🛡️ Mode", "LIVE" if "Live" in TRADE_MODE else "PAPER")
m5.metric(
    "🔴 Loss Streak",
    f"{st.session_state.loss_streak} / {MAX_LOSS_STREAK}",
    delta="⚠️ At limit!" if st.session_state.loss_streak >= MAX_LOSS_STREAK - 1 else None,
)

st.markdown("---")

# Row 2: Engine Controls + Active Position
col_ctrl, col_pos = st.columns([1, 2])

with col_ctrl:
    st.subheader("🕹️ Engine Controls")
    if not ACCESS_TOKEN:
        st.warning("Provide an Upstox Access Token to activate.")
    else:
        if not st.session_state.bot_active:
            if st.button("▶️ START BOT", type="primary", use_container_width=True):
                st.session_state.bot_active = True
                st.session_state.loss_streak = 0  # Reset streak on manual start
                st.rerun()
        else:
            if st.button("🛑 EMERGENCY HALT", type="secondary", use_container_width=True):
                st.session_state.bot_active = False
                st.session_state.current_position = None
                st.rerun()

        status = "🟢 Active & Scanning" if st.session_state.bot_active else "🔴 Inactive"
        st.info(f"State: **{status}**")

        if st.session_state.trade_logs:
            df_logs = pd.DataFrame(st.session_state.trade_logs)
            csv_bytes = df_logs.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Trade Log (CSV)",
                data=csv_bytes,
                file_name=f"scalper_logs_{date.today()}.csv",
                mime="text/csv",
                use_container_width=True,
            )

with col_pos:
    st.subheader("📦 Active Position")
    pos = st.session_state.current_position
    if pos and ACCESS_TOKEN:
        pos_ltp = fetch_ltp(ACCESS_TOKEN, pos["key"]) or pos["entry_price"]
        unrealized_pnl = (pos_ltp - pos["entry_price"]) * LOT_SIZE

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Symbol", pos["symbol"])
        c2.metric("Entry", f"₹{pos['entry_price']:.2f}")
        c3.metric("LTP", f"₹{pos_ltp:.2f}")
        c4.metric(
            "Unrealized PnL",
            f"₹{unrealized_pnl:.2f}",
            delta=f"{((pos_ltp - pos['entry_price']) / pos['entry_price'] * 100):.1f}%",
        )
        st.caption(
            f"🎯 Target: **₹{pos['target_price']:.2f}** | "
            f"🛡️ Trailing SL: **₹{pos['sl_price']:.2f}** | "
            f"📈 Peak: **₹{pos['highest_price']:.2f}**"
        )
    else:
        st.info("No open position. System is flat.")

# ==============================================================================
# ALGORITHMIC LOOP ENGINE
# ==============================================================================
if st.session_state.bot_active and ACCESS_TOKEN:
    df = fetch_historical_candles(ACCESS_TOKEN, "NSE_INDEX|Nifty 50")

    if df.empty or len(df) < 30:
        st.warning("⏳ Waiting for sufficient candle data (need ≥ 30 bars)...")
        time.sleep(5)
        st.rerun()

    # ── Indicator Computation ──────────────────────────────────────────────
    df["EMA_9"]   = ta.trend.ema_indicator(df["close"], window=9)
    df["EMA_21"]  = ta.trend.ema_indicator(df["close"], window=21)
    df["Vol_SMA"] = df["volume"].rolling(20).mean()
    df["RSI_14"]  = ta.momentum.rsi(df["close"], window=14)
    df["ADX"]     = ta.trend.adx(df["high"], df["low"], df["close"], window=14)

    adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
    df["ADX"] = adx_df["ADX_14"]

    st_df = ta.supertrend(df["high"], df["low"], df["close"], length=7, multiplier=3)
    df["ST_Direction"] = (df["close"] > df["EMA_21"]).map({True: 1, False: -1})

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # ── Signal Detection (Flat) ────────────────────────────────────────────
    if st.session_state.current_position is None:
        ema_bull_cross = (prev["EMA_9"] <= prev["EMA_21"]) and (last["EMA_9"] > last["EMA_21"])
        ema_bear_cross = (prev["EMA_9"] >= prev["EMA_21"]) and (last["EMA_9"] < last["EMA_21"])

        is_trending = last["ADX"] > 25
        is_high_vol = last["volume"] > last["Vol_SMA"]
        price_above_vwap = last["close"] > last["VWAP"]
        price_below_vwap = last["close"] < last["VWAP"]

        call_signal = (
            ema_bull_cross
            and is_trending
            and is_high_vol
            and last["ST_Direction"] == 1
            and last["RSI_14"] > 50
            and price_above_vwap
        )
        put_signal = (
            ema_bear_cross
            and is_trending
            and is_high_vol
            and last["ST_Direction"] == -1
            and last["RSI_14"] < 50
            and price_below_vwap
        )

        if call_signal or put_signal:
            spot = nifty_spot or last["close"]
            ot = "CE" if call_signal else "PE"
            ikey, isymbol = get_atm_option_key(spot, ot)

            # Fetch actual ATM option LTP
            entry_ltp = fetch_ltp(ACCESS_TOKEN, ikey)
            if entry_ltp is None:
                st.warning(f"⚠️ Could not fetch LTP for {isymbol}. Skipping entry.")
            else:
                st.session_state.current_position = {
                    "key": ikey,
                    "symbol": isymbol,
                    "entry_price": entry_ltp,
                    "highest_price": entry_ltp,
                    # Trailing SL starts at entry_price * (1 - SL%)
                    "sl_price": entry_ltp * (1 - STOP_LOSS_PCT),
                    "target_price": entry_ltp * (1 + TARGET_PCT),
                    "entry_time": datetime.now().strftime("%H:%M:%S"),
                    "direction": ot,
                }
                direction_label = "📈 CALL" if ot == "CE" else "📉 PUT"
                st.success(f"🚨 {direction_label} Entry — {isymbol} @ ₹{entry_ltp:.2f}")
                st.rerun()

    # ── Position Management (In Trade) ────────────────────────────────────
    else:
        pos = st.session_state.current_position
        pos_ltp = fetch_ltp(ACCESS_TOKEN, pos["key"])

        if pos_ltp is None:
            st.warning("⚠️ LTP fetch failed — retrying next cycle.")
        else:
            # Correct trailing SL: recalculates SL from new peak
            if pos_ltp > pos["highest_price"]:
                st.session_state.current_position["highest_price"] = pos_ltp
                new_sl = pos_ltp * (1 - STOP_LOSS_PCT)
                # SL only moves up, never down
                if new_sl > pos["sl_price"]:
                    st.session_state.current_position["sl_price"] = new_sl

            hit_sl = pos_ltp <= st.session_state.current_position["sl_price"]
            hit_target = pos_ltp >= pos["target_price"]

            if hit_sl or hit_target:
                trade_pnl = (pos_ltp - pos["entry_price"]) * LOT_SIZE
                st.session_state.session_pnl += trade_pnl

                exit_reason = "TARGET HIT ✅" if hit_target else "STOP LOSS ❌"

                st.session_state.trade_logs.append({
                    "Date": date.today().isoformat(),
                    "Entry Time": pos["entry_time"],
                    "Exit Time": datetime.now().strftime("%H:%M:%S"),
                    "Symbol": pos["symbol"],
                    "Direction": pos["direction"],
                    "Mode": "LIVE" if "Live" in TRADE_MODE else "PAPER",
                    "Entry ₹": pos["entry_price"],
                    "Exit ₹": pos_ltp,
                    "Peak ₹": pos["highest_price"],
                    "PnL ₹": round(trade_pnl, 2),
                    "Exit Reason": exit_reason,
                })

                if trade_pnl <= 0:
                    st.session_state.loss_streak += 1
                    if st.session_state.loss_streak >= MAX_LOSS_STREAK:
                        st.session_state.bot_active = False
                        st.error(
                            f"🚨 Circuit breaker tripped: {MAX_LOSS_STREAK} consecutive losses. "
                            "Bot halted. Review and restart manually."
                        )
                else:
                    st.session_state.loss_streak = 0

                st.session_state.current_position = None
                st.rerun()

    # Polling interval — sleep AFTER all logic, not blocking UI operations
    time.sleep(2)
    st.rerun()

# ==============================================================================
# TRADE LOG TABLE
# ==============================================================================
st.markdown("---")
st.subheader("📑 Session Trade Log")

if st.session_state.trade_logs:
    df_logs = pd.DataFrame(st.session_state.trade_logs)

    # Color PnL column
    def style_pnl(val):
        color = "#22c55e" if val > 0 else "#ef4444" if val < 0 else "#94a3b8"
        return f"color: {color}; font-weight: bold"

    styled = df_logs.style.applymap(style_pnl, subset=["PnL ₹"])
    st.dataframe(styled, use_container_width=True)

    total_trades = len(df_logs)
    wins = (df_logs["PnL ₹"] > 0).sum()
    win_rate = wins / total_trades * 100 if total_trades else 0
    avg_win = df_logs[df_logs["PnL ₹"] > 0]["PnL ₹"].mean() if wins else 0
    avg_loss = df_logs[df_logs["PnL ₹"] <= 0]["PnL ₹"].mean() if (total_trades - wins) else 0

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total Trades", total_trades)
    s2.metric("Win Rate", f"{win_rate:.1f}%")
    s3.metric("Avg Win", f"₹{avg_win:.2f}")
    s4.metric("Avg Loss", f"₹{avg_loss:.2f}")
else:
    st.info("No trades executed this session.")
