import streamlit as st
import time
import requests
import pandas as pd
import ta
import io
import json
from datetime import date, datetime, timedelta

# ==============================================================================
# CONFIGURATION CONSTANTS
# ==============================================================================
UPSTOX_BASE_URL   = "https://api.upstox.com/v2"
NIFTY_LOT_SIZE    = 65          # Update if SEBI revises
MAX_LOSS_STREAK   = 2           # Consecutive losses before circuit breaker trips
OI_SURGE_RATIO    = 1.5         # OI must be ≥ this multiple of 5-bar OI average
INSTRUMENT_MASTER_URL = (
    "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
)

# ==============================================================================
# ── LIMITATION 1 FIX: RATE-LIMIT AWARE HTTP CLIENT ──────────────────────────
# Every outbound API call goes through api_get(). It implements:
#   • Automatic retry with exponential back-off on 429 / 5xx
#   • Per-endpoint last-call timestamp to enforce a minimum gap (rate budget)
#   • Hard cap of MAX_RETRIES attempts before giving up
# ==============================================================================
MAX_RETRIES    = 4
BASE_BACKOFF_S = 1.0   # seconds; doubles each attempt
MIN_GAP_S      = 0.25  # 250 ms floor between any two calls to the same path

if "api_last_call" not in st.session_state:
    st.session_state.api_last_call = {}   # {url_path: epoch_float}


def api_get(token: str, url: str, params: dict | None = None, timeout: int = 6) -> dict | None:
    """
    Rate-limit-aware GET wrapper.
    Returns parsed JSON dict on success, None on permanent failure.
    Retries on 429 (using Retry-After header when present) and 5xx,
    with exponential back-off. Enforces MIN_GAP_S between repeated
    calls to the same endpoint path.
    """
    path = url.split("api.upstox.com")[-1]  # key by path only

    # Enforce minimum inter-call gap for this endpoint
    last = st.session_state.api_last_call.get(path, 0)
    gap  = time.time() - last
    if gap < MIN_GAP_S:
        time.sleep(MIN_GAP_S - gap)

    headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}
    delay   = BASE_BACKOFF_S

    for attempt in range(MAX_RETRIES):
        try:
            st.session_state.api_last_call[path] = time.time()
            res = requests.get(url, headers=headers, params=params, timeout=timeout)

            if res.status_code == 200:
                return res.json()

            if res.status_code == 429:
                retry_after = float(res.headers.get("Retry-After", delay))
                time.sleep(retry_after)
                delay *= 2
                continue

            if res.status_code >= 500:
                time.sleep(delay)
                delay *= 2
                continue

            # 4xx (not 429) — client error, no point retrying
            return None

        except requests.exceptions.Timeout:
            time.sleep(delay)
            delay *= 2
        except Exception:
            return None

    return None   # exhausted retries


# ==============================================================================
# ── LIMITATION 2 FIX: INSTRUMENT MASTER — RELIABLE KEY RESOLUTION ───────────
# Rather than guessing instrument keys from date formulae, we:
#   1. Download and cache the NSE instrument master (JSON.gz) once per session.
#   2. Build an index keyed by (expiry_date, strike, option_type).
#   3. Resolve ATM key by looking up the real nearest weekly expiry
#      from the master — handles holiday expiry rollovers automatically.
#   4. Walk ±strikes from ATM until a live LTP confirms the key is tradeable.
#   5. Formula-based fallback if the master download fails.
# ==============================================================================

def load_instrument_master() -> pd.DataFrame:
    """
    Downloads and caches the Upstox NSE instrument master.
    Returns a DataFrame filtered to NIFTY weekly options only.
    Falls back to an empty DataFrame on download failure.
    """
    if "instrument_master" in st.session_state:
        return st.session_state.instrument_master

    try:
        res = requests.get(INSTRUMENT_MASTER_URL, timeout=20)
        res.raise_for_status()
        instruments = res.json()
        df = pd.DataFrame(instruments)
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

        # Filter to NIFTY F&O options only
        df = df[
            (df["name"].str.upper() == "NIFTY") &
            (df["instrument_type"].str.upper().isin(["CE", "PE"])) &
            (df["segment"].str.upper() == "NSE_FO")
        ].copy()

        df["expiry_date"] = pd.to_datetime(df["expiry"], dayfirst=True).dt.date
        df["strike"]      = pd.to_numeric(df["strike_price"], errors="coerce")
        df["option_type"] = df["instrument_type"].str.upper()

        st.session_state.instrument_master = df
        return df

    except Exception as e:
        st.warning(f"⚠️ Instrument master load failed: {e}. Using formula fallback.")
        empty = pd.DataFrame(columns=[
            "instrument_key", "trading_symbol", "expiry_date", "strike", "option_type"
        ])
        st.session_state.instrument_master = empty
        return empty


def get_weekly_expiries(master: pd.DataFrame) -> list[date]:
    """Returns sorted list of upcoming expiry dates from the master."""
    today    = date.today()
    expiries = sorted(master["expiry_date"].dropna().unique())
    return [e for e in expiries if e >= today]


def resolve_atm_option_key(
    token: str,
    spot: float,
    option_type: str,
    master: pd.DataFrame,
) -> tuple[str, str]:
    """
    Resolves the ATM Nifty option instrument key.

    1. Tries instrument master → nearest weekly expiry → walk strikes from ATM.
    2. Validates each candidate with a live LTP fetch.
    3. Falls back to the date-formula approach if the master is empty.
    """
    atm_strike = round(spot / 50) * 50
    ot         = option_type.upper()

    if not master.empty:
        expiries = get_weekly_expiries(master)
        if expiries:
            nearest = expiries[0]
            candidates = master[
                (master["expiry_date"] == nearest) &
                (master["option_type"] == ot) &
                (master["strike"].between(atm_strike - 200, atm_strike + 200))
            ].copy()
            candidates["dist"] = (candidates["strike"] - atm_strike).abs()
            candidates = candidates.sort_values("dist")

            for _, row in candidates.iterrows():
                ikey    = row["instrument_key"]
                isymbol = row.get("trading_symbol", ikey)
                ltp     = fetch_ltp(token, ikey)
                if ltp is not None and ltp > 0:
                    return ikey, isymbol

    # Formula fallback
    today            = date.today()
    days_to_thursday = (3 - today.weekday()) % 7
    if days_to_thursday == 0 and datetime.now().hour >= 15:
        days_to_thursday = 7
    expiry     = today + timedelta(days=days_to_thursday)
    expiry_str = expiry.strftime("%y%b%d").upper()
    symbol     = f"NIFTY{expiry_str}{atm_strike}{ot}"
    return f"NSE_FO|{symbol}", symbol


# ==============================================================================
# CORE API HELPERS
# ==============================================================================

def fetch_ltp(token: str, instrument_key: str) -> float | None:
    """Fetches LTP via /v2/market-quote/ltp. Routes through api_get()."""
    url  = f"{UPSTOX_BASE_URL}/market-quote/ltp"
    data = api_get(token, url, params={"instrument_key": instrument_key})
    if data is None:
        return None
    try:
        normalized = instrument_key.replace("|", ":")
        return float(data["data"][normalized]["last_price"])
    except (KeyError, TypeError, ValueError):
        return None


def fetch_ltp_batch(token: str, instrument_keys: list[str]) -> dict:
    """Batch-fetches LTPs for any number of keys (auto-chunked to 50 per call)."""
    results = {}
    for i in range(0, len(instrument_keys), 50):
        batch = instrument_keys[i : i + 50]
        url   = f"{UPSTOX_BASE_URL}/market-quote/ltp"
        data  = api_get(token, url, params={"instrument_key": ",".join(batch)})
        if data is None:
            continue
        for raw_key, val in data.get("data", {}).items():
            original = raw_key.replace(":", "|")
            try:
                results[original] = float(val["last_price"])
            except (KeyError, TypeError, ValueError):
                continue
    return results


def fetch_historical_candles(token: str, instrument_key: str) -> pd.DataFrame:
    """Fetches today's 1-minute intraday OHLCV+OI candles."""
    url  = f"{UPSTOX_BASE_URL}/historical-candle/intraday/{instrument_key}/1minute"
    data = api_get(token, url)
    if data is None:
        return pd.DataFrame()
    try:
        candles = data["data"]["candles"]
        df = pd.DataFrame(
            candles, columns=["timestamp", "open", "high", "low", "close", "volume", "oi"]
        )
        df = df.iloc[::-1].reset_index(drop=True)
        for col in ["open", "high", "low", "close", "volume", "oi"]:
            df[col] = pd.to_numeric(df[col])
        return df
    except Exception:
        return pd.DataFrame()

def fetch_vwap_from_ohlc(token: str, instrument_key: str) -> float | None:
    """
    Fetches VWAP directly from Upstox /market-quote/ohlc.
    Returns the vwap float, or None on failure.
    """
    url  = f"{UPSTOX_BASE_URL}/market-quote/ohlc"
    data = api_get(token, url, params={
        "instrument_key": instrument_key,
        "interval": "1d"   # intraday VWAP for today
    })
    if data is None:
        return None
    try:
        normalized = instrument_key.replace("|", ":")
        return float(data["data"][normalized]["ohlc"]["vwap"])
    except (KeyError, TypeError, ValueError):
        return None
        
# ==============================================================================
# ── LIMITATION 3 FIX: OPTION CHAIN OI DATA ──────────────────────────────────
# Fetches live OI at the ATM strike from the Upstox option chain endpoint.
# Computes PCR, OI change vs previous snapshot, and OI surge flag (current OI
# vs trailing 5-bar average). Both are used as confluence filters in the signal.
# ==============================================================================

def fetch_option_chain_oi(token: str, spot: float) -> dict:
    """
    Fetches option chain OI for the nearest weekly NIFTY expiry at the ATM
    strike. Returns a dict with: atm_strike, ce_oi, pe_oi, pcr,
    ce_oi_chg, pe_oi_chg, oi_surge_ce, oi_surge_pe.
    """
    atm_strike = round(spot / 50) * 50

    master   = st.session_state.get("instrument_master", pd.DataFrame())
    expiries = get_weekly_expiries(master) if not master.empty else []
    if expiries:
        expiry_str = expiries[0].strftime("%Y-%m-%d")
    else:
        today            = date.today()
        days_to_thursday = (3 - today.weekday()) % 7
        if days_to_thursday == 0 and datetime.now().hour >= 15:
            days_to_thursday = 7
        expiry_str = (today + timedelta(days=days_to_thursday)).strftime("%Y-%m-%d")

    url    = f"{UPSTOX_BASE_URL}/option/chain"
    params = {"instrument_key": "NSE_INDEX|Nifty 50", "expiry_date": expiry_str}
    data   = api_get(token, url, params=params)

    empty = {
        "atm_strike": atm_strike, "ce_oi": 0, "pe_oi": 0, "pcr": 1.0,
        "ce_oi_chg": 0, "pe_oi_chg": 0, "oi_surge_ce": False, "oi_surge_pe": False,
    }
    if data is None:
        return empty

    try:
        chain   = data["data"]
        atm_row = min(chain, key=lambda r: abs(float(r.get("strike_price", 0)) - atm_strike))

        ce_oi = float(atm_row.get("call_options", {}).get("market_data", {}).get("oi", 0))
        pe_oi = float(atm_row.get("put_options",  {}).get("market_data", {}).get("oi", 0))
        pcr   = (pe_oi / ce_oi) if ce_oi > 0 else 1.0

        prev      = st.session_state.get("oi_snapshot", {})
        ce_oi_chg = ce_oi - prev.get("ce_oi", ce_oi)
        pe_oi_chg = pe_oi - prev.get("pe_oi", pe_oi)

        history = st.session_state.get("oi_history", [])
        if len(history) >= 5:
            avg_ce = sum(h["ce_oi"] for h in history[-5:]) / 5
            avg_pe = sum(h["pe_oi"] for h in history[-5:]) / 5
        else:
            avg_ce, avg_pe = ce_oi, pe_oi

        oi_surge_ce = ce_oi >= OI_SURGE_RATIO * avg_ce if avg_ce > 0 else False
        oi_surge_pe = pe_oi >= OI_SURGE_RATIO * avg_pe if avg_pe > 0 else False

        st.session_state.oi_snapshot = {"ce_oi": ce_oi, "pe_oi": pe_oi}
        history.append({"ce_oi": ce_oi, "pe_oi": pe_oi})
        st.session_state.oi_history = history[-20:]

        return {
            "atm_strike":  float(atm_row.get("strike_price", atm_strike)),
            "ce_oi":       ce_oi,  "pe_oi":      pe_oi,
            "pcr":         pcr,    "ce_oi_chg":  ce_oi_chg,
            "pe_oi_chg":   pe_oi_chg,
            "oi_surge_ce": oi_surge_ce, "oi_surge_pe": oi_surge_pe,
        }
    except Exception:
        return empty


# ==============================================================================
# ── LIMITATION 4 FIX: ORDER ROUTING ─────────────────────────────────────────
# place_order() calls the Upstox V2 /order/place endpoint in Live mode.
# In Paper mode it skips the API call and returns a simulated order ID.
# Retries on 429 with Retry-After header. Hard-rejects on 4xx order errors.
# All placements (live and paper) are written to session_state.order_log
# for a full intra-session audit trail.
# ==============================================================================

def place_order(
    token:            str,
    instrument_key:   str,
    transaction_type: str,    # "BUY" | "SELL"
    quantity:         int,
    order_type:       str   = "MARKET",
    price:            float = 0.0,
    paper_mode:       bool  = True,
) -> str | None:
    """
    Places a Upstox V2 market/limit order or simulates one in Paper mode.
    Returns order_id on success, None on failure.
    """
    sim_id = f"PAPER-{datetime.now().strftime('%H%M%S%f')}"

    if paper_mode:
        st.session_state.setdefault("order_log", []).append({
            "time": datetime.now().isoformat(), "mode": "PAPER",
            "instrument_key": instrument_key, "type": transaction_type,
            "qty": quantity, "order_type": order_type, "price": price,
            "order_id": sim_id, "status": "SIMULATED",
        })
        return sim_id

    # ── Live placement ───────────────────────────────────────────────────
    url     = f"{UPSTOX_BASE_URL}/order/place"
    headers = {
        "Accept":        "application/json",
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {token}",
    }
    payload = {
        "quantity":           quantity,
        "product":            "I",        # Intraday (MIS)
        "validity":           "DAY",
        "price":              price if order_type == "LIMIT" else 0,
        "tag":                "SCALPER",
        "instrument_token":   instrument_key,
        "order_type":         order_type,
        "transaction_type":   transaction_type,
        "disclosed_quantity": 0,
        "trigger_price":      0,
        "is_amo":             False,
    }

    delay = BASE_BACKOFF_S
    for attempt in range(MAX_RETRIES):
        try:
            res = requests.post(url, headers=headers,
                                data=json.dumps(payload), timeout=8)
            if res.status_code == 200:
                order_id = res.json()["data"]["order_id"]
                st.session_state.setdefault("order_log", []).append({
                    "time": datetime.now().isoformat(), "mode": "LIVE",
                    "instrument_key": instrument_key, "type": transaction_type,
                    "qty": quantity, "order_type": order_type, "price": price,
                    "order_id": order_id, "status": "PLACED",
                })
                return order_id

            if res.status_code == 429:
                wait = float(res.headers.get("Retry-After", delay))
                time.sleep(wait)
                delay *= 2
                continue

            # Non-retriable client error
            err = res.json().get("errors", [{}])[0].get("message", res.text)
            st.error(f"🚨 Order rejected ({res.status_code}): {err}")
            return None

        except Exception:
            time.sleep(delay)
            delay *= 2

    st.error("🚨 Order placement failed after max retries.")
    return None


def exit_position(
    token:       str,
    pos:         dict,
    exit_price:  float,
    exit_reason: str,
    paper_mode:  bool,
    lot_size:    int,
) -> None:
    """
    Unified exit: places SELL order → updates PnL → appends trade log
    → manages loss streak / circuit breaker.
    """
    order_id  = place_order(token, pos["key"], "SELL", lot_size,
                            order_type="MARKET", paper_mode=paper_mode)
    trade_pnl = (exit_price - pos["entry_price"]) * lot_size
    st.session_state.session_pnl += trade_pnl

    st.session_state.trade_logs.append({
        "Date":          date.today().isoformat(),
        "Entry Time":    pos["entry_time"],
        "Exit Time":     datetime.now().strftime("%H:%M:%S"),
        "Symbol":        pos["symbol"],
        "Direction":     pos["direction"],
        "Mode":          "LIVE" if not paper_mode else "PAPER",
        "Entry ₹":       pos["entry_price"],
        "Exit ₹":        exit_price,
        "Peak ₹":        pos["highest_price"],
        "PnL ₹":         round(trade_pnl, 2),
        "Exit Reason":   exit_reason,
        "Entry Order":   pos.get("entry_order_id", "—"),
        "Exit Order":    order_id or "—",
    })

    if trade_pnl <= 0:
        st.session_state.loss_streak += 1
        if st.session_state.loss_streak >= MAX_LOSS_STREAK:
            st.session_state.bot_active = False
            st.error(
                f"🚨 Circuit breaker: {MAX_LOSS_STREAK} consecutive losses. "
                "Bot halted. Review and restart manually."
            )
    else:
        st.session_state.loss_streak = 0

    st.session_state.current_position = None


# ==============================================================================
# STREAMLIT SESSION STATE BOOTSTRAP
# ==============================================================================
st.set_page_config(page_title="Upstox Algo Scalper", layout="wide")

defaults = {
    "bot_active":       False,
    "current_position": None,
    "trade_logs":       [],
    "session_pnl":      0.0,
    "loss_streak":      0,
    "oi_snapshot":      {},
    "oi_history":       [],
    "order_log":        [],
    "api_last_call":    {},
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ==============================================================================
# SIDEBAR
# ==============================================================================
st.sidebar.header("🔌 Authentication")
ACCESS_TOKEN = st.sidebar.text_input(
    "Upstox Access Token", type="password",
    help="Expires midnight IST — generate fresh each trading day.",
)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Strategy Parameters")
INITIAL_CAPITAL = st.sidebar.number_input("Starting Capital (₹)", value=50_000, step=5_000)
STOP_LOSS_PCT   = st.sidebar.slider("Stop Loss (%)",    1, 10, 6)  / 100.0
TARGET_PCT      = st.sidebar.slider("Target Profit (%)", 1, 25, 12) / 100.0
LOT_SIZE        = st.sidebar.number_input("Lot Size", value=NIFTY_LOT_SIZE, min_value=1, step=1)
TRADE_MODE      = st.sidebar.radio(
    "🎯 Execution Mode",
    ["📄 Paper Trading (Simulated)", "⚡ Live Trading (Real Money)"],
)
IS_PAPER = "Paper" in TRADE_MODE

st.sidebar.markdown("---")
st.sidebar.caption(f"⚡ Circuit breaker after **{MAX_LOSS_STREAK}** consecutive losses")
st.sidebar.caption(f"📊 OI surge threshold: **{OI_SURGE_RATIO}×** 5-bar average")

current_capital = INITIAL_CAPITAL + st.session_state.session_pnl

# Pre-load instrument master once per session
if ACCESS_TOKEN and "instrument_master" not in st.session_state:
    with st.sidebar:
        with st.spinner("Loading instrument master…"):
            load_instrument_master()

# ==============================================================================
# MAIN DASHBOARD
# ==============================================================================
st.title("🦅 Upstox Options Advanced Scalping Engine")

# ── Row 1: KPI Metrics ────────────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)
nifty_spot = fetch_ltp(ACCESS_TOKEN, "NSE_INDEX|Nifty 50") if ACCESS_TOKEN else None

m1.metric("📊 Nifty 50 Spot",   f"₹{nifty_spot:,.2f}" if nifty_spot else "—")
m2.metric("💰 Account Balance", f"₹{current_capital:,.2f}")
m3.metric(
    "📈 Session PnL",
    f"₹{st.session_state.session_pnl:,.2f}",
    delta=f"{(st.session_state.session_pnl / INITIAL_CAPITAL * 100):.2f}%" if INITIAL_CAPITAL else "0%",
)
m4.metric("🛡️ Mode",        "LIVE" if not IS_PAPER else "PAPER")
m5.metric("🔴 Loss Streak",  f"{st.session_state.loss_streak} / {MAX_LOSS_STREAK}",
          delta="⚠️ At limit!" if st.session_state.loss_streak >= MAX_LOSS_STREAK - 1 else None)

st.markdown("---")

# ── Row 2: Engine Controls | OI Panel | Active Position ──────────────────────
col_ctrl, col_oi, col_pos = st.columns([1, 1, 2])

with col_ctrl:
    st.subheader("🕹️ Engine Controls")
    if not ACCESS_TOKEN:
        st.warning("Provide an Upstox Access Token to activate.")
    else:
        if not st.session_state.bot_active:
            if st.button("▶️ START BOT", type="primary", use_container_width=True):
                st.session_state.bot_active  = True
                st.session_state.loss_streak = 0
                st.rerun()
        else:
            if st.button("🛑 EMERGENCY HALT", type="secondary", use_container_width=True):
                # Attempt live market-exit before halting
                if st.session_state.current_position and not IS_PAPER:
                    pos     = st.session_state.current_position
                    ltp_now = fetch_ltp(ACCESS_TOKEN, pos["key"]) or pos["entry_price"]
                    exit_position(ACCESS_TOKEN, pos, ltp_now, "MANUAL HALT 🛑", IS_PAPER, LOT_SIZE)
                st.session_state.bot_active       = False
                st.session_state.current_position = None
                st.rerun()

        status = "🟢 Active & Scanning" if st.session_state.bot_active else "🔴 Inactive"
        st.info(f"State: **{status}**")

        if st.session_state.trade_logs:
            csv_bytes = pd.DataFrame(st.session_state.trade_logs).to_csv(index=False).encode()
            st.download_button("📥 Trade Log (CSV)", data=csv_bytes,
                               file_name=f"scalper_{date.today()}.csv",
                               mime="text/csv", use_container_width=True)

with col_oi:
    st.subheader("📊 OI Intelligence")
    snap = st.session_state.get("oi_snapshot", {})
    if snap:
        ce_oi = snap.get("ce_oi", 0)
        pe_oi = snap.get("pe_oi", 0)
        pcr   = pe_oi / ce_oi if ce_oi > 0 else 1.0
        o1, o2 = st.columns(2)
        o1.metric("CE OI", f"{ce_oi/1e5:.2f}L")
        o2.metric("PE OI", f"{pe_oi/1e5:.2f}L")
        pcr_icon = "🟢" if pcr > 1.2 else "🔴" if pcr < 0.8 else "🟡"
        st.metric("PCR", f"{pcr_icon} {pcr:.2f}",
                  help="PCR > 1.2 = bullish, < 0.8 = bearish")
        hist = st.session_state.get("oi_history", [])
        if len(hist) > 1:
            st.line_chart(pd.DataFrame(hist).tail(15)[["ce_oi", "pe_oi"]],
                          height=120, use_container_width=True)
    else:
        st.info("OI data loads when the bot is active.")

with col_pos:
    st.subheader("📦 Active Position")
    pos = st.session_state.current_position
    if pos and ACCESS_TOKEN:
        pos_ltp = fetch_ltp(ACCESS_TOKEN, pos["key"]) or pos["entry_price"]
        unr_pnl = (pos_ltp - pos["entry_price"]) * LOT_SIZE
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Symbol", pos["symbol"])
        c2.metric("Entry",  f"₹{pos['entry_price']:.2f}")
        c3.metric("LTP",    f"₹{pos_ltp:.2f}")
        c4.metric("Unrealized PnL", f"₹{unr_pnl:.2f}",
                  delta=f"{((pos_ltp-pos['entry_price'])/pos['entry_price']*100):.1f}%")
        st.caption(
            f"🎯 Target: **₹{pos['target_price']:.2f}** | "
            f"🛡️ Trailing SL: **₹{pos['sl_price']:.2f}** | "
            f"📈 Peak: **₹{pos['highest_price']:.2f}** | "
            f"🆔 Order: `{pos.get('entry_order_id','—')}`"
        )
    else:
        st.info("No open position. System is flat.")

# ==============================================================================
# ALGORITHMIC LOOP ENGINE
# ==============================================================================
if st.session_state.bot_active and ACCESS_TOKEN:

    df = fetch_historical_candles(ACCESS_TOKEN, "NSE_INDEX|Nifty 50")
    if df.empty or len(df) < 30:
        st.warning("⏳ Waiting for ≥ 30 candles…")
        time.sleep(5)
        st.rerun()

    # ── Indicators ────────────────────────────────────────────────────────────
    df["EMA_9"]        = ta.trend.ema_indicator(df["close"], window=9)
    df["EMA_21"]       = ta.trend.ema_indicator(df["close"], window=21)
    df["Vol_SMA"]      = df["volume"].rolling(window=20).mean()
    df["RSI_14"]       = ta.momentum.rsi(df["close"], window=14)
    df["ADX"]          = ta.trend.adx(df["high"], df["low"], df["close"], window=14)

    def compute_supertrend(df: pd.DataFrame, length: int = 7, multiplier: float = 3.0) -> pd.Series:
      hl_avg = (df["high"] + df["low"]) / 2
      atr    = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=length)
      upper  = hl_avg + multiplier * atr
      lower  = hl_avg - multiplier * atr
    
      direction = pd.Series(1, index=df.index)
      for i in range(1, len(df)):
        if df["close"].iloc[i] > upper.iloc[i - 1]:
            direction.iloc[i] = 1
        elif df["close"].iloc[i] < lower.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]
      return direction

    df["ST_Direction"] = compute_supertrend(df)
    
    last = df.iloc[-1]
    prev = df.iloc[-2]


    live_vwap = fetch_vwap_from_ohlc(ACCESS_TOKEN, "NSE_INDEX|Nifty 50")
    df["VWAP"] = live_vwap if live_vwap else df["close"]  # fallback to close  

    # ── OI snapshot (every cycle) ─────────────────────────────────────────────
    spot_ref = nifty_spot or float(last["close"])
    oi       = fetch_option_chain_oi(ACCESS_TOKEN, spot_ref)
    master   = load_instrument_master()

    # ── Signal Detection ──────────────────────────────────────────────────────
    if st.session_state.current_position is None:
        ema_bull = (prev["EMA_9"] <= prev["EMA_21"]) and (last["EMA_9"] > last["EMA_21"])
        ema_bear = (prev["EMA_9"] >= prev["EMA_21"]) and (last["EMA_9"] < last["EMA_21"])

        is_trending        = last["ADX"]    > 25
        is_high_vol        = last["volume"] > last["Vol_SMA"]
        above_vwap         = last["close"]  > last["VWAP"]
        below_vwap         = last["close"]  < last["VWAP"]
        oi_confirms_call   = oi["oi_surge_ce"] or (oi["pcr"] > 1.0)
        oi_confirms_put    = oi["oi_surge_pe"] or (oi["pcr"] < 1.0)

        call_signal = (
            ema_bull and is_trending and is_high_vol
            and last["ST_Direction"] == 1
            and last["RSI_14"] > 50
            and above_vwap
            and oi_confirms_call
        )
        put_signal = (
            ema_bear and is_trending and is_high_vol
            and last["ST_Direction"] == -1
            and last["RSI_14"] < 50
            and below_vwap
            and oi_confirms_put
        )

        if call_signal or put_signal:
            ot            = "CE" if call_signal else "PE"
            ikey, isymbol = resolve_atm_option_key(ACCESS_TOKEN, spot_ref, ot, master)
            entry_ltp     = fetch_ltp(ACCESS_TOKEN, ikey)

            if entry_ltp is None or entry_ltp <= 0:
                st.warning(f"⚠️ No valid LTP for {isymbol} — skipping entry.")
            else:
                order_id = place_order(ACCESS_TOKEN, ikey, "BUY", LOT_SIZE,
                                       order_type="MARKET", paper_mode=IS_PAPER)
                if order_id is None and not IS_PAPER:
                    st.error("🚨 BUY order failed — entry aborted.")
                else:
                    st.session_state.current_position = {
                        "key":            ikey,
                        "symbol":         isymbol,
                        "entry_price":    entry_ltp,
                        "highest_price":  entry_ltp,
                        "sl_price":       entry_ltp * (1 - STOP_LOSS_PCT),
                        "target_price":   entry_ltp * (1 + TARGET_PCT),
                        "entry_time":     datetime.now().strftime("%H:%M:%S"),
                        "direction":      ot,
                        "entry_order_id": order_id,
                    }
                    lbl = "📈 CALL" if ot == "CE" else "📉 PUT"
                    st.success(f"🚨 {lbl} — {isymbol} @ ₹{entry_ltp:.2f} | Order: {order_id}")
                    st.rerun()

    # ── Position Management ───────────────────────────────────────────────────
    else:
        pos     = st.session_state.current_position
        pos_ltp = fetch_ltp(ACCESS_TOKEN, pos["key"])

        if pos_ltp is None:
            st.warning("⚠️ LTP fetch failed — retrying next cycle.")
        else:
            # Trailing SL ratchet (only moves up)
            if pos_ltp > pos["highest_price"]:
                st.session_state.current_position["highest_price"] = pos_ltp
                new_sl = pos_ltp * (1 - STOP_LOSS_PCT)
                if new_sl > pos["sl_price"]:
                    st.session_state.current_position["sl_price"] = new_sl

            hit_sl     = pos_ltp <= st.session_state.current_position["sl_price"]
            hit_target = pos_ltp >= pos["target_price"]

            if hit_sl or hit_target:
                reason = "TARGET HIT ✅" if hit_target else "STOP LOSS ❌"
                exit_position(ACCESS_TOKEN, pos, pos_ltp, reason, IS_PAPER, LOT_SIZE)
                st.rerun()

    time.sleep(2)
    st.rerun()

# ==============================================================================
# TRADE LOG + SESSION STATS
# ==============================================================================
st.markdown("---")
st.subheader("📑 Session Trade Log")

if st.session_state.trade_logs:
    df_logs = pd.DataFrame(st.session_state.trade_logs)

    def style_pnl(val):
        color = "#22c55e" if val > 0 else "#ef4444" if val < 0 else "#94a3b8"
        return f"color: {color}; font-weight: bold"

    st.dataframe(df_logs.style.applymap(style_pnl, subset=["PnL ₹"]),
                 use_container_width=True)

    total = len(df_logs)
    wins  = (df_logs["PnL ₹"] > 0).sum()
    wr    = wins / total * 100 if total else 0
    avg_w = df_logs[df_logs["PnL ₹"] > 0]["PnL ₹"].mean() if wins else 0
    avg_l = df_logs[df_logs["PnL ₹"] <= 0]["PnL ₹"].mean() if (total - wins) else 0

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total Trades", total)
    s2.metric("Win Rate",     f"{wr:.1f}%")
    s3.metric("Avg Win",      f"₹{avg_w:.2f}")
    s4.metric("Avg Loss",     f"₹{avg_l:.2f}")
else:
    st.info("No trades executed this session.")

# ── Raw Order Log (audit trail) ───────────────────────────────────────────────
if st.session_state.order_log:
    with st.expander("🗂️ Raw Order Log"):
        st.dataframe(pd.DataFrame(st.session_state.order_log), use_container_width=True)
