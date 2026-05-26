import streamlit as st
import time
import requests
import pandas as pd
import ta
import json
from datetime import date, datetime, timedelta, timezone

# ==============================================================================
# CONFIGURATION CONSTANTS
# ==============================================================================
UPSTOX_BASE_URL       = "https://api.upstox.com/v2"
NIFTY_LOT_SIZE        = 65
MAX_LOSS_STREAK       = 2
OI_SURGE_RATIO        = 1.5
SLIPPAGE_PER_SIDE     = 1.5          # ₹ per unit per side
IST                   = timezone(timedelta(hours=5, minutes=30))
INSTRUMENT_MASTER_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"

# ── Risk guardrail defaults ───────────────────────────────────────────────────
DEFAULT_MAX_DAILY_LOSS = 5000
DEFAULT_MAX_TRADES     = 5
DEFAULT_STOP_LOSS_PCT  = 6
DEFAULT_TARGET_PCT     = 12
DEFAULT_RR_MIN         = 1.5

# ── Prime trading windows in IST ──────────────────────────────────────────────
PRIME_WINDOWS = [
    ("09:30", "11:30"),
    ("13:30", "14:45"),
]

# ── Signal state machine: bars confirmation must hold before entry ────────────
SIGNAL_HOLD_BARS = 2

# ==============================================================================
# RATE-LIMIT AWARE HTTP CLIENT
# ==============================================================================
MAX_RETRIES    = 4
BASE_BACKOFF_S = 1.0
MIN_GAP_S      = 0.25

if "api_last_call" not in st.session_state:
    st.session_state.api_last_call = {}


# ==============================================================================
# ── KEY FIX: RAW URL BUILDER ─────────────────────────────────────────────────
# requests.get(params=...) URL-encodes query strings, turning
# "NSE_INDEX|Nifty 50" → "NSE_INDEX%7CNifty%2050".
# Upstox validates the raw string server-side and rejects the encoded form
# with error UDAPI100011 "Invalid Instrument".
# _build_url() builds the query string manually so | and spaces are sent raw.
# ==============================================================================

def _build_url(base: str, params: dict) -> str:
    """Builds URL with raw (un-encoded) query params — required by Upstox."""
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    return f"{base}?{qs}"


def _auth_headers(token: str) -> dict:
    return {"Accept": "application/json", "Authorization": f"Bearer {token}"}


def _raw_get(token: str, url: str, timeout: int = 8) -> dict | None:
    """
    GET using a pre-built raw URL (no params= encoding).
    Handles 429 / 5xx with exponential backoff, logs 4xx errors.
    """
    path  = url.split("?")[0].split("api.upstox.com")[-1]
    gap   = time.time() - st.session_state.api_last_call.get(path, 0)
    if gap < MIN_GAP_S:
        time.sleep(MIN_GAP_S - gap)

    delay = BASE_BACKOFF_S
    for attempt in range(MAX_RETRIES):
        try:
            st.session_state.api_last_call[path] = time.time()
            res = requests.get(url, headers=_auth_headers(token), timeout=timeout)
            if res.status_code == 200:
                return res.json()
            if res.status_code == 429:
                time.sleep(float(res.headers.get("Retry-After", delay)))
                delay *= 2
                continue
            if res.status_code >= 500:
                time.sleep(delay)
                delay *= 2
                continue
            # 4xx — log and return None
            st.session_state.setdefault("api_errors", []).append({
                "time": now_ist().strftime("%H:%M:%S"),
                "endpoint": path,
                "status": res.status_code,
                "body": res.text[:300],
            })
            return None
        except requests.exceptions.Timeout:
            time.sleep(delay)
            delay *= 2
        except Exception as e:
            st.session_state.setdefault("api_errors", []).append({
                "time": now_ist().strftime("%H:%M:%S"),
                "endpoint": path,
                "status": "exception",
                "body": str(e),
            })
            return None
    return None


def api_get(token: str, url: str, params: dict | None = None, timeout: int = 8) -> dict | None:
    """
    Wrapper for endpoints that do NOT contain special chars in params.
    For instrument_key params use _raw_get(_build_url(...)) instead.
    """
    if params:
        built = _build_url(url, params)
        return _raw_get(token, built, timeout)
    return _raw_get(token, url, timeout)


# ==============================================================================
# IST TIME HELPERS
# ── Streamlit Cloud runs UTC. All time comparisons use explicit IST offset.
# ==============================================================================

def now_ist() -> datetime:
    return datetime.now(tz=IST)


def in_prime_session() -> bool:
    now = now_ist().time()
    for start_str, end_str in PRIME_WINDOWS:
        start = datetime.strptime(start_str, "%H:%M").time()
        end   = datetime.strptime(end_str,   "%H:%M").time()
        if start <= now <= end:
            return True
    return False


def next_window_str() -> str:
    now = now_ist().time()
    for start_str, end_str in PRIME_WINDOWS:
        start = datetime.strptime(start_str, "%H:%M").time()
        if now < start:
            return f"{start_str} – {end_str} IST"
    return "09:30 IST tomorrow"


def ist_time_str() -> str:
    return now_ist().strftime("%H:%M:%S IST")


# ==============================================================================
# INSTRUMENT MASTER
# ==============================================================================

def load_instrument_master() -> pd.DataFrame:
    if "instrument_master" in st.session_state:
        return st.session_state.instrument_master
    try:
        res = requests.get(INSTRUMENT_MASTER_URL, timeout=20)
        res.raise_for_status()
        df  = pd.DataFrame(res.json())
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
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
        empty = pd.DataFrame(columns=["instrument_key", "trading_symbol",
                                       "expiry_date", "strike", "option_type"])
        st.session_state.instrument_master = empty
        return empty


def get_weekly_expiries(master: pd.DataFrame) -> list[date]:
    today    = date.today()
    expiries = sorted(master["expiry_date"].dropna().unique())
    return [e for e in expiries if e >= today]


def resolve_atm_option_key(token: str, spot: float, option_type: str,
                            master: pd.DataFrame) -> tuple[str, str]:
    atm_strike = round(spot / 50) * 50
    ot         = option_type.upper()
    if not master.empty:
        expiries = get_weekly_expiries(master)
        if expiries:
            candidates = master[
                (master["expiry_date"] == expiries[0]) &
                (master["option_type"] == ot) &
                (master["strike"].between(atm_strike - 200, atm_strike + 200))
            ].copy()
            candidates["dist"] = (candidates["strike"] - atm_strike).abs()
            for _, row in candidates.sort_values("dist").iterrows():
                ikey = row["instrument_key"]
                ltp  = fetch_ltp(token, ikey)
                if ltp is not None and ltp > 0:
                    return ikey, row.get("trading_symbol", ikey)
    # Formula fallback
    today            = date.today()
    days_to_thursday = (3 - today.weekday()) % 7
    if days_to_thursday == 0 and now_ist().hour >= 15:
        days_to_thursday = 7
    expiry = today + timedelta(days=days_to_thursday)
    symbol = f"NIFTY{expiry.strftime('%y%b%d').upper()}{atm_strike}{ot}"
    return f"NSE_FO|{symbol}", symbol


# ==============================================================================
# CORE API HELPERS
# All instrument_key params are passed via _build_url to avoid encoding.
# ==============================================================================

def fetch_ltp(token: str, instrument_key: str) -> float | None:
    url  = _build_url(f"{UPSTOX_BASE_URL}/market-quote/ltp",
                      {"instrument_key": instrument_key})
    data = _raw_get(token, url)
    if data is None:
        return None
    try:
        normalized = instrument_key.replace("|", ":")
        return float(data["data"][normalized]["last_price"])
    except (KeyError, TypeError, ValueError):
        return None


def fetch_ltp_batch(token: str, instrument_keys: list[str]) -> dict:
    results = {}
    for i in range(0, len(instrument_keys), 50):
        batch = instrument_keys[i : i + 50]
        url   = _build_url(f"{UPSTOX_BASE_URL}/market-quote/ltp",
                           {"instrument_key": ",".join(batch)})
        data  = _raw_get(token, url)
        if data is None:
            continue
        for raw_key, val in data.get("data", {}).items():
            try:
                results[raw_key.replace(":", "|")] = float(val["last_price"])
            except (KeyError, TypeError, ValueError):
                continue
    return results


def fetch_historical_candles(token: str, instrument_key: str,
                              interval: str = "1minute") -> pd.DataFrame:
    # instrument_key embedded in path — encode spaces as %20 for path safety
    safe_key = instrument_key.replace(" ", "%20")
    url      = f"{UPSTOX_BASE_URL}/historical-candle/intraday/{safe_key}/{interval}"
    data     = _raw_get(token, url)
    if data is None:
        return pd.DataFrame()
    try:
        candles = data["data"]["candles"]
        df = pd.DataFrame(candles,
                          columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
        df = df.iloc[::-1].reset_index(drop=True)
        for col in ["open", "high", "low", "close", "volume", "oi"]:
            df[col] = pd.to_numeric(df[col])
        return df
    except Exception:
        return pd.DataFrame()


def fetch_vwap_from_ohlc(token: str, instrument_key: str) -> float | None:
    """Fetches session VWAP from /market-quote/ohlc using raw URL."""
    if not token:
        return None
    url  = _build_url(f"{UPSTOX_BASE_URL}/market-quote/ohlc",
                      {"instrument_key": instrument_key, "interval": "1d"})
    data = _raw_get(token, url)
    if not data:
        return None
    try:
        normalized = instrument_key.replace("|", ":")
        vwap = data["data"][normalized]["ohlc"].get("vwap")
        return float(vwap) if vwap else None
    except (KeyError, TypeError, ValueError):
        return None


# ==============================================================================
# OPTION CHAIN OI
# ── Uses _build_url so "NSE_INDEX|Nifty 50" is sent raw (not URL-encoded).
# ── oi_available flag: when False, OI gate is bypassed in signal logic
#    so a temporary OI outage does not silently block all entries.
# ==============================================================================

def fetch_option_chain_oi(token: str, spot: float) -> dict:
    atm_strike = round(spot / 50) * 50
    master     = st.session_state.get("instrument_master", pd.DataFrame())
    expiries   = get_weekly_expiries(master) if not master.empty else []
    if expiries:
        expiry_str = expiries[0].strftime("%Y-%m-%d")
    else:
        today            = date.today()
        days_to_thursday = (3 - today.weekday()) % 7
        if days_to_thursday == 0 and now_ist().hour >= 15:
            days_to_thursday = 7
        expiry_str = (today + timedelta(days=days_to_thursday)).strftime("%Y-%m-%d")

    # ── Raw URL — pipe and space sent exactly as Upstox expects ──────────────
    url  = _build_url(f"{UPSTOX_BASE_URL}/option/chain",
                      {"instrument_key": "NSE_INDEX|Nifty 50",
                       "expiry_date":    expiry_str})
    data = _raw_get(token, url)

    empty = {
        "atm_strike": atm_strike, "ce_oi": 0, "pe_oi": 0, "pcr": 1.0,
        "ce_oi_chg": 0, "pe_oi_chg": 0,
        "oi_surge_ce": False, "oi_surge_pe": False,
        "oi_available": False,
    }
    if data is None:
        return empty

    try:
        chain   = data["data"]
        atm_row = min(chain, key=lambda r: abs(float(r.get("strike_price", 0)) - atm_strike))
        ce_oi   = float(atm_row.get("call_options", {}).get("market_data", {}).get("oi", 0))
        pe_oi   = float(atm_row.get("put_options",  {}).get("market_data", {}).get("oi", 0))
        pcr     = (pe_oi / ce_oi) if ce_oi > 0 else 1.0
        prev    = st.session_state.get("oi_snapshot", {})
        history = st.session_state.get("oi_history", [])
        avg_ce, avg_pe = (
            (sum(h["ce_oi"] for h in history[-5:]) / 5,
             sum(h["pe_oi"] for h in history[-5:]) / 5)
            if len(history) >= 5 else (ce_oi, pe_oi)
        )
        st.session_state.oi_snapshot = {"ce_oi": ce_oi, "pe_oi": pe_oi}
        history.append({"ce_oi": ce_oi, "pe_oi": pe_oi})
        st.session_state.oi_history = history[-20:]
        return {
            "atm_strike":  float(atm_row.get("strike_price", atm_strike)),
            "ce_oi":       ce_oi,
            "pe_oi":       pe_oi,
            "pcr":         pcr,
            "ce_oi_chg":   ce_oi - prev.get("ce_oi", ce_oi),
            "pe_oi_chg":   pe_oi - prev.get("pe_oi", pe_oi),
            "oi_surge_ce": ce_oi >= OI_SURGE_RATIO * avg_ce if avg_ce > 0 else False,
            "oi_surge_pe": pe_oi >= OI_SURGE_RATIO * avg_pe if avg_pe > 0 else False,
            "oi_available": True,
        }
    except Exception:
        return empty


# ==============================================================================
# ORDER ROUTING
# ==============================================================================

def place_order(token: str, instrument_key: str, transaction_type: str,
                quantity: int, order_type: str = "MARKET",
                price: float = 0.0, paper_mode: bool = True) -> str | None:
    sim_id = f"PAPER-{now_ist().strftime('%H%M%S%f')}"
    if paper_mode:
        st.session_state.setdefault("order_log", []).append({
            "time": now_ist().isoformat(), "mode": "PAPER",
            "instrument_key": instrument_key, "type": transaction_type,
            "qty": quantity, "order_type": order_type, "price": price,
            "order_id": sim_id, "status": "SIMULATED",
        })
        return sim_id

    url     = f"{UPSTOX_BASE_URL}/order/place"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    payload = {
        "quantity":           quantity,
        "product":            "I",
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
            res = requests.post(url, headers=headers, data=json.dumps(payload), timeout=8)
            if res.status_code == 200:
                order_id = res.json()["data"]["order_id"]
                st.session_state.setdefault("order_log", []).append({
                    "time": now_ist().isoformat(), "mode": "LIVE",
                    "instrument_key": instrument_key, "type": transaction_type,
                    "qty": quantity, "order_type": order_type, "price": price,
                    "order_id": order_id, "status": "PLACED",
                })
                return order_id
            if res.status_code == 429:
                time.sleep(float(res.headers.get("Retry-After", delay)))
                delay *= 2
                continue
            err = res.json().get("errors", [{}])[0].get("message", res.text)
            st.error(f"🚨 Order rejected ({res.status_code}): {err}")
            return None
        except Exception:
            time.sleep(delay)
            delay *= 2
    st.error("🚨 Order placement failed after max retries.")
    return None


def exit_position(token: str, pos: dict, exit_price: float, exit_reason: str,
                  paper_mode: bool, lot_size: int) -> None:
    order_id      = place_order(token, pos["key"], "SELL", lot_size,
                                order_type="MARKET", paper_mode=paper_mode)
    gross_pnl     = (exit_price - pos["entry_price"]) * lot_size
    slippage      = SLIPPAGE_PER_SIDE * 2 * lot_size
    net_pnl       = gross_pnl - slippage
    risk_per_unit = abs(pos["entry_price"] - pos["sl_price"])
    rr_realised   = round(gross_pnl / (risk_per_unit * lot_size), 2) if risk_per_unit > 0 else 0.0

    st.session_state.session_pnl += net_pnl
    st.session_state.trade_logs.append({
        "Date":        date.today().isoformat(),
        "Entry Time":  pos["entry_time"],
        "Exit Time":   now_ist().strftime("%H:%M:%S"),
        "Symbol":      pos["symbol"],
        "Direction":   pos["direction"],
        "Mode":        "LIVE" if not paper_mode else "PAPER",
        "Entry ₹":     pos["entry_price"],
        "Exit ₹":      exit_price,
        "Peak ₹":      pos["highest_price"],
        "SL ₹":        pos["sl_price"],
        "Target ₹":    pos["target_price"],
        "Gross PnL ₹": round(gross_pnl, 2),
        "Slippage ₹":  round(slippage, 2),
        "Net PnL ₹":   round(net_pnl, 2),
        "RR Realised": rr_realised,
        "Exit Reason": exit_reason,
        "Entry Order": pos.get("entry_order_id", "—"),
        "Exit Order":  order_id or "—",
    })

    if net_pnl <= 0:
        st.session_state.loss_streak += 1
        if st.session_state.loss_streak >= MAX_LOSS_STREAK:
            st.session_state.bot_active = False
            st.error(f"🚨 Circuit breaker: {MAX_LOSS_STREAK} consecutive losses. Bot halted.")
    else:
        st.session_state.loss_streak = 0
    st.session_state.current_position = None


# ==============================================================================
# GUARDRAIL + INDICATOR HELPERS
# ==============================================================================

def check_daily_guardrails(session_pnl: float, trade_count: int,
                            max_daily_loss: float, max_trades: int) -> tuple[bool, str]:
    if session_pnl <= -abs(max_daily_loss):
        return False, f"🚨 Daily loss limit ₹{abs(max_daily_loss):,.0f} hit. Bot halted."
    if trade_count >= max_trades:
        return False, f"📊 Max {max_trades} trades/day reached. Bot paused."
    if not in_prime_session():
        return False, f"🕐 Outside prime window. Next: {next_window_str()}"
    return True, ""


def get_htf_trend(token: str) -> int:
    """Returns +1 (bull), -1 (bear), 0 (neutral) based on 5-min EMA 9/21."""
    df5 = fetch_historical_candles(token, "NSE_INDEX|Nifty 50", interval="5minute")
    if df5.empty or len(df5) < 22:
        return 0
    df5["EMA_9"]  = ta.trend.ema_indicator(df5["close"], window=9)
    df5["EMA_21"] = ta.trend.ema_indicator(df5["close"], window=21)
    last5 = df5.iloc[-1]
    if last5["EMA_9"] > last5["EMA_21"]:
        return 1
    if last5["EMA_9"] < last5["EMA_21"]:
        return -1
    return 0


def compute_atr_sl_target(df: pd.DataFrame, entry_ltp: float,
                           atr_multiplier: float, rr_min: float,
                           delta: float = 0.5) -> tuple[float, float] | None:
    atr     = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
    atr_val = float(atr.dropna().iloc[-1]) if not atr.dropna().empty else 0
    if atr_val <= 0:
        return None
    sl_distance  = atr_val * atr_multiplier * delta
    tgt_distance = sl_distance * rr_min
    sl_price     = entry_ltp - sl_distance
    target_price = entry_ltp + tgt_distance
    if sl_price <= 0 or sl_price >= entry_ltp:
        return None
    return round(sl_price, 2), round(target_price, 2)


def compute_supertrend(df: pd.DataFrame, length: int = 7,
                        multiplier: float = 3.0) -> pd.Series:
    hl_avg    = (df["high"] + df["low"]) / 2
    atr       = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=length)
    upper     = hl_avg + multiplier * atr
    lower     = hl_avg - multiplier * atr
    direction = pd.Series(1, index=df.index)
    for i in range(1, len(df)):
        if df["close"].iloc[i] > upper.iloc[i - 1]:
            direction.iloc[i] = 1
        elif df["close"].iloc[i] < lower.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]
    return direction


# ==============================================================================
# SIGNAL STATE MACHINE
# Two-stage pipeline:
#   Stage 1 — SETUP: EMA crossover + ADX trend + Supertrend latched in state.
#   Stage 2 — CONFIRM: volume, VWAP, RSI slope, OI, HTF must hold for
#             SIGNAL_HOLD_BARS consecutive cycles before entry fires.
# This prevents the old problem where crossover + all conditions had to
# coincide in the same 2-second polling window (near-zero probability).
# ==============================================================================

def evaluate_signal(df: pd.DataFrame, vwap_value: float,
                    oi: dict, htf_trend: int) -> str:
    """Returns 'call', 'put', or 'none'."""
    last = df.iloc[-1]
    prev = df.iloc[-2]

    # ── Stage 1: structural latch ─────────────────────────────────────────────
    ema_bull    = prev["EMA_9"] <= prev["EMA_21"] and last["EMA_9"] > last["EMA_21"]
    ema_bear    = prev["EMA_9"] >= prev["EMA_21"] and last["EMA_9"] < last["EMA_21"]
    is_trending = float(last["ADX"]) > 25
    st_bull     = int(last["ST_Direction"]) == 1
    st_bear     = int(last["ST_Direction"]) == -1

    if ema_bull and is_trending and st_bull:
        st.session_state.pending_signal      = "call"
        st.session_state.pending_signal_bars = 0
    elif ema_bear and is_trending and st_bear:
        st.session_state.pending_signal      = "put"
        st.session_state.pending_signal_bars = 0

    pending = st.session_state.get("pending_signal", "none")
    if pending == "none":
        return "none"

    # ── Stage 2: confirmation ─────────────────────────────────────────────────
    last_close = float(last["close"])
    rsi_now    = float(last["RSI_14"])
    rsi_prev   = float(prev["RSI_14"])
    is_high_vol = float(last["volume"]) > float(last["Vol_SMA"])

    # OI gate: bypassed when OI API is unavailable (prevents silent blocking)
    if oi.get("oi_available", False):
        oi_ok_call = oi["oi_surge_ce"] or (oi["pcr"] > 1.0)
        oi_ok_put  = oi["oi_surge_pe"] or (oi["pcr"] < 1.0)
    else:
        oi_ok_call = True
        oi_ok_put  = True

    htf_ok_call = htf_trend >= 0   # bull or neutral on 5-min
    htf_ok_put  = htf_trend <= 0   # bear or neutral on 5-min

    if pending == "call":
        confirmed = (
            is_high_vol
            and last_close > vwap_value
            and rsi_now > rsi_prev and 45 < rsi_now < 75
            and oi_ok_call
            and htf_ok_call
        )
    elif pending == "put":
        confirmed = (
            is_high_vol
            and last_close < vwap_value
            and rsi_now < rsi_prev and 25 < rsi_now < 55
            and oi_ok_put
            and htf_ok_put
        )
    else:
        confirmed = False

    if confirmed:
        st.session_state.pending_signal_bars = \
            st.session_state.get("pending_signal_bars", 0) + 1
    else:
        st.session_state.pending_signal      = "none"
        st.session_state.pending_signal_bars = 0
        return "none"

    if st.session_state.pending_signal_bars >= SIGNAL_HOLD_BARS:
        fired = pending
        st.session_state.pending_signal      = "none"
        st.session_state.pending_signal_bars = 0
        return fired

    return "none"


# ==============================================================================
# SESSION STATE BOOTSTRAP
# ==============================================================================
st.set_page_config(page_title="Upstox Algo Scalper", layout="wide")

defaults = {
    "bot_active":          False,
    "current_position":    None,
    "trade_logs":          [],
    "session_pnl":         0.0,
    "loss_streak":         0,
    "oi_snapshot":         {},
    "oi_history":          [],
    "order_log":           [],
    "api_last_call":       {},
    "api_errors":          [],
    "pending_signal":      "none",
    "pending_signal_bars": 0,
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
STOP_LOSS_PCT   = st.sidebar.slider("Stop Loss % (ATR fallback)", 1, 10, DEFAULT_STOP_LOSS_PCT) / 100.0
TARGET_PCT      = st.sidebar.slider("Target % (ATR fallback)",   1, 25, DEFAULT_TARGET_PCT)    / 100.0
ATR_MULTIPLIER  = st.sidebar.slider("ATR SL Multiplier", 1.0, 3.0, 1.5, step=0.1)
RR_MIN          = st.sidebar.slider("Min Risk:Reward",   1.0, 4.0, float(DEFAULT_RR_MIN), step=0.1)
LOT_SIZE        = st.sidebar.number_input("Lot Size", value=NIFTY_LOT_SIZE, min_value=1, step=1)

st.sidebar.markdown("---")
st.sidebar.header("🛡️ Risk Guardrails")
MAX_DAILY_LOSS = st.sidebar.number_input("Daily Loss Limit (₹)", value=DEFAULT_MAX_DAILY_LOSS,
                                          step=500, min_value=500)
MAX_TRADES     = st.sidebar.number_input("Max Trades / Day", value=DEFAULT_MAX_TRADES,
                                          step=1, min_value=1)

st.sidebar.markdown("---")
TRADE_MODE = st.sidebar.radio(
    "🎯 Execution Mode",
    ["📄 Paper Trading (Simulated)", "⚡ Live Trading (Real Money)"],
)
IS_PAPER = "Paper" in TRADE_MODE

st.sidebar.markdown("---")
st.sidebar.caption(f"⚡ Circuit breaker after **{MAX_LOSS_STREAK}** consecutive losses")
st.sidebar.caption(f"📊 OI surge threshold: **{OI_SURGE_RATIO}×** 5-bar avg")
st.sidebar.caption(f"💸 Slippage: ₹{SLIPPAGE_PER_SIDE}/unit/side")
st.sidebar.caption(f"🕐 Server IST: **{ist_time_str()}**")

current_capital = INITIAL_CAPITAL + st.session_state.session_pnl

if ACCESS_TOKEN and "instrument_master" not in st.session_state:
    with st.sidebar:
        with st.spinner("Loading instrument master…"):
            load_instrument_master()

# ==============================================================================
# MAIN DASHBOARD
# ==============================================================================
st.title("🦅 Upstox Options Advanced Scalping Engine")

# ── Row 1: KPI Metrics ────────────────────────────────────────────────────────
m1, m2, m3, m4, m5, m6 = st.columns(6)
nifty_spot  = fetch_ltp(ACCESS_TOKEN, "NSE_INDEX|Nifty 50") if ACCESS_TOKEN else None
trade_count = len(st.session_state.trade_logs)

m1.metric("📊 Nifty Spot",   f"₹{nifty_spot:,.2f}" if nifty_spot else "—")
m2.metric("💰 Balance",      f"₹{current_capital:,.2f}")
m3.metric("📈 Session PnL",  f"₹{st.session_state.session_pnl:,.2f}",
          delta=f"{(st.session_state.session_pnl/INITIAL_CAPITAL*100):.2f}%" if INITIAL_CAPITAL else "0%")
m4.metric("🛡️ Mode",         "LIVE" if not IS_PAPER else "PAPER")
m5.metric("🔴 Loss Streak",  f"{st.session_state.loss_streak} / {MAX_LOSS_STREAK}",
          delta="⚠️ At limit!" if st.session_state.loss_streak >= MAX_LOSS_STREAK - 1 else None)
m6.metric("📋 Trades Today", f"{trade_count} / {MAX_TRADES}",
          delta="⚠️ Near limit!" if trade_count >= MAX_TRADES - 1 else None)

# ── Session window banner ──────────────────────────────────────────────────────
if in_prime_session():
    st.success(f"🟢 **Prime Session Active** ({ist_time_str()}) — scanning for entries")
else:
    st.warning(f"🕐 **Outside Prime Window** ({ist_time_str()}) — next: {next_window_str()}")

# ── Pending signal indicator ───────────────────────────────────────────────────
pending = st.session_state.get("pending_signal", "none")
if pending != "none":
    bars  = st.session_state.get("pending_signal_bars", 0)
    dlbl  = "📈 CALL" if pending == "call" else "📉 PUT"
    st.info(f"⏳ **Stage 1 latched: {dlbl}** — confirming "
            f"({bars}/{SIGNAL_HOLD_BARS} bars confirmed)")

st.markdown("---")

# ── Row 2: Controls | OI | Position ──────────────────────────────────────────
col_ctrl, col_oi, col_pos = st.columns([1, 1, 2])

with col_ctrl:
    st.subheader("🕹️ Engine Controls")
    if not ACCESS_TOKEN:
        st.warning("Provide an Upstox Access Token to activate.")
    else:
        if not st.session_state.bot_active:
            if st.button("▶️ START BOT", type="primary", width='stretch'):
                st.session_state.bot_active          = True
                st.session_state.loss_streak         = 0
                st.session_state.pending_signal      = "none"
                st.session_state.pending_signal_bars = 0
                st.rerun()
        else:
            if st.button("🛑 EMERGENCY HALT", type="secondary", width='stretch'):
                if st.session_state.current_position and not IS_PAPER:
                    pos     = st.session_state.current_position
                    ltp_now = fetch_ltp(ACCESS_TOKEN, pos["key"]) or pos["entry_price"]
                    exit_position(ACCESS_TOKEN, pos, ltp_now, "MANUAL HALT 🛑", IS_PAPER, LOT_SIZE)
                st.session_state.bot_active       = False
                st.session_state.current_position = None
                st.rerun()

        status = "🟢 Active" if st.session_state.bot_active else "🔴 Inactive"
        st.info(f"State: **{status}**")

        if st.session_state.trade_logs:
            csv_bytes = pd.DataFrame(st.session_state.trade_logs).to_csv(index=False).encode()
            st.download_button("📥 Trade Log (CSV)", data=csv_bytes,
                               file_name=f"scalper_{date.today()}.csv",
                               mime="text/csv", width='stretch')

with col_oi:
    st.subheader("\U0001f4ca OI Intelligence")

    if ACCESS_TOKEN:
        _spot_for_oi = nifty_spot if nifty_spot else 24000.0
        _master   = st.session_state.get("instrument_master", pd.DataFrame())
        _expiries = get_weekly_expiries(_master) if not _master.empty else []
        if _expiries:
            _expiry_str = _expiries[0].strftime("%Y-%m-%d")
        else:
            _today = date.today()
            _dtt   = (3 - _today.weekday()) % 7
            if _dtt == 0 and now_ist().hour >= 15:
                _dtt = 7
            _expiry_str = (_today + timedelta(days=_dtt)).strftime("%Y-%m-%d")

        _oi_url = _build_url(
            f"{UPSTOX_BASE_URL}/option/chain",
            {"instrument_key": "NSE_INDEX|Nifty 50", "expiry_date": _expiry_str}
        )

        with st.expander("\U0001f50d OI Diagnostic", expanded=not bool(st.session_state.get("oi_snapshot"))):
            st.caption(f"Spot ref: \u20b9{_spot_for_oi:,.0f} | Expiry: **{_expiry_str}**")
            st.code(_oi_url, language="text")
            if st.button("\U0001f504 Force Refresh OI", width='stretch'):
                st.session_state.oi_snapshot = {}
                st.session_state.oi_history  = []
                st.rerun()
            _errs = [e for e in st.session_state.get("api_errors", [])
                     if "option" in e.get("endpoint", "")]
            if _errs:
                _e = _errs[-1]
                st.error(f"Last error {_e['time']} \u2014 HTTP {_e['status']}: {_e['body']}")
            else:
                st.success("No OI API errors logged.")

        if not st.session_state.get("oi_snapshot"):
            fetch_option_chain_oi(ACCESS_TOKEN, _spot_for_oi)

    snap = st.session_state.get("oi_snapshot", {})
    if snap:
        ce_oi    = snap.get("ce_oi", 0)
        pe_oi    = snap.get("pe_oi", 0)
        pcr      = pe_oi / ce_oi if ce_oi > 0 else 1.0
        o1, o2   = st.columns(2)
        o1.metric("CE OI", f"{ce_oi/1e5:.2f}L")
        o2.metric("PE OI", f"{pe_oi/1e5:.2f}L")
        pcr_icon = "\U0001f7e2" if pcr > 1.2 else "\U0001f534" if pcr < 0.8 else "\U0001f7e1"
        st.metric("PCR", f"{pcr_icon} {pcr:.2f}",
                  help="PCR > 1.2 = bullish, < 0.8 = bearish")
        hist = st.session_state.get("oi_history", [])
        if len(hist) > 1:
            st.line_chart(pd.DataFrame(hist).tail(15)[["ce_oi", "pe_oi"]],
                          height=120, width='stretch')
    elif ACCESS_TOKEN:
        st.warning("\u26a0\ufe0f OI fetch returned no data \u2014 see diagnostic above.")

with col_pos:
    st.subheader("📦 Active Position")
    pos = st.session_state.current_position
    if pos and ACCESS_TOKEN:
        pos_ltp = fetch_ltp(ACCESS_TOKEN, pos["key"]) or pos["entry_price"]
        unr_pnl = (pos_ltp - pos["entry_price"]) * LOT_SIZE - SLIPPAGE_PER_SIDE * 2 * LOT_SIZE
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Symbol", pos["symbol"])
        c2.metric("Entry",  f"₹{pos['entry_price']:.2f}")
        c3.metric("LTP",    f"₹{pos_ltp:.2f}")
        c4.metric("Est. Net PnL", f"₹{unr_pnl:.2f}",
                  delta=f"{((pos_ltp-pos['entry_price'])/pos['entry_price']*100):.1f}%")
        risk_pts = abs(pos["entry_price"] - pos["sl_price"])
        rwd_pts  = abs(pos["target_price"] - pos["entry_price"])
        rr_setup = round(rwd_pts / risk_pts, 2) if risk_pts > 0 else 0
        st.caption(
            f"🎯 Target: **₹{pos['target_price']:.2f}** | "
            f"🛡️ Trailing SL: **₹{pos['sl_price']:.2f}** | "
            f"📈 Peak: **₹{pos['highest_price']:.2f}** | "
            f"⚖️ RR: **{rr_setup}** | "
            f"🆔 Order: `{pos.get('entry_order_id','—')}`"
        )
    else:
        st.info("No open position. System is flat.")

# ==============================================================================
# ALGORITHMIC LOOP ENGINE
# ==============================================================================
if st.session_state.bot_active and ACCESS_TOKEN:

    # ── Daily guardrail checks ────────────────────────────────────────────────
    allowed, reason = check_daily_guardrails(
        st.session_state.session_pnl, trade_count, MAX_DAILY_LOSS, MAX_TRADES
    )
    if not allowed:
        if "limit" in reason.lower() or "max" in reason.lower():
            st.session_state.bot_active = False
        st.warning(reason)
        time.sleep(30)
        st.rerun()

    # ── 1-min candle data ─────────────────────────────────────────────────────
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
    df["ST_Direction"] = compute_supertrend(df)

    last       = df.iloc[-1]
    prev       = df.iloc[-2]
    last_close = float(last["close"])

    # ── VWAP scalar ───────────────────────────────────────────────────────────
    live_vwap = fetch_vwap_from_ohlc(ACCESS_TOKEN, "NSE_INDEX|Nifty 50")
    if live_vwap and live_vwap > 0:
        vwap_value = live_vwap
    else:
        tp          = (df["high"] + df["low"] + df["close"]) / 3
        vwap_series = (tp * df["volume"]).cumsum() / df["volume"].cumsum().replace(0, float("nan"))
        valid       = vwap_series.dropna()
        vwap_value  = float(valid.iloc[-1]) if not valid.empty else last_close

    # ── OI + HTF ──────────────────────────────────────────────────────────────
    spot_ref  = nifty_spot or last_close
    oi        = fetch_option_chain_oi(ACCESS_TOKEN, spot_ref)
    htf_trend = get_htf_trend(ACCESS_TOKEN)
    master    = load_instrument_master()

    # ── Signal evaluation ─────────────────────────────────────────────────────
    if st.session_state.current_position is None:
        signal = evaluate_signal(df, vwap_value, oi, htf_trend)

        if signal in ("call", "put"):
            ot            = "CE" if signal == "call" else "PE"
            ikey, isymbol = resolve_atm_option_key(ACCESS_TOKEN, spot_ref, ot, master)
            entry_ltp     = fetch_ltp(ACCESS_TOKEN, ikey)

            if entry_ltp is None or entry_ltp <= 0:
                st.warning(f"⚠️ No valid LTP for {isymbol} — skipping.")
            else:
                atr_levels = compute_atr_sl_target(df, entry_ltp, ATR_MULTIPLIER, RR_MIN)
                if atr_levels:
                    sl_price, target_price = atr_levels
                else:
                    sl_price     = entry_ltp * (1 - STOP_LOSS_PCT)
                    target_price = entry_ltp * (1 + TARGET_PCT)

                risk_pts = entry_ltp - sl_price
                rwd_pts  = target_price - entry_ltp
                rr_setup = rwd_pts / risk_pts if risk_pts > 0 else 0

                if rr_setup < RR_MIN:
                    st.warning(f"⚖️ RR {rr_setup:.2f} < min {RR_MIN} — skipping {isymbol}.")
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
                            "sl_price":       sl_price,
                            "target_price":   target_price,
                            "entry_time":     now_ist().strftime("%H:%M:%S"),
                            "direction":      ot,
                            "entry_order_id": order_id,
                            "rr_setup":       round(rr_setup, 2),
                        }
                        lbl = "📈 CALL" if ot == "CE" else "📉 PUT"
                        st.success(
                            f"🚨 {lbl} — {isymbol} @ ₹{entry_ltp:.2f} | "
                            f"SL ₹{sl_price:.2f} | Tgt ₹{target_price:.2f} | "
                            f"RR {rr_setup:.2f} | Order: {order_id}"
                        )
                        st.rerun()

    # ── Position management ───────────────────────────────────────────────────
    else:
        pos     = st.session_state.current_position
        pos_ltp = fetch_ltp(ACCESS_TOKEN, pos["key"])
        if pos_ltp is None:
            st.warning("⚠️ LTP fetch failed — retrying next cycle.")
        else:
            if pos_ltp > pos["highest_price"]:
                st.session_state.current_position["highest_price"] = pos_ltp
                atr_levels = compute_atr_sl_target(df, pos_ltp, ATR_MULTIPLIER, RR_MIN)
                new_sl     = atr_levels[0] if atr_levels else pos_ltp * (1 - STOP_LOSS_PCT)
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

    st.dataframe(df_logs.style.applymap(style_pnl, subset=["Net PnL ₹"]),
                 width='stretch')

    total      = len(df_logs)
    wins       = (df_logs["Net PnL ₹"] > 0).sum()
    losses     = total - wins
    wr         = wins / total * 100 if total else 0
    avg_w      = df_logs[df_logs["Net PnL ₹"] > 0]["Net PnL ₹"].mean() if wins   else 0
    avg_l      = df_logs[df_logs["Net PnL ₹"] <= 0]["Net PnL ₹"].mean() if losses else 0
    rr_avg     = df_logs["RR Realised"].mean() if "RR Realised" in df_logs.columns else 0
    ev         = (wr / 100 * avg_w) + ((1 - wr / 100) * avg_l)
    total_slip = df_logs["Slippage ₹"].sum() if "Slippage ₹" in df_logs.columns else 0

    s1, s2, s3, s4, s5, s6, s7 = st.columns(7)
    s1.metric("Trades",         total)
    s2.metric("Win Rate",       f"{wr:.1f}%")
    s3.metric("Avg Win",        f"₹{avg_w:.0f}")
    s4.metric("Avg Loss",       f"₹{avg_l:.0f}")
    s5.metric("Avg RR",         f"{rr_avg:.2f}",
              delta="✅ On target" if rr_avg >= RR_MIN else "⚠️ Below target")
    s6.metric("Expected Value", f"₹{ev:.0f}/trade",
              delta="✅ Positive" if ev > 0 else "🔴 Negative edge")
    s7.metric("Total Slippage", f"₹{total_slip:.0f}")

    with st.expander("📈 Equity Curve"):
        df_logs["Cumulative PnL"] = df_logs["Net PnL ₹"].cumsum()
        st.line_chart(df_logs["Cumulative PnL"], width='stretch')
else:
    st.info("No trades executed this session.")

# ── Order log + API error log ─────────────────────────────────────────────────
if st.session_state.order_log:
    with st.expander("🗂️ Raw Order Log"):
        st.dataframe(pd.DataFrame(st.session_state.order_log), width='stretch')

if st.session_state.get("api_errors"):
    with st.expander(f"⚠️ API Error Log ({len(st.session_state.api_errors)} errors)"):
        st.dataframe(pd.DataFrame(st.session_state.api_errors), width='stretch')
