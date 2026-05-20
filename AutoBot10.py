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
                           atr_multiplie
