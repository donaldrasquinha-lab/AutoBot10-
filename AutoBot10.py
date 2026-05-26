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


def get_active_expiry_str(master: pd.DataFrame | None = None) -> str:
    """
    Returns the correct weekly Nifty expiry date (YYYY-MM-DD) in IST.

    Rules:
      - Non-Tuesday: nearest upcoming Tuesday
      - Tuesday before 15:25 IST: TODAY (contracts still live)
      - Tuesday after  15:25 IST: next Tuesday (roll over after settlement)

    If an instrument master is provided and has upcoming expiries,
    the first valid date from the master is used instead of the formula
    — this handles exchange holiday rollovers automatically.
    """
    now       = now_ist()
    today_ist = now.date()
    cutoff    = now.replace(hour=15, minute=25, second=0, microsecond=0)

    # ── Master-based resolution (most accurate) ───────────────────────────────
    if master is not None and not master.empty:
        expiries = get_weekly_expiries(master)   # already filtered to >= today IST
        if expiries:
            nearest = expiries[0]
            # If nearest expiry is today AND we are past cutoff → use next one
            if nearest == today_ist and now >= cutoff and len(expiries) > 1:
                return expiries[1].strftime("%Y-%m-%d")
            return nearest.strftime("%Y-%m-%d")

    # ── Formula fallback ──────────────────────────────────────────────────────
    days_to_expiry = (1 - today_ist.weekday()) % 7  # 0 when today is Tuesday (expiry day)

    if days_to_expiry == 0:
        # Today is expiry day
        if now >= cutoff:
            days_to_expiry = 7   # roll to next week after 15:25
        # else: stay at 0 — use today's expiry

    expiry = today_ist + timedelta(days=days_to_expiry)
    return expiry.strftime("%Y-%m-%d")


def fetch_valid_expiries(token: str) -> list[str]:
    """
    Fetches real available expiry dates directly from Upstox
    /v2/option/contract endpoint. Returns a sorted list of
    YYYY-MM-DD strings >= today IST. Falls back to formula if API fails.
    This is the only reliable way to know which expiries Upstox has live data for.
    """
    cached = st.session_state.get("valid_expiries", [])
    if cached:
        return cached

    url  = _build_url(
        f"{UPSTOX_BASE_URL}/option/contract",
        {"instrument_key": "NSE_INDEX|Nifty 50"}
    )
    data = _raw_get(token, url)
    if data is None:
        return []

    try:
        contracts  = data.get("data", [])
        today_ist  = now_ist().date()
        expiry_set = set()
        for c in contracts:
            exp_str = c.get("expiry", "")
            if exp_str:
                try:
                    exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                    if exp_date >= today_ist:
                        expiry_set.add(exp_str)
                except ValueError:
                    continue
        sorted_expiries = sorted(expiry_set)
        st.session_state["valid_expiries"] = sorted_expiries
        return sorted_expiries
    except Exception:
        return []


def get_active_expiry_from_upstox(token: str) -> str:
    """
    Gets the correct active expiry date from Upstox directly.
    On expiry day before 15:25 IST  → returns today's expiry.
    On expiry day after  15:25 IST  → returns next available expiry.
    All other days                  → returns nearest upcoming expiry.
    Falls back to get_active_expiry_str() formula if API unavailable.
    """
    expiries = fetch_valid_expiries(token)

    if not expiries:
        # API unavailable — use formula fallback
        master = st.session_state.get("instrument_master", pd.DataFrame())
        return get_active_expiry_str(master)

    now       = now_ist()
    today_str = now_ist().date().strftime("%Y-%m-%d")
    cutoff    = now.replace(hour=15, minute=25, second=0, microsecond=0)

    # If today is in the list and we're before cutoff — use today
    if today_str in expiries and now < cutoff:
        return today_str

    # Otherwise use the first future expiry (skips today if past cutoff)
    for exp in expiries:
        if exp > today_str:
            return exp
        if exp == today_str and now < cutoff:
            return exp

    # All expiries are in the past — formula fallback
    master = st.session_state.get("instrument_master", pd.DataFrame())
    return get_active_expiry_str(master)



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
    today    = now_ist().date()
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
    expiry_str = get_active_expiry_str()
    expiry     = datetime.strptime(expiry_str, "%Y-%m-%d").date()
    symbol     = f"NIFTY{expiry.strftime('%y%b%d').upper()}{atm_strike}{ot}"
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
    """
    Fetches intraday candles. Results are cached in session_state for 30 seconds
    per interval to avoid redundant API calls when multiple TF layers share the
    same underlying feed (e.g. 15M and 3M both built from 1min).
    """
    cache_key = f"_candle_cache_{interval}"
    cached    = st.session_state.get(cache_key)
    if cached and (time.time() - cached["ts"]) < 30:
        return cached["df"].copy()

    safe_key = instrument_key.replace(" ", "%20")
    url      = f"{UPSTOX_BASE_URL}/historical-candle/intraday/{safe_key}/{interval}"
    data = _raw_get(token, url)
    if data is None:
        return pd.DataFrame()
    try:
        candles = data["data"]["candles"]
        if not candles:
            # Empty candles list — do NOT cache, allow retry next cycle
            return pd.DataFrame()
        df = pd.DataFrame(candles,
                          columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
        df = df.iloc[::-1].reset_index(drop=True)
        for col in ["open", "high", "low", "close", "volume", "oi"]:
            df[col] = pd.to_numeric(df[col])
        # Only cache non-empty successful results
        if not df.empty:
            st.session_state[cache_key] = {"df": df, "ts": time.time()}
        return df
    except Exception as e:
        # Log the parse error so it appears in the API error log
        st.session_state.setdefault("api_errors", []).append({
            "time":     now_ist().strftime("%H:%M:%S"),
            "endpoint": url.split("api.upstox.com")[-1].split("?")[0],
            "status":   "parse_error",
            "body":     str(e)[:200],
        })
        return pd.DataFrame()


def fetch_historical_candles_multi_day(token: str, instrument_key: str,
                                        interval: str, days_back: int = 60) -> pd.DataFrame:
    """
    Fetches historical candles using the NON-intraday endpoint:
      GET /v2/historical-candle/{key}/{interval}/{to_date}/{from_date}

    Required for 30minute trend analysis — intraday endpoint only returns today's bars.
    Returns up to days_back calendar days of data, newest-last (chronological).
    """
    from datetime import date, timedelta
    safe_key  = instrument_key.replace(" ", "%20")
    to_date   = now_ist().date().strftime("%Y-%m-%d")
    from_date = (now_ist().date() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    cache_key = f"_candle_cache_multiday_{interval}"
    cached    = st.session_state.get(cache_key)
    if cached and (time.time() - cached["ts"]) < 300:   # 5-minute cache for 30min bars
        return cached["df"].copy()

    url  = f"{UPSTOX_BASE_URL}/historical-candle/{safe_key}/{interval}/{to_date}/{from_date}"
    data = _raw_get(token, url)
    if data is None:
        return pd.DataFrame()
    try:
        candles = data["data"]["candles"]
        df = pd.DataFrame(candles,
                          columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
        df = df.iloc[::-1].reset_index(drop=True)
        for col in ["open", "high", "low", "close", "volume", "oi"]:
            df[col] = pd.to_numeric(df[col])
        st.session_state[cache_key] = {"df": df, "ts": time.time()}
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
    # Get expiry from Upstox directly — most reliable source
    expiry_str = get_active_expiry_from_upstox(token)

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

    # Store raw response for diagnostic panel
    st.session_state["oi_last_raw"] = {
        "status": "200",
        "data_keys": list(data.keys()) if isinstance(data, dict) else str(type(data)),
        "data_length": len(data.get("data", [])) if isinstance(data, dict) else 0,
        "sample": str(data)[:400],
    }

    try:
        chain = data.get("data", [])

        # Empty chain — API returned 200 but no contracts for this expiry
        if not chain:
            st.session_state.setdefault("api_errors", []).append({
                "time": now_ist().strftime("%H:%M:%S"),
                "endpoint": "/option/chain",
                "status": "200_EMPTY",
                "body": (
                    f"Empty data array for expiry_date={expiry_str}. "
                    "Possible causes: (1) expiry not yet listed by Upstox, "
                    "(2) outside market hours — OI data unavailable, "
                    "(3) try next Tuesday expiry."
                ),
            })
            return empty

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
        "Date":        now_ist().date().isoformat(),
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


# ==============================================================================
# MULTI-TIMEFRAME INDICATOR ENGINE
# Architecture:
#   1H  bars → Trend direction  (EMA 20/50, ADX, Supertrend)
#   15M bars → Momentum confirm (EMA 9/21 state, RSI, VWAP side)
#   3M  bars → Trade trigger    (EMA 9/21 state, volume surge, RSI slope)
#
# FIX applied: EMA STATE (EMA_fast > EMA_slow) used everywhere — NOT crossover.
# State is true for many bars during a trend, giving realistic signal frequency.
# ==============================================================================

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


def compute_atr_sl_target(df, entry_ltp, atr_multiplier, rr_min, delta=0.5):
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


def resample_candles(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Resamples 1-min OHLCV candles into n-minute bars by grouping every n rows.
    Used to synthesise 3-min and 15-min candles from the 1-min intraday feed
    since Upstox intraday only supports 1minute and 30minute intervals.
    """
    if df.empty or n <= 1:
        return df
    records = []
    for i in range(0, len(df) - (len(df) % n), n):
        chunk = df.iloc[i : i + n]
        records.append({
            "timestamp": chunk["timestamp"].iloc[0],
            "open":      chunk["open"].iloc[0],
            "high":      chunk["high"].max(),
            "low":       chunk["low"].min(),
            "close":     chunk["close"].iloc[-1],
            "volume":    chunk["volume"].sum(),
            "oi":        chunk["oi"].iloc[-1],
        })
    return pd.DataFrame(records).reset_index(drop=True)


def build_resampled_indicators(df_1min: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Resamples 1-min OHLCV into N-min candles and computes indicators.

    Key insight: ta library fills NaN with 0 by default — so 9 resampled bars
    with EMA21 gives wrong values (0-filled), not NaN. We cannot use dropna
    to detect bad values. Instead we ensure enough resampled bars exist:
      - EMA21 needs >=21 bars to be accurate
      - RSI14 needs >=14 bars
      - Minimum safe threshold: 22 resampled bars

    If fewer than 22 bars exist after resampling, return empty df so the
    caller handles it gracefully rather than using wrong indicator values.
    """
    MIN_RESAMPLED = 22   # enough for EMA21 + 1 buffer bar

    if df_1min.empty or len(df_1min) < n * MIN_RESAMPLED:
        return pd.DataFrame()

    # Resample OHLCV
    records = []
    for i in range(0, len(df_1min) - (len(df_1min) % n), n):
        chunk = df_1min.iloc[i : i + n]
        records.append({
            "timestamp": chunk["timestamp"].iloc[-1],
            "open":      chunk["open"].iloc[0],
            "high":      chunk["high"].max(),
            "low":       chunk["low"].min(),
            "close":     chunk["close"].iloc[-1],
            "volume":    chunk["volume"].sum(),
            "oi":        chunk["oi"].iloc[-1],
        })
    if not records or len(records) < MIN_RESAMPLED:
        return pd.DataFrame()

    df = pd.DataFrame(records).reset_index(drop=True)

    # Compute EMA and RSI on the properly-sized resampled series
    df["EMA_9"]  = ta.trend.ema_indicator(df["close"], window=9)
    df["EMA_21"] = ta.trend.ema_indicator(df["close"], window=21)
    df["RSI"]    = ta.momentum.rsi(df["close"], window=14)

    return df



def get_tf_data(token: str, interval: str, min_bars: int):
    """
    Fetch candles for a given interval. Returns (df, ok: bool).

    Upstox intraday  endpoint accepts: 1minute, 30minute only.
    Upstox historical endpoint accepts: 1minute, 30minute, day, week, month.

    Mapping:
      30minute (1H proxy) -> multi-day historical (30minute)
      15minute            -> intraday 1minute, resampled x15
      3minute             -> intraday 1minute, resampled x3
      1minute             -> intraday 1minute directly
    """
    if interval == "30minute":
        df = fetch_historical_candles_multi_day(
            token, "NSE_INDEX|Nifty 50", interval="30minute", days_back=60
        )
    elif interval == "15minute":
        raw = fetch_historical_candles(token, "NSE_INDEX|Nifty 50", interval="1minute")
        df  = resample_candles(raw, 15)
    elif interval == "3minute":
        raw = fetch_historical_candles(token, "NSE_INDEX|Nifty 50", interval="1minute")
        df  = resample_candles(raw, 3)
    else:
        df = fetch_historical_candles(token, "NSE_INDEX|Nifty 50", interval=interval)

    if df.empty or len(df) < min_bars:
        return pd.DataFrame(), False
    return df, True
def build_1h_trend(token: str) -> dict:
    """
    30-Minute bars used as 1H proxy (Upstox historical API supports 30minute, not 60minute).
    104 bars = ~52 trading hours of context.
    Trend LAYER.
    Determines overall market direction.
    Indicators: EMA 20/50 state, Supertrend only. No ADX — direction is enough here.
    Returns dict of filter states + direction (+1 bull, -1 bear, 0 neutral).
    """
    df, ok = get_tf_data(token, "30minute", 104)
    # Store bar count for debug display in filter panel
    st.session_state["_debug_1h_bars"] = list(range(len(df))) if not df.empty else []
    if not ok:
        return {
            "ok": False, "direction": 0, "filters": {},
            "debug": f"{len(df)} bars received (need 104 × 30min bars = ~52 hours of context)",
        }

    df["EMA_20"] = ta.trend.ema_indicator(df["close"], window=40)   # 40×30min ≈ 20×1H
    df["EMA_50"] = ta.trend.ema_indicator(df["close"], window=100)  # 100×30min ≈ 50×1H
    df["ST"]     = compute_supertrend(df, length=14, multiplier=3.0)  # scaled for 30min

    last = df.iloc[-1]
    ema_bull = float(last["EMA_20"]) > float(last["EMA_50"])
    ema_bear = float(last["EMA_20"]) < float(last["EMA_50"])
    st_bull  = int(last["ST"]) == 1
    st_bear  = int(last["ST"]) == -1

    if ema_bull and st_bull:
        direction = 1
    elif ema_bear and st_bear:
        direction = -1
    else:
        direction = 0

    # Only show filters relevant to the detected direction
    if direction == 1:
        filters = {
            "EMA 20 > EMA 50": ema_bull,
            "Supertrend bull":  st_bull,
        }
    elif direction == -1:
        filters = {
            "EMA 20 < EMA 50": ema_bear,
            "Supertrend bear":  st_bear,
        }
    else:
        # Neutral — show why both directions failed
        filters = {
            "EMA 20 > EMA 50 (bull need)": ema_bull,
            "Supertrend bull (bull need)":  st_bull,
            "EMA 20 < EMA 50 (bear need)":  ema_bear,
            "Supertrend bear (bear need)":  st_bear,
        }
    return {
        "ok": True, "direction": direction, "filters": filters,
        "ema20": float(last["EMA_20"]), "ema50": float(last["EMA_50"]),
        "st": int(last["ST"]), "close": float(last["close"]),
    }


def build_15m_confirm(token: str, trend_dir: int, vwap_value: float, oi: dict) -> dict:
    """
    15-Minute timeframe — MOMENTUM LAYER.
    Confirms the 1H trend has momentum behind it.
    Indicators: EMA 9/21 state, RSI 14, VWAP side, OI/PCR.
    Only evaluated when 1H direction is non-zero.
    """
    # Fetch 1min bars. Need 330+ for accurate 15M resampling (22 bars × 15min).
    # With fewer bars, use 1min indicators directly — less accurate but valid.
    df_1min, ok_1min = get_tf_data(token, "1minute", 20)
    if not ok_1min:
        return {"ok": False, "confirmed": False, "filters": {}, "bars": 0}

    df_15m = build_resampled_indicators(df_1min, 15)

    if df_15m.empty:
        # Not enough bars for proper 15M — use 1min series with standard windows
        # This is used early session (<5.5 hrs elapsed) only
        df_work = df_1min.copy()
        df_work["EMA_9"]  = ta.trend.ema_indicator(df_work["close"], window=9)
        df_work["EMA_21"] = ta.trend.ema_indicator(df_work["close"], window=21)
        df_work["RSI"]    = ta.momentum.rsi(df_work["close"], window=14)
        df_work = df_work.dropna(subset=["EMA_9", "EMA_21", "RSI"])
        if df_work.empty:
            return {"ok": False, "confirmed": False, "filters": {}, "bars": len(df_1min)}
        df_15m = df_work   # use 1min as proxy — note in panel
        using_proxy = True
    else:
        using_proxy = False

    last = df_15m.iloc[-1]
    ema_state_bull = float(last["EMA_9"]) > float(last["EMA_21"])
    ema_state_bear = float(last["EMA_9"]) < float(last["EMA_21"])
    rsi_val        = float(last["RSI"])
    rsi_mid_bull   = 45 < rsi_val < 75  # 45 floor accounts for 1min proxy underestimation
    rsi_mid_bear   = 25 < rsi_val < 55  # 55 ceiling accounts for 1min proxy overestimation
    close_15m      = float(last["close"])
    above_vwap     = close_15m > vwap_value
    below_vwap     = close_15m < vwap_value

    # OI gate — bypassed if API unavailable
    if oi.get("oi_available", False):
        oi_ok_call = oi["oi_surge_ce"] or (oi["pcr"] > 1.0)
        oi_ok_put  = oi["oi_surge_pe"] or (oi["pcr"] < 1.0)
    else:
        oi_ok_call = True
        oi_ok_put  = True

    if trend_dir == 1:
        confirmed = ema_state_bull and rsi_mid_bull and above_vwap and oi_ok_call
        filters = {
            "EMA 9>21 (bull state)": ema_state_bull,
            "RSI 45–75":             rsi_mid_bull,
            "Price > VWAP":          above_vwap,
            "OI/PCR confirms call":  oi_ok_call,
        }
    elif trend_dir == -1:
        confirmed = ema_state_bear and rsi_mid_bear and below_vwap and oi_ok_put
        filters = {
            "EMA 9<21 (bear state)": ema_state_bear,
            "RSI 25–55":             rsi_mid_bear,
            "Price < VWAP":          below_vwap,
            "OI/PCR confirms put":   oi_ok_put,
        }
    else:
        confirmed = False
        filters = {"No trend on 1H": False}

    return {
        "ok": True, "confirmed": confirmed, "filters": filters,
        "ema9": float(last["EMA_9"]), "ema21": float(last["EMA_21"]),
        "rsi": rsi_val, "close": close_15m,
        "vwap": vwap_value,
        "pcr": oi.get("pcr", 0),
        "oi_available": oi.get("oi_available", False),
        "bars": len(df_15m) if not df_15m.empty else 0,
        "proxy": using_proxy,
    }


def build_3m_trigger(token: str, trend_dir: int) -> dict:
    """
    3-Minute timeframe — ENTRY TRIGGER LAYER.
    Fine-tunes exact entry timing. Only evaluated when 1H + 15M agree.
    Indicators: EMA 9/21 state, RSI slope, volume surge, ADX > 20.
    ADX here confirms the 3M move has real momentum — not just a random wiggle.
    Uses STATE not crossover — avoids the timing coincidence problem entirely.
    """
    # Fetch 1min bars and resample to proper 3M OHLCV, then compute indicators
    df_1min_3m, ok_1min_3m = get_tf_data(token, "1minute", 30)
    if not ok_1min_3m:
        return {"ok": False, "triggered": False, "filters": {}, "df": pd.DataFrame()}

    df = build_resampled_indicators(df_1min_3m, 3)
    if len(df) < 2:
        return {"ok": False, "triggered": False, "filters": {}, "df": pd.DataFrame()}

    # ADX needs at least (2 × window + 1) bars — only compute if enough exist
    adx_window = 7
    if len(df) >= adx_window * 2 + 1:
        df["ADX"] = ta.trend.adx(df["high"], df["low"], df["close"], window=adx_window)
        adx_val   = float(df["ADX"].dropna().iloc[-1]) if not df["ADX"].dropna().empty else 25.0
    else:
        adx_val = 25.0   # assume trending when not enough bars

    # Bar range expansion — proxy for volume surge on index instruments
    # (NSE_INDEX volume is always 0; candle range expansion signals momentum)
    df["Bar_Range"]     = df["high"] - df["low"]
    df["Bar_Range_SMA"] = df["Bar_Range"].rolling(window=min(10, len(df))).mean()
    df = df.dropna(subset=["Bar_Range_SMA"]).reset_index(drop=True)
    if len(df) < 2:
        return {"ok": False, "triggered": False, "filters": {}, "df": pd.DataFrame()}

    last = df.iloc[-1]
    prev = df.iloc[-2]

    ema_state_bull = float(last["EMA_9"]) > float(last["EMA_21"])
    ema_state_bear = float(last["EMA_9"]) < float(last["EMA_21"])
    rsi_now        = float(last["RSI"])
    rsi_prev       = float(prev["RSI"])
    rsi_rising     = rsi_now > rsi_prev and 45 < rsi_now < 78
    rsi_falling    = rsi_now < rsi_prev and 22 < rsi_now < 55
    # Range expansion: current bar range > 10-bar average range (momentum proxy)
    range_expand   = float(last["Bar_Range"]) > float(last["Bar_Range_SMA"])
    adx_strong     = adx_val > 20

    if trend_dir == 1:
        triggered = ema_state_bull and rsi_rising and range_expand and adx_strong
        filters = {
            "EMA 9>21 (bull state)":   ema_state_bull,
            "RSI rising 45–78":        rsi_rising,
            "Range expansion":         range_expand,
            "ADX > 20 (momentum)":     adx_strong,
        }
    elif trend_dir == -1:
        triggered = ema_state_bear and rsi_falling and range_expand and adx_strong
        filters = {
            "EMA 9<21 (bear state)":   ema_state_bear,
            "RSI falling 22–55":       rsi_falling,
            "Range expansion":         range_expand,
            "ADX > 20 (momentum)":     adx_strong,
        }
    else:
        triggered = False
        filters = {"No trend direction": False}

    return {
        "ok": True, "triggered": triggered, "filters": filters,
        "ema9":     float(last["EMA_9"]),
        "ema21":    float(last["EMA_21"]),
        "rsi":      rsi_now,
        "rsi_prev": rsi_prev,
        "adx":      adx_val,
        "range":     float(last["Bar_Range"]),
        "range_sma": float(last["Bar_Range_SMA"]) if not pd.isna(last["Bar_Range_SMA"]) else 0.0,
        "df":       df,
    }


def evaluate_mtf_signal(token: str, vwap_value: float, oi: dict) -> tuple:
    """
    Master MTF evaluator. Runs all three timeframe layers in sequence.
    Returns (signal: str, tf_data: dict) where signal is 'call'/'put'/'none'.
    tf_data carries all filter states for the dashboard status panel.
    Short-circuits: if 1H is neutral, 15M and 3M are not fetched.
    """
    tf = {}

    # ── Layer 1: 1H Trend ────────────────────────────────────────────────────
    tf["1h"] = build_1h_trend(token)
    trend_dir = tf["1h"].get("direction", 0)

    if trend_dir == 0:
        tf["15m"] = {"ok": False, "confirmed": False, "filters": {"Waiting for 1H trend": False}}
        tf["3m"]  = {"ok": False, "triggered": False, "filters": {"Waiting for 1H trend": False}, "df": pd.DataFrame()}
        return "none", tf

    # ── Layer 2: 15M Momentum ────────────────────────────────────────────────
    tf["15m"] = build_15m_confirm(token, trend_dir, vwap_value, oi)
    if not tf["15m"]["confirmed"]:
        tf["3m"] = {"ok": False, "triggered": False, "filters": {"Waiting for 15M confirm": False}, "df": pd.DataFrame()}
        return "none", tf

    # ── Layer 3: 3M Trigger ──────────────────────────────────────────────────
    tf["3m"] = build_3m_trigger(token, trend_dir)
    if not tf["3m"]["triggered"]:
        return "none", tf

    signal = "call" if trend_dir == 1 else "put"
    return signal, tf


# ==============================================================================
# SESSION STATE BOOTSTRAP
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
    "api_errors":       [],
    "last_tf_data":     {},
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Bust stale cache keys from old invalid interval calls ─────────────────────
# Previous versions called 15minute and 3minute directly against the Upstox
# intraday endpoint which rejects them. Clear any cached failures so the new
# resample-based approach starts fresh on every app load.
for _stale_key in ["_candle_cache_15minute", "_candle_cache_3minute"]:
    if _stale_key in st.session_state:
        del st.session_state[_stale_key]

# Clear API errors older than today to avoid confusing stale errors in the log
_today_str = now_ist().strftime("%H")   # clear if new session hour
if st.session_state.get("_error_log_hour") != _today_str:
    st.session_state["api_errors"]     = []
    st.session_state["_error_log_hour"] = _today_str

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
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Bust stale cache keys from old invalid interval calls ─────────────────────
# Previous versions called 15minute and 3minute directly against the Upstox
# intraday endpoint which rejects them. Clear any cached failures so the new
# resample-based approach starts fresh on every app load.
for _stale_key in ["_candle_cache_15minute", "_candle_cache_3minute"]:
    if _stale_key in st.session_state:
        del st.session_state[_stale_key]

# Clear API errors older than today to avoid confusing stale errors in the log
_today_str = now_ist().strftime("%H")   # clear if new session hour
if st.session_state.get("_error_log_hour") != _today_str:
    st.session_state["api_errors"]     = []
    st.session_state["_error_log_hour"] = _today_str

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
                               file_name=f"scalper_{now_ist().date()}.csv",
                               mime="text/csv", width='stretch')

with col_oi:
    st.subheader("\U0001f4ca OI Intelligence")

    if ACCESS_TOKEN:
        _spot_for_oi = nifty_spot if nifty_spot else 24000.0
        # Get expiry directly from Upstox
        _expiry_str = get_active_expiry_from_upstox(ACCESS_TOKEN)

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
                st.error(f"Last error {_e['time']} — HTTP {_e['status']}: {_e['body']}")
            else:
                st.success("No OI API errors logged.")

            # Show raw API response for deeper diagnosis
            raw = st.session_state.get("oi_last_raw")
            if raw:
                st.caption(
                    f"Last 200 response — keys: `{raw['data_keys']}` | "
                    f"data length: **{raw['data_length']}** rows"
                )
                if raw["data_length"] == 0:
                    st.warning(
                        "Upstox returned 200 OK but **0 contracts** for this expiry. "
                        "This usually means:\n"
                        "- OI data is only available **during market hours** (09:15–15:30 IST)\n"
                        "- The expiry date has no contracts listed yet\n"
                        "- Try enabling the **Options Chain** scope in your Upstox app settings"
                    )

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
# ==============================================================================
# MTF FILTER STATUS PANEL — always visible when token is present
# ==============================================================================
if ACCESS_TOKEN:
    st.markdown("---")
    st.subheader("📡 Multi-Timeframe Filter Status")

    # ── Always fetch fresh MTF data for the panel on every render ───────────────
    # Do NOT read stale last_tf_data — always run a fresh evaluation so the
    # panel reflects the current market state regardless of bot_active status.
    _vwap_panel = fetch_vwap_from_ohlc(ACCESS_TOKEN, "NSE_INDEX|Nifty 50") or 0.0
    _oi_panel   = st.session_state.get("oi_snapshot", {})
    _oi_dict    = {
        "oi_available": bool(_oi_panel),
        "oi_surge_ce":  False,
        "oi_surge_pe":  False,
        "pcr": (_oi_panel.get("pe_oi", 0) / _oi_panel.get("ce_oi", 1))
               if _oi_panel.get("ce_oi", 0) > 0 else 1.0,
    }
    _spot_panel  = nifty_spot or 24000.0
    _signal_p, _tf_p = evaluate_mtf_signal(
        ACCESS_TOKEN, _vwap_panel or _spot_panel, _oi_dict
    )
    st.session_state.last_tf_data = _tf_p
    tf_data = _tf_p

    # ── Manual refresh button ─────────────────────────────────────────────────
    if st.button("🔄 Refresh MTF Status", key="mtf_refresh"):
        st.session_state.last_tf_data = {}
        st.rerun()

    col_1h, col_15m, col_3m = st.columns(3)

    def _filter_rows(filters: dict, label: str):
        """Render a filter status card for one timeframe."""
        all_pass = all(filters.values()) if filters else False
        status_icon = "🟢" if all_pass else "🔴"
        st.markdown(f"**{status_icon} {label}**")
        for name, passed in filters.items():
            icon = "✅" if passed else "❌"
            st.caption(f"{icon} {name}")

    with col_1h:
        h1 = tf_data.get("1h", {})
        dir_map = {1: "📈 Bullish", -1: "📉 Bearish", 0: "➡️ Neutral"}
        direction = h1.get("direction", 0)
        st.markdown(f"**1H Trend (30M proxy) — {dir_map.get(direction, '—')}**")
        if h1.get("ok"):
            st.caption(f"EMA20: {h1.get('ema20', 0):.1f} | EMA50: {h1.get('ema50', 0):.1f} | ST: {'▲' if h1.get('st') == 1 else '▼'}")
            for name, passed in h1.get("filters", {}).items():
                icon = "✅" if passed else "❌"
                st.caption(f"{icon} {name}")
        else:
            # Show raw API error if any, so we can debug the endpoint
            _h1_errs = [e for e in st.session_state.get("api_errors", [])
                        if "historical-candle" in e.get("endpoint", "")]
            if _h1_errs:
                _e = _h1_errs[-1]
                st.caption(f"❌ API error {_e['status']}: {_e['body'][:120]}")
            else:
                st.caption("⏳ Fetching 30min history (60-day, multi-day endpoint)…")
                _debug_msg = tf_data.get("1h", {}).get("debug", "")
                bars_got   = len(st.session_state.get("_debug_1h_bars", []))
                st.caption(f"📊 {bars_got} bars received / 104 needed (30min × 104 = ~52hr)")
                if _debug_msg:
                    st.caption(f"ℹ️ {_debug_msg}")

    with col_15m:
        m15 = tf_data.get("15m", {})
        confirmed = m15.get("confirmed", False)
        h1_dir    = tf_data.get("1h", {}).get("direction", 0)
        st.markdown(f"**15M Momentum — {'🟢 Confirmed' if confirmed else '🔴 Not confirmed'}**")
        if m15.get("ok"):
            if m15.get("proxy"):
                st.caption("ℹ️ Using 1min proxy (< 5.5hr elapsed — 15M needs 330 bars)")
            ema9_v  = m15.get("ema9",  0)
            ema21_v = m15.get("ema21", 0)
            rsi_v   = m15.get("rsi",   0)
            close_v = m15.get("close", 0)
            vwap_v  = m15.get("vwap",  0)
            pcr_v   = m15.get("pcr",   0)
            ema_ok  = ema9_v > ema21_v if h1_dir == 1 else ema9_v < ema21_v
            rsi_ok  = (45 < rsi_v < 75) if h1_dir == 1 else (25 < rsi_v < 55)
            vwap_ok = close_v > vwap_v  if h1_dir == 1 else close_v < vwap_v
            oi_ok   = m15.get("oi_available", False)

            st.caption(
                f"EMA9: {ema9_v:.1f} {'>' if ema_ok else '<'} EMA21: {ema21_v:.1f} "
                f"{'✅' if ema_ok else '❌'}"
            )
            st.caption(
                f"RSI: {rsi_v:.1f} "
                f"({'need 45–75' if h1_dir == 1 else 'need 25–55'}) "
                f"{'✅' if rsi_ok else '❌'}"
            )
            st.caption(
                f"Close: {close_v:.1f} vs VWAP: {vwap_v:.1f} "
                f"{'✅' if vwap_ok else '❌'}"
            )
            if m15.get("oi_available"):
                st.caption(f"PCR: {pcr_v:.2f} ✅")
            else:
                st.caption("OI: bypassed (API unavailable) ✅")
        else:
            _h1_dir   = tf_data.get("1h",  {}).get("direction", 0)
            _15m_bars = tf_data.get("15m", {}).get("bars", 0)
            if _h1_dir == 0:
                st.caption("⏳ Waiting for 1H trend direction")
            elif _15m_bars == 0:
                _1min_cache = st.session_state.get("_candle_cache_1minute", {})
                _1min_bars  = len(_1min_cache.get("df", [])) if _1min_cache else 0
                st.caption(f"⚠️ 1min bars in cache: {_1min_bars} — check API error log if 0")
            else:
                st.caption(f"ℹ️ {_15m_bars} bars fetched — awaiting data")

    with col_3m:
        m3 = tf_data.get("3m", {})
        triggered = m3.get("triggered", False)
        h1_dir    = tf_data.get("1h", {}).get("direction", 0)
        st.markdown(f"**3M Trigger — {'🟢 FIRE' if triggered else '🔴 Waiting'}**")
        if m3.get("ok"):
            ema9_v   = m3.get("ema9",  0)
            ema21_v  = m3.get("ema21", 0)
            rsi_v    = m3.get("rsi",   0)
            adx_v    = m3.get("adx",   0)
            range_v    = m3.get("range",     0)
            rangesma_v = m3.get("range_sma", 0)
            rsi_prev_v = m3.get("rsi_prev", 0)

            ema_ok   = ema9_v > ema21_v if h1_dir == 1 else ema9_v < ema21_v
            rsi_ok   = (rsi_v > rsi_prev_v and 45 < rsi_v < 78) if h1_dir == 1 \
                       else (rsi_v < rsi_prev_v and 22 < rsi_v < 55)
            range_ok = range_v > rangesma_v
            adx_ok   = adx_v > 20

            st.caption(
                f"EMA9: {ema9_v:.1f} {'>' if ema_ok else '<'} "
                f"EMA21: {ema21_v:.1f} {'✅' if ema_ok else '❌'}"
            )
            st.caption(
                f"RSI: {rsi_v:.1f} (prev {rsi_prev_v:.1f}) "
                f"{'rising' if rsi_v > rsi_prev_v else 'falling'} "
                f"{'✅' if rsi_ok else '❌'}"
            )
            st.caption(
                f"Range: {range_v:.1f} vs SMA: {rangesma_v:.1f} "
                f"{'✅' if range_ok else '❌'}"
            )
            st.caption(
                f"ADX: {adx_v:.1f} (need > 20) "
                f"{'✅' if adx_ok else '❌'}"
            )
        else:
            st.caption("⏳ Waiting for 15M confirm first")

    # Overall signal readiness bar
    h1_ok  = tf_data.get("1h",  {}).get("direction", 0) != 0
    m15_ok = tf_data.get("15m", {}).get("confirmed", False)
    m3_ok  = tf_data.get("3m",  {}).get("triggered", False)
    layers_passed = sum([h1_ok, m15_ok, m3_ok])
    bar_colors = ["🟥", "🟧", "🟨", "🟩"]
    st.caption(
        f"Signal readiness: {bar_colors[layers_passed]} "
        f"{layers_passed}/3 layers passed "
        f"({'Entry armed — RR check next' if layers_passed == 3 else 'Monitoring…'})"
    )

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

    # ── VWAP scalar ───────────────────────────────────────────────────────────
    spot_ref  = nifty_spot or 24000.0
    live_vwap = fetch_vwap_from_ohlc(ACCESS_TOKEN, "NSE_INDEX|Nifty 50")
    if live_vwap and live_vwap > 0:
        vwap_value = live_vwap
    else:
        # Fallback: compute from 3M candles
        df3_vwap = fetch_historical_candles(ACCESS_TOKEN, "NSE_INDEX|Nifty 50", interval="1minute")
        if not df3_vwap.empty:
            tp         = (df3_vwap["high"] + df3_vwap["low"] + df3_vwap["close"]) / 3
            vwap_s     = (tp * df3_vwap["volume"]).cumsum() / df3_vwap["volume"].cumsum().replace(0, float("nan"))
            valid      = vwap_s.dropna()
            vwap_value = float(valid.iloc[-1]) if not valid.empty else spot_ref
        else:
            vwap_value = spot_ref

    # ── OI snapshot ───────────────────────────────────────────────────────────
    oi     = fetch_option_chain_oi(ACCESS_TOKEN, spot_ref)
    master = load_instrument_master()

    # ── MTF signal evaluation ─────────────────────────────────────────────────
    if st.session_state.current_position is None:
        signal, tf_data = evaluate_mtf_signal(ACCESS_TOKEN, vwap_value, oi)

        # Store for the filter status panel (rerenders even when no trade)
        st.session_state.last_tf_data = tf_data

        if signal in ("call", "put"):
            ot            = "CE" if signal == "call" else "PE"
            ikey, isymbol = resolve_atm_option_key(ACCESS_TOKEN, spot_ref, ot, master)
            entry_ltp     = fetch_ltp(ACCESS_TOKEN, ikey)

            if entry_ltp is None or entry_ltp <= 0:
                st.warning(f"⚠️ No valid LTP for {isymbol} — skipping.")
            else:
                # ATR-based SL/Target from 3M df
                df3 = tf_data.get("3m", {}).get("df", pd.DataFrame())
                atr_levels = compute_atr_sl_target(df3, entry_ltp, ATR_MULTIPLIER, RR_MIN)                              if not df3.empty else None
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
        else:
            # No signal — still update tf_data for dashboard
            st.session_state.last_tf_data = tf_data

    # ── Position management ───────────────────────────────────────────────────
    else:
        pos     = st.session_state.current_position
        pos_ltp = fetch_ltp(ACCESS_TOKEN, pos["key"])
        if pos_ltp is None:
            st.warning("⚠️ LTP fetch failed — retrying.")
        else:
            if pos_ltp > pos["highest_price"]:
                st.session_state.current_position["highest_price"] = pos_ltp
                # Use fresh 3M ATR for trailing SL
                df3_trail = fetch_historical_candles(ACCESS_TOKEN, "NSE_INDEX|Nifty 50", interval="1minute")
                atr_levels = compute_atr_sl_target(df3_trail, pos_ltp, ATR_MULTIPLIER, RR_MIN)                              if not df3_trail.empty else None
                new_sl = atr_levels[0] if atr_levels else pos_ltp * (1 - STOP_LOSS_PCT)
                if new_sl > pos["sl_price"]:
                    st.session_state.current_position["sl_price"] = new_sl

            hit_sl     = pos_ltp <= st.session_state.current_position["sl_price"]
            hit_target = pos_ltp >= pos["target_price"]
            if hit_sl or hit_target:
                reason = "TARGET HIT ✅" if hit_target else "STOP LOSS ❌"
                exit_position(ACCESS_TOKEN, pos, pos_ltp, reason, IS_PAPER, LOT_SIZE)
                st.rerun()

    time.sleep(3)
    st.rerun()

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
