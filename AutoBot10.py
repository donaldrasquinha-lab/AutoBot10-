"""
╔══════════════════════════════════════════════════════════════════╗
║       UPSTOX ALPHA TRADING ENGINE v2 — Live Options Matrix      ║
║  Tabbed layout · Plotly charts · IV Percentile · P&L Heatmap   ║
║  OI Change Tracking · PCR Trend · Interactive Strategy Builder  ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
from scipy.stats import norm
import requests
import urllib.parse
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import colorsys

# ═══════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════

st.set_page_config(
    page_title="Upstox Alpha Engine v2",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════
#  THEME CSS
# ═══════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600;700&display=swap');

:root {
    --bg-primary: #0a0e1a;
    --bg-card: #111827;
    --bg-card-alt: #1a2236;
    --border: #1e293b;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --accent-green: #22c55e;
    --accent-red: #ef4444;
    --accent-blue: #3b82f6;
    --accent-purple: #8b5cf6;
    --accent-amber: #f59e0b;
}

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif !important;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 16px;
    overflow: visible !important;
}
div[data-testid="stMetric"] label {
    color: #cbd5e1 !important;
    font-size: 10px !important;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    font-weight: 600 !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 800 !important;
    font-size: 20px !important;
    color: #ffffff !important;
    white-space: nowrap !important;
    overflow: visible !important;
    text-overflow: unset !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {
    color: #94a3b8 !important;
    font-weight: 600 !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricDelta"] svg {
    display: none;
}

/* Tabs */
div[data-testid="stTabs"] button[data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    letter-spacing: 0.3px;
}

/* Sentiment card */
.signal-card {
    border-radius: 12px; padding: 20px; margin: 12px 0;
    font-family: 'JetBrains Mono', monospace;
    border: 1px solid rgba(255,255,255,0.08);
}
.signal-label { font-size: 10px; letter-spacing: 2.5px; color: rgba(255,255,255,0.7); margin-bottom: 2px; text-transform: uppercase; }
.signal-value { font-size: 26px; font-weight: 700; color: #fff; }
.signal-sub { font-size: 12px; color: rgba(255,255,255,0.75); margin-top: 6px; line-height: 1.6; }

/* Strategy card */
.strat-card {
    background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px;
    padding: 18px; margin: 10px 0; color: #ffffff;
}
.strat-title {
    font-size: 14px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.5px;
    margin-bottom: 10px; padding-bottom: 8px; border-bottom: 1px solid var(--border);
}
.strat-leg {
    font-size: 14px; line-height: 2.0; font-family: 'JetBrains Mono', monospace;
    color: #f1f5f9 !important; font-weight: 600;
}
.strat-leg b { color: #ffffff; font-weight: 800; }
.strat-profit { font-size: 18px; font-weight: 800; margin-top: 12px; }

/* Badge pills */
.badge { display: inline-block; padding: 3px 10px; border-radius: 5px; font-size: 11px; font-weight: 600; font-family: 'JetBrains Mono', monospace; margin: 2px 3px; }
.badge-green { background: rgba(34,197,94,0.15); color: #4ade80; border: 1px solid rgba(34,197,94,0.3); }
.badge-red { background: rgba(239,68,68,0.15); color: #f87171; border: 1px solid rgba(239,68,68,0.3); }
.badge-blue { background: rgba(59,130,246,0.15); color: #60a5fa; border: 1px solid rgba(59,130,246,0.3); }
.badge-purple { background: rgba(139,92,246,0.15); color: #a78bfa; border: 1px solid rgba(139,92,246,0.3); }
.badge-amber { background: rgba(245,158,11,0.15); color: #fbbf24; border: 1px solid rgba(245,158,11,0.3); }

/* IV Gauge */
.iv-gauge-container { display: flex; align-items: center; gap: 12px; margin: 6px 0; }
.iv-gauge-bar { flex: 1; height: 10px; border-radius: 5px; background: linear-gradient(90deg, #22c55e 0%, #f59e0b 50%, #ef4444 100%); position: relative; }
.iv-gauge-marker { position: absolute; top: -4px; width: 4px; height: 18px; background: #fff; border-radius: 2px; transform: translateX(-50%); box-shadow: 0 0 6px rgba(255,255,255,0.5); }
.iv-pct-label { font-family: 'JetBrains Mono', monospace; font-size: 22px; font-weight: 700; min-width: 60px; text-align: right; }

footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
#  CONSTANTS & INDEX DEFINITIONS
# ═══════════════════════════════════════════════

UPSTOX_BASE = "https://api.upstox.com/v2"

INDICES = {
    "NIFTY 50":     {"key": "NSE_INDEX|Nifty 50",          "symbol": "NIFTY",       "diff": 50,  "lot": 25},
    "BANK NIFTY":   {"key": "NSE_INDEX|Nifty Bank",        "symbol": "BANKNIFTY",   "diff": 100, "lot": 15},
    "FINNIFTY":     {"key": "NSE_INDEX|Nifty Fin Service",  "symbol": "FINNIFTY",    "diff": 50,  "lot": 25},
    "MIDCAP NIFTY": {"key": "NSE_INDEX|NIFTY MID SELECT",  "symbol": "MIDCPNIFTY",  "diff": 25,  "lot": 50},
}

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(17,24,39,0.6)",
    font=dict(family="Inter, sans-serif", size=11, color="#94a3b8"),
    margin=dict(l=50, r=30, t=40, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
    xaxis=dict(gridcolor="rgba(30,41,59,0.6)", zeroline=False),
    yaxis=dict(gridcolor="rgba(30,41,59,0.6)", zeroline=False),
)

# ═══════════════════════════════════════════════
#  UPSTOX API CLIENT
# ═══════════════════════════════════════════════

class UpstoxClient:
    def __init__(self, token: str):
        clean = token.strip().replace("Bearer ", "")
        self.headers = {"Authorization": f"Bearer {clean}", "Accept": "application/json"}

    def _safe_json(self, r):
        ct = r.headers.get("Content-Type", "").lower()
        if "application/json" not in ct:
            raise ValueError(f"Non-JSON response (token expired?). Status {r.status_code}: {r.text[:200]}")
        body = r.json()
        if body.get("status") == "error":
            errs = body.get("errors", [])
            msg = errs[0].get("message", str(errs)) if errs else str(body)
            raise ValueError(f"Upstox API: {msg}")
        return body

    def get_spot_price(self, instrument_key: str) -> float:
        r = requests.get(f"{UPSTOX_BASE}/market-quote/ltp",
                         headers=self.headers, params={"instrument_key": instrument_key}, timeout=10)
        r.raise_for_status()
        data = self._safe_json(r).get("data", {})
        for k, v in data.items():
            if k == instrument_key or k.lower().replace(" ", "") == instrument_key.lower().replace(" ", ""):
                return float(v["last_price"])
        first = next(iter(data.values()), None)
        if first:
            return float(first["last_price"])
        raise ValueError(f"Symbol not found: {instrument_key}")

    def get_expiries(self, instrument_key: str) -> list:
        r = requests.get(f"{UPSTOX_BASE}/option/contract",
                         headers=self.headers, params={"instrument_key": instrument_key}, timeout=10)
        r.raise_for_status()
        data = self._safe_json(r).get("data", [])
        expiries = sorted(set(
            str(c.get("expiry", ""))[:10] for c in data
        ))
        return [e for e in expiries if e and e != "None"]

    def get_option_chain(self, instrument_key: str, expiry_date: str) -> list:
        r = requests.get(f"{UPSTOX_BASE}/option/chain",
                         headers=self.headers,
                         params={"instrument_key": instrument_key, "expiry_date": expiry_date}, timeout=10)
        r.raise_for_status()
        return self._safe_json(r).get("data", [])

    def get_option_contracts(self, instrument_key: str, expiry_date: str) -> list:
        """
        Fetch ALL option contracts for an instrument+expiry via /option/contract.
        Returns list of contract dicts with instrument_key, strike_price, option_type etc.
        This gives the FULL strike range, unlike /option/chain which is limited.
        """
        r = requests.get(f"{UPSTOX_BASE}/option/contract",
                         headers=self.headers,
                         params={"instrument_key": instrument_key}, timeout=10)
        r.raise_for_status()
        all_contracts = self._safe_json(r).get("data", [])
        # Filter to the target expiry
        matched = []
        for c in all_contracts:
            exp = str(c.get("expiry", ""))[:10]
            if exp == expiry_date:
                matched.append(c)
        return matched

    def get_ltp_batch(self, instrument_keys: list) -> dict:
        """
        Fetch LTP for up to 50 instruments in one call via /market-quote/ltp.
        Returns {instrument_key: ltp} dict.
        """
        result = {}
        # API allows comma-separated keys, max ~50 per call
        for i in range(0, len(instrument_keys), 50):
            batch = instrument_keys[i:i+50]
            keys_param = ",".join(batch)
            r = requests.get(f"{UPSTOX_BASE}/market-quote/ltp",
                             headers=self.headers,
                             params={"instrument_key": keys_param}, timeout=10)
            r.raise_for_status()
            data = self._safe_json(r).get("data", {})
            for k, v in data.items():
                result[k] = float(v.get("last_price", 0) or 0)
        return result

    def get_historical_candles(self, instrument_key: str, interval="day", days=45) -> pd.DataFrame:
        to_d = datetime.now().strftime("%Y-%m-%d")
        from_d = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        enc = urllib.parse.quote(instrument_key, safe="")
        r = requests.get(f"{UPSTOX_BASE}/historical-candle/{enc}/{interval}/{to_d}/{from_d}",
                         headers=self.headers, timeout=10)
        r.raise_for_status()
        candles = self._safe_json(r).get("data", {}).get("candles", [])
        if not candles:
            return pd.DataFrame()
        rows = [{"ts": c[0], "open": float(c[1]), "high": float(c[2]),
                 "low": float(c[3]), "close": float(c[4]),
                 "volume": int(c[5]) if len(c) > 5 else 0} for c in candles if len(c) >= 5]
        return pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)


# ═══════════════════════════════════════════════
#  ANALYTICS FUNCTIONS
# ═══════════════════════════════════════════════

def compute_adx(df: pd.DataFrame, period=14):
    if df.empty or len(df) < period * 2 + 2:
        return None
    d = df.copy()
    d["ph"], d["pl"], d["pc"] = d["high"].shift(1), d["low"].shift(1), d["close"].shift(1)
    d["tr"] = d.apply(lambda r: max(r["high"]-r["low"],
        abs(r["high"]-r["pc"]) if pd.notna(r["pc"]) else 0,
        abs(r["low"]-r["pc"]) if pd.notna(r["pc"]) else 0), axis=1)
    d["+dm"] = d.apply(lambda r: max(r["high"]-r["ph"],0)
        if pd.notna(r["ph"]) and (r["high"]-r["ph"])>(r["pl"]-r["low"]) else 0, axis=1)
    d["-dm"] = d.apply(lambda r: max(r["pl"]-r["low"],0)
        if pd.notna(r["pl"]) and (r["pl"]-r["low"])>(r["high"]-r["ph"]) else 0, axis=1)
    d = d.iloc[1:].reset_index(drop=True)
    tr_s = [d["tr"].iloc[:period].sum()]
    pd_s = [d["+dm"].iloc[:period].sum()]
    nd_s = [d["-dm"].iloc[:period].sum()]
    for i in range(period, len(d)):
        tr_s.append(tr_s[-1] - tr_s[-1]/period + d["tr"].iloc[i])
        pd_s.append(pd_s[-1] - pd_s[-1]/period + d["+dm"].iloc[i])
        nd_s.append(nd_s[-1] - nd_s[-1]/period + d["-dm"].iloc[i])
    pdi_l, ndi_l, dx_l = [], [], []
    for i in range(len(tr_s)):
        pdi = pd_s[i]/tr_s[i]*100 if tr_s[i]>0 else 0
        ndi = nd_s[i]/tr_s[i]*100 if tr_s[i]>0 else 0
        pdi_l.append(pdi); ndi_l.append(ndi)
        dx_l.append(abs(pdi-ndi)/(pdi+ndi)*100 if (pdi+ndi)>0 else 0)
    if len(dx_l) < period:
        return None
    adx_l = [sum(dx_l[:period])/period]
    for i in range(period, len(dx_l)):
        adx_l.append((adx_l[-1]*(period-1)+dx_l[i])/period)
    return {"adx": round(adx_l[-1],2), "plus_di": round(pdi_l[-1],2), "minus_di": round(ndi_l[-1],2)}


def bs_greeks(S, K, T, r, sigma, opt="CE"):
    if T <= 0 or sigma <= 0:
        return {"price":0,"delta":0,"gamma":0,"theta":0,"vega":0}
    d1 = (np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    gamma = norm.pdf(d1)/(S*sigma*np.sqrt(T))
    vega = S*norm.pdf(d1)*np.sqrt(T)/100
    if opt == "CE":
        price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = (-(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2))/365
    else:
        price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
        delta = norm.cdf(d1)-1
        theta = (-(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T)) + r*K*np.exp(-r*T)*norm.cdf(-d2))/365
    return {"price":round(price,2),"delta":round(delta,3),"gamma":round(gamma,5),"theta":round(theta,2),"vega":round(vega,2)}


def implied_vol(market_price, S, K, T, r, opt="CE", max_iter=50, tol=1e-5):
    """
    Compute implied volatility from a market price using Newton-Raphson.
    Returns IV as a decimal (e.g. 0.15 for 15%), or None if it can't converge.
    """
    if market_price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return None

    # Intrinsic value check — price must exceed intrinsic
    if opt == "CE":
        intrinsic = max(0, S - K * np.exp(-r * T))
    else:
        intrinsic = max(0, K * np.exp(-r * T) - S)

    if market_price < intrinsic * 0.95:
        return None  # Price below intrinsic — bad data

    # Initial guess from Brenner-Subrahmanyam approximation
    sigma = np.sqrt(2 * np.pi / T) * (market_price / S)
    sigma = max(0.01, min(sigma, 5.0))  # Clamp to sane range

    for _ in range(max_iter):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if opt == "CE":
            bs_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            bs_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        vega = S * norm.pdf(d1) * np.sqrt(T)  # NOT divided by 100 here

        diff = bs_price - market_price

        if abs(diff) < tol:
            if 0.01 <= sigma <= 3.0:
                return sigma
            return None

        if vega < 1e-10:
            break  # Vega too small to move sigma meaningfully

        sigma -= diff / vega
        sigma = max(0.005, min(sigma, 5.0))  # Keep bounded

    return None  # Didn't converge


def compute_max_pain(df_chain: pd.DataFrame) -> float:
    strikes = df_chain["Strike"].astype(float).values
    ce_oi = df_chain["CE OI"].astype(float).values
    pe_oi = df_chain["PE OI"].astype(float).values
    pain = {}
    for s in strikes:
        total = 0.0
        for k in range(len(strikes)):
            total += max(0.0, s - strikes[k]) * pe_oi[k]
            total += max(0.0, strikes[k] - s) * ce_oi[k]
        pain[s] = total
    return min(pain, key=pain.get) if pain else 0.0


def compute_iv_percentile(candles_df: pd.DataFrame, current_iv: float, window=30) -> float:
    """IV Percentile: % of last N days where realized vol was below current IV."""
    if candles_df.empty or len(candles_df) < window + 2:
        return 50.0  # default mid
    closes = candles_df["close"].values
    log_ret = np.diff(np.log(closes))
    if len(log_ret) < window:
        return 50.0
    # Rolling realized vol (annualised)
    rv_series = []
    for i in range(len(log_ret) - window + 1):
        chunk = log_ret[i:i+window]
        rv = np.std(chunk) * np.sqrt(252) * 100
        rv_series.append(rv)
    if not rv_series:
        return 50.0
    count_below = sum(1 for rv in rv_series if rv < current_iv * 100)
    return round(count_below / len(rv_series) * 100, 1)


def compute_pnl_heatmap(strategy_legs, lot_size, spot_price, diff, iv_used, rfr):
    """
    Compute strategy P&L across a grid of spot prices and days-to-expiry.
    Each leg: {"strike": K, "type": "CE"/"PE", "action": "BUY"/"SELL", "premium": ltp}
    Uses fewer spot steps for readability (strike-aligned grid).
    """
    # Create a strike-aligned grid: every 'diff' points, ±6 strikes from ATM
    atm = round(spot_price / diff) * diff
    spot_range = np.arange(atm - 6*diff, atm + 6*diff + 1, diff)
    dte_range = np.array([0, 1, 2, 3, 5, 7, 10, 14])

    pnl_matrix = np.zeros((len(dte_range), len(spot_range)))

    for i, dte in enumerate(dte_range):
        for j, s in enumerate(spot_range):
            total = 0.0
            for leg in strategy_legs:
                K = leg["strike"]
                prem = leg["premium"]
                if dte == 0:
                    if leg["type"] == "CE":
                        val = max(0, s - K)
                    else:
                        val = max(0, K - s)
                else:
                    T = dte / 365
                    sigma = iv_used if iv_used > 0 else 0.15
                    val = bs_greeks(s, K, T, rfr, sigma, leg["type"])["price"]

                if leg["action"] == "BUY":
                    total += (val - prem)
                else:
                    total += (prem - val)
            pnl_matrix[i, j] = total * lot_size
    return spot_range, dte_range, pnl_matrix


# ═══════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════

st.sidebar.markdown("### 🔐 Authentication")
api_token = st.sidebar.text_input("Access Token", type="password", value="",
                                   help="Paste your Upstox OAuth access_token. Expires midnight IST daily.")

st.sidebar.markdown("---")
selected_index_name = st.sidebar.selectbox("🎯 Underlying Index", list(INDICES.keys()))
index_meta = INDICES[selected_index_name]

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Engine Parameters")
iv_override = st.sidebar.slider("IV Fallback (used when live IV unavailable) %", 5.0, 80.0, 15.0, 0.5) / 100
strike_depth = st.sidebar.slider("Strike Depth Around ATM", 3, 15, 7)

# ── Risk-Free Rate: fetch live India 10Y G-Sec yield ──
def fetch_risk_free_rate():
    """
    Fetch India 10Y government bond yield as risk-free rate proxy.
    Uses RBI/market data. Falls back to 7.0% if unavailable.
    """
    try:
        # Try fetching from a public API
        r = requests.get(
            "https://api.worldbank.org/v2/country/IND/indicator/FR.INR.RINR?date=2024:2026&format=json",
            timeout=5
        )
        if r.status_code == 200:
            data = r.json()
            if len(data) > 1 and data[1]:
                for entry in data[1]:
                    if entry.get("value") is not None:
                        return round(float(entry["value"]), 2)
    except Exception:
        pass

    # Fallback: use India 91-day T-bill rate approximation
    # As of mid-2025, India 10Y is ~7.0-7.1%
    return 7.0

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_cached_rfr():
    return fetch_risk_free_rate()

live_rfr = get_cached_rfr()
rfr_is_live = live_rfr != 7.0

risk_free_rate = st.sidebar.number_input(
    f"Risk-Free Rate % {'(live)' if rfr_is_live else '(default)'}",
    min_value=0.0, max_value=15.0, value=float(live_rfr), step=0.1,
    help="India 10Y G-Sec yield. Auto-fetched; editable if you want to override."
) / 100

st.sidebar.markdown("---")
auto_refresh = st.sidebar.checkbox("🔄 Auto-Refresh", value=False)
refresh_interval = 30
if auto_refresh:
    refresh_interval = st.sidebar.slider("Interval (sec)", 10, 120, 30, 5)

# ═══════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════

st.markdown(f"""
<div style="display:flex; align-items:center; gap:12px; margin-bottom:4px;">
    <span style="font-size:28px;">⚡</span>
    <span style="font-family:'Inter',sans-serif; font-size:22px; font-weight:700;">
        Upstox Alpha Engine
    </span>
    <span style="font-size:12px; color:#64748b; background:rgba(59,130,246,0.12); padding:2px 8px;
                 border-radius:4px; font-weight:600; margin-left:4px;">v2.0</span>
    <span style="font-size:13px; color:#94a3b8; margin-left:auto; font-family:'JetBrains Mono',monospace;">
        {selected_index_name}
    </span>
</div>
""", unsafe_allow_html=True)

# ── Help Guide ──
HELP_CONTENT = """
## 📖 Upstox Alpha Engine — User Guide

---

### 🔐 Getting Started

**Step 1: Get Your Access Token**
1. Log in to [Upstox Developer Console](https://api.upstox.com)
2. Complete the OAuth2 authentication flow
3. Copy the `access_token` from the redirect URL
4. Paste it in the sidebar under **Access Token**

> ⚠️ Tokens expire at **midnight IST daily**. You'll need a fresh one each trading day.

**Step 2: Select Your Index**
Choose from NIFTY 50, BANK NIFTY, FINNIFTY, or MIDCAP NIFTY in the sidebar.

**Step 3: Pick an Expiry**
The nearest expiry is auto-selected. Switch to weekly/monthly as needed.

---

### 📊 Top KPI Row — What Each Metric Means

| Metric | What It Tells You |
|--------|-------------------|
| **Spot** | Current index price from Upstox. This is where the market is right now. |
| **ATM** | At-The-Money strike — the option strike closest to spot. This is your anchor point. |
| **PCR** | Put-Call Ratio (by OI). **> 1.0** = more puts written = bullish bias. **< 1.0** = more calls written = bearish bias. |
| **Max Pain** | The strike where option sellers lose the least money. Markets tend to gravitate here by expiry. |
| **VWAS** | Volume-Weighted Average Strike — where actual trading volume concentrates. If VWAS > Spot, money is flowing to higher strikes (bullish). |
| **ADX (14)** | Trend strength (not direction). **< 20** = no trend, range-bound. **20-25** = trend emerging. **> 25** = strong trend. |
| **DTE** | Days to expiry. Theta decay accelerates sharply below 5 DTE. |

---

### 🎯 IV Percentile Gauge

The horizontal bar shows where current implied volatility sits relative to its recent history.

- **Green zone (0-30%)**: IV is LOW → options are cheap → favor buying strategies (long straddle, debit spreads)
- **Amber zone (30-70%)**: IV is normal → no strong edge either way
- **Red zone (70-100%)**: IV is HIGH → options are expensive → favor selling strategies (iron condor, short straddle)

**LIVE** badge = IV computed from actual market prices via Newton-Raphson solver.
**MANUAL** badge = using the sidebar slider (market closed or no data).

---

### 🟢🔴 Sentiment Card

Combines PCR analysis with OI distribution:

| Signal | PCR Range | Meaning |
|--------|-----------|---------|
| **STRONG BULLISH** | ≥ 1.25 | Heavy put writing = writers believe market won't fall. Strong support below. |
| **MILDLY BULLISH** | 1.05 – 1.25 | Slight bullish tilt. |
| **NEUTRAL** | 0.95 – 1.05 | No clear directional bias. |
| **MILDLY BEARISH** | 0.75 – 0.95 | Slight bearish tilt. |
| **STRONG BEARISH** | ≤ 0.75 | Heavy call writing = writers believe market won't rise. Resistance above. |

**Support** = strike with highest Put OI (floor the market respects).
**Resistance** = strike with highest Call OI (ceiling the market faces).

---

### 📊 Dashboard Tab

**Statistical Price Forecast (Bell Curve)**
Uses the normal distribution to predict where the index will settle by expiry.
- **1σ zone (68.2%)**: The index has a 68% chance of staying within this range.
- **2σ zone (95.4%)**: 95% probability zone.
- Vertical lines mark current spot, max pain, support, and resistance.

**PCR Trend**: If auto-refresh is on, tracks how PCR shifts during the session. A rising PCR = growing bullish sentiment.

**OI Change Tables**: Shows which strikes saw the most new position build-up today.
- Fresh Call OI build-up at higher strikes = resistance strengthening = bearish.
- Fresh Put OI build-up at lower strikes = support strengthening = bullish.

---

### 📋 Options Chain Tab

The full chain with live Greeks, OI, volume, and IV per strike.

**How to read it:**
- **Yellow row** = ATM strike
- **Green tint** = in-the-money calls (below spot)
- **Red tint** = in-the-money puts (above spot)
- **Blue intensity on OI** = brighter = higher OI = stronger support/resistance at that strike
- **Green OI Change** = new positions being built (bullish for puts, bearish for calls)
- **Red OI Change** = positions being unwound

**Key Greeks:**
- **Delta**: How much the option moves per ₹1 move in spot. ATM ≈ 0.5.
- **Theta**: How much you lose per day from time decay. Higher near expiry.
- **Gamma**: How fast delta changes. Highest at ATM, near expiry.
- **Vega**: Sensitivity to IV changes. Higher for longer-dated options.

---

### 🛡️ Strategy Builder Tab

**Engine Recommendation**: Auto-suggests a strategy based on:
- **ADX > 25** → Trending market → use directional strategies (debit spreads)
- **IV Pct > 70%** → Expensive options → sell premium (iron butterfly, short straddle)
- **IV Pct < 30%** → Cheap options → buy premium (long straddle)
- **Otherwise** → Range-bound → iron condor

**Available Strategies:**

| Strategy | When to Use | Risk Profile |
|----------|-------------|--------------|
| **Iron Condor** | Range-bound, moderate IV | Defined risk, defined reward. Profits if index stays between sell strikes. |
| **Short Straddle** | Expecting minimal movement, high IV | Unlimited risk! Maximum premium collection but dangerous if market moves big. |
| **Iron Butterfly** | Expecting pinning at ATM, high IV | Defined risk. Like a tighter iron condor centered at ATM. |
| **Bull Put Spread** | Mildly bullish | Defined risk. Profits if index stays above sell strike. |
| **Bear Call Spread** | Mildly bearish | Defined risk. Profits if index stays below sell strike. |

**P&L Heatmap**: Each cell shows your exact ₹ profit or loss based on:
- **Columns** = where the index might be at various levels
- **Rows** = how many days remain until expiry
- **Green cells** = you make money. **Red cells** = you lose money.
- The **"You Are Here"** marker shows where spot currently sits.

**Payoff at Expiry**: The classic hockey-stick diagram showing your final P&L.
- **Green zone** = profit area. **Red zone** = loss area.
- Annotated arrows point to max profit and max loss levels.

**Summary Box**: Quick-glance numbers — what you collect, best/worst case, risk:reward ratio.

---

### 📈 Charts Tab

**OI Distribution**: Bar chart of Call vs Put open interest per strike.
- Tallest Call OI bar = strongest resistance.
- Tallest Put OI bar = strongest support.
- Max Pain marker shows the "magnet" strike.

**OI Change**: Shows where NEW positions were opened today.
- Positive bars = build-up (new positions). Negative = unwinding.
- Interpreting: Call build-up at higher strikes = bearish. Put build-up at lower strikes = bullish.

**Delta Skew**: Shows how delta varies across strikes.
- Steep skew = market pricing in directional risk.
- Flat = neutral expectations.

**IV Smile**: Implied volatility across strikes.
- U-shape ("smile") = normal. Skewed = market expects movement in one direction.
- Higher IV at OTM puts = market pricing in downside risk (fear).

**Cumulative OI**: Running total of Call vs Put OI from lowest to highest strike.
- Where Put cumulative OI exceeds Call = bullish zone.
- Where Call exceeds Put = bearish zone.

---

### ⚙️ Engine Parameters

| Parameter | What It Controls |
|-----------|-----------------|
| **IV Fallback** | Only used when live IV can't be computed (market closed). During market hours, all IV is live. |
| **Risk-Free Rate** | Auto-fetched from World Bank (India rate). Editable. Feeds into all Black-Scholes calculations. |
| **Strike Depth** | How many strikes above/below ATM to show in the chain. Higher = more strikes but slower load. |
| **Auto-Refresh** | When enabled, the entire dashboard reloads every N seconds with fresh data. |

---

### 💡 Pro Tips

1. **Best used during market hours** (9:15 AM – 3:30 PM IST). After hours, LTPs freeze and some features show theoretical values.
2. **Watch PCR + OI Change together**: A rising PCR with heavy put OI build-up at support = very bullish setup.
3. **Max Pain is a magnet, not a guarantee**: It works best 2-3 days before expiry.
4. **IV Percentile > 70% on expiry week** = premium selling sweet spot. Time decay is on your side.
5. **ADX below 20 + high IV** = the ideal iron condor/butterfly setup. No trend + expensive options = sell premium.
6. **VWAS diverging from spot** = smart money positioning differently from current price. Follow VWAS for directional hints.

---

*Built for Indian equity derivatives on NSE. Data from Upstox V2 API. Greeks via Black-Scholes. IV via Newton-Raphson.*
"""

# Help button in sidebar — uses expander, no session state, no pop-up on refresh
st.sidebar.markdown("---")
with st.sidebar.expander("📖 Help & User Guide"):
    st.markdown(HELP_CONTENT)

# ═══════════════════════════════════════════════
#  MAIN ENGINE
# ═══════════════════════════════════════════════

if not api_token:
    st.info("💡 Enter your **Upstox Access Token** in the sidebar to begin. "
            "Get it via the Upstox OAuth2 flow — it resets at midnight IST.")
    st.stop()

try:
    client = UpstoxClient(token=api_token)
    diff = index_meta["diff"]
    lot_size = index_meta["lot"]

    spot_price = client.get_spot_price(index_meta["key"])
    expiries = client.get_expiries(index_meta["key"])

    if not expiries:
        st.error("No active derivative contracts found.")
        st.stop()

    selected_expiry = st.sidebar.selectbox("📅 Expiry", expiries, index=0)

    expiry_dt = datetime.strptime(selected_expiry, "%Y-%m-%d").replace(hour=15, minute=30)
    tte_years = max((expiry_dt - datetime.now()).total_seconds() / (86400*365), 0.0001)
    tte_days = max(tte_years * 365, 0.01)

    with st.spinner("Loading chain, contracts & candles..."):
        candles_df = client.get_historical_candles(index_meta["key"], "day", 60)
        adx_metrics = compute_adx(candles_df)
        chain_raw = client.get_option_chain(index_meta["key"], selected_expiry)
        # Fetch ALL contracts for this expiry — gives full strike range with instrument keys
        all_contracts = client.get_option_contracts(index_meta["key"], selected_expiry)

    if not chain_raw and not all_contracts:
        st.warning("Empty chain for this expiry.")
        st.stop()

    atm_strike = round(spot_price / diff) * diff

    # ── Build COMPLETE instrument key map from /option/contract ──
    # This covers ALL strikes for the expiry, not just the limited /option/chain window
    strike_inst_map = {}  # (strike, "CE"/"PE") -> instrument_key
    for c in all_contracts:
        sp = float(c.get("strike_price", 0))
        opt_type = str(c.get("option_type", "")).upper()
        inst_key = c.get("instrument_key", "")
        if inst_key and opt_type in ("CE", "PE"):
            strike_inst_map[(sp, opt_type)] = inst_key

    # Also populate from chain_raw in case /option/contract used different field names
    for sd in chain_raw:
        sp = float(sd.get("strike_price", 0))
        for side, label in [("call_options", "CE"), ("put_options", "PE")]:
            opt = sd.get(side, {}) or {}
            ik = opt.get("instrument_key", "")
            if ik and (sp, label) not in strike_inst_map:
                strike_inst_map[(sp, label)] = ik

    # ── Determine which strikes we need LTPs for ──
    # Display range + strategy strikes
    display_strikes = set()
    for sd in chain_raw:
        sp = float(sd.get("strike_price", 0))
        if abs(sp - atm_strike) <= strike_depth * diff:
            display_strikes.add(sp)

    # σ bounds for pre-fetch (uses slider as estimate; real calc happens after live IV is known)
    # Use a generous range (max of slider IV and 30%) to ensure we fetch enough keys
    _pre_iv = max(iv_override, 0.30)
    _std_tmp = spot_price * _pre_iv * np.sqrt(tte_years)
    _lo1_tmp = spot_price - _std_tmp
    _hi1_tmp = spot_price + _std_tmp
    _ic_sell_put = round(_lo1_tmp / diff) * diff
    _ic_sell_call = round(_hi1_tmp / diff) * diff
    strategy_strikes = {atm_strike, _ic_sell_put, _ic_sell_call,
                        _ic_sell_put - diff, _ic_sell_call + diff,
                        _ic_sell_put - 2*diff, _ic_sell_call + 2*diff}  # Extra buffer

    all_needed_strikes = display_strikes | strategy_strikes

    # ── Batch-fetch LTPs from Upstox for ALL needed strikes ──
    keys_for_ltp = []
    key_to_strike_opt = {}
    for strike in all_needed_strikes:
        for opt_type in ["CE", "PE"]:
            if (strike, opt_type) in strike_inst_map:
                ik = strike_inst_map[(strike, opt_type)]
                keys_for_ltp.append(ik)
                key_to_strike_opt[ik] = (strike, opt_type)

    live_ltp_map = {}  # (strike, "CE"/"PE") -> ltp
    if keys_for_ltp:
        try:
            with st.spinner(f"Fetching live prices for {len(keys_for_ltp)} contracts..."):
                fetched = client.get_ltp_batch(keys_for_ltp)
            for ik, price in fetched.items():
                if ik in key_to_strike_opt:
                    live_ltp_map[key_to_strike_opt[ik]] = price
        except Exception as e:
            st.warning(f"⚠️ Could not fetch live LTPs: {e}. Using chain data + theoretical prices.")

    # ── Build chain records (filtered by strike_depth for display) ──
    records = []

    for sd in chain_raw:
        sp = float(sd.get("strike_price", 0))
        if abs(sp - atm_strike) > strike_depth * diff:
            continue
        ce = sd.get("call_options", {}) or {}
        pe = sd.get("put_options", {}) or {}
        ce_md = ce.get("market_data", {}) or {}
        pe_md = pe.get("market_data", {}) or {}

        ce_oi = int(float(ce_md.get("oi", 0) or 0))
        pe_oi = int(float(pe_md.get("oi", 0) or 0))
        # Use live LTP if available, else chain LTP
        ce_ltp = live_ltp_map.get((sp, "CE"), 0.0) or float(ce_md.get("ltp", 0) or 0)
        pe_ltp = live_ltp_map.get((sp, "PE"), 0.0) or float(pe_md.get("ltp", 0) or 0)
        ce_vol = int(float(ce_md.get("volume", 0) or 0))
        pe_vol = int(float(pe_md.get("volume", 0) or 0))
        ce_prev_oi = int(float(ce_md.get("prev_oi", ce_oi) or ce_oi))
        pe_prev_oi = int(float(pe_md.get("prev_oi", pe_oi) or pe_oi))
        ce_iv_raw = float(ce_md.get("iv", 0) or 0)
        pe_iv_raw = float(pe_md.get("iv", 0) or 0)

        # ── IV Resolution: Live Market Price → Chain IV → Manual Override ──
        # Tier 1: Compute IV from live market LTP using Newton-Raphson
        ce_sigma = None
        pe_sigma = None

        if ce_ltp > 0:
            ce_sigma = implied_vol(ce_ltp, spot_price, sp, tte_years, risk_free_rate, "CE")
        if pe_ltp > 0:
            pe_sigma = implied_vol(pe_ltp, spot_price, sp, tte_years, risk_free_rate, "PE")

        # Tier 2: Chain-reported IV (if API provides it)
        if ce_sigma is None and ce_iv_raw > 0:
            ce_sigma = ce_iv_raw / 100
        if pe_sigma is None and pe_iv_raw > 0:
            pe_sigma = pe_iv_raw / 100

        # Tier 3: Manual slider fallback (will be replaced by iv_for_sigma after ATM IV is known)
        if ce_sigma is None:
            ce_sigma = iv_override  # Temporary; best we can do before ATM IV is computed
        if pe_sigma is None:
            pe_sigma = iv_override

        # Track source for display
        ce_iv_source = "live" if (ce_ltp > 0 and implied_vol(ce_ltp, spot_price, sp, tte_years, risk_free_rate, "CE") is not None) else ("chain" if ce_iv_raw > 0 else "manual")
        pe_iv_source = "live" if (pe_ltp > 0 and implied_vol(pe_ltp, spot_price, sp, tte_years, risk_free_rate, "PE") is not None) else ("chain" if pe_iv_raw > 0 else "manual")

        ce_g = bs_greeks(spot_price, sp, tte_years, risk_free_rate, ce_sigma, "CE")
        pe_g = bs_greeks(spot_price, sp, tte_years, risk_free_rate, pe_sigma, "PE")

        records.append({
            "CE OI": ce_oi, "CE OI Chg": ce_oi - ce_prev_oi, "CE Vol": ce_vol,
            "CE IV": round(ce_sigma*100, 1),
            "CE Delta": ce_g["delta"], "CE Gamma": ce_g["gamma"],
            "CE Theta": ce_g["theta"], "CE Vega": ce_g["vega"],
            "CE LTP": ce_ltp,
            "Strike": sp,
            "PE LTP": pe_ltp,
            "PE Vega": pe_g["vega"], "PE Theta": pe_g["theta"],
            "PE Gamma": pe_g["gamma"], "PE Delta": pe_g["delta"],
            "PE IV": round(pe_sigma*100, 1),
            "PE Vol": pe_vol, "PE OI Chg": pe_oi - pe_prev_oi, "PE OI": pe_oi,
        })

    df = pd.DataFrame(records).sort_values("Strike").reset_index(drop=True)

    # Force numeric types on all data columns
    numeric_cols = [c for c in df.columns if c != "Strike"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["Strike"] = df["Strike"].astype(float)

    # ── Derived metrics ──
    total_ce_oi = df["CE OI"].sum()
    total_pe_oi = df["PE OI"].sum()
    pcr = round(total_pe_oi / total_ce_oi, 2) if total_ce_oi > 0 else 0
    max_pain = compute_max_pain(df)
    resistance = df.loc[df["CE OI"].idxmax(), "Strike"] if not df.empty else atm_strike
    support = df.loc[df["PE OI"].idxmax(), "Strike"] if not df.empty else atm_strike

    # IV Percentile — use live-computed ATM IV
    atm_row = df[df["Strike"] == atm_strike]
    atm_iv = float(atm_row.iloc[0]["CE IV"]) if not atm_row.empty else iv_override * 100
    atm_iv_is_live = atm_iv != iv_override * 100  # True if IV came from market, not slider
    iv_pct = compute_iv_percentile(candles_df, atm_iv / 100, window=20)

    # Use live ATM IV for σ calculations when available
    iv_for_sigma = atm_iv / 100 if atm_iv_is_live else iv_override

    # ── Recompute Greeks for strikes that used manual IV fallback ──
    # Now that iv_for_sigma is known, upgrade any strike stuck on iv_override
    if atm_iv_is_live:
        for idx, row in df.iterrows():
            sp = row["Strike"]
            ce_iv_val = row["CE IV"]
            pe_iv_val = row["PE IV"]
            recompute = False

            # If CE IV equals the slider value, it was a fallback — use live ATM IV instead
            if abs(ce_iv_val - iv_override * 100) < 0.01:
                ce_sigma_new = iv_for_sigma
                df.at[idx, "CE IV"] = round(iv_for_sigma * 100, 1)
                recompute = True
            else:
                ce_sigma_new = ce_iv_val / 100

            if abs(pe_iv_val - iv_override * 100) < 0.01:
                pe_sigma_new = iv_for_sigma
                df.at[idx, "PE IV"] = round(iv_for_sigma * 100, 1)
            else:
                pe_sigma_new = pe_iv_val / 100

            if recompute or abs(pe_iv_val - iv_override * 100) < 0.01:
                ce_g = bs_greeks(spot_price, sp, tte_years, risk_free_rate, ce_sigma_new, "CE")
                pe_g = bs_greeks(spot_price, sp, tte_years, risk_free_rate, pe_sigma_new, "PE")
                df.at[idx, "CE Delta"] = ce_g["delta"]
                df.at[idx, "CE Gamma"] = ce_g["gamma"]
                df.at[idx, "CE Theta"] = ce_g["theta"]
                df.at[idx, "CE Vega"] = ce_g["vega"]
                df.at[idx, "PE Delta"] = pe_g["delta"]
                df.at[idx, "PE Gamma"] = pe_g["gamma"]
                df.at[idx, "PE Theta"] = pe_g["theta"]
                df.at[idx, "PE Vega"] = pe_g["vega"]

    # Volume-Weighted Average Strike
    total_vol = df["CE Vol"].sum() + df["PE Vol"].sum()
    vwas = ((df["Strike"] * (df["CE Vol"] + df["PE Vol"])).sum()) / total_vol if total_vol > 0 else atm_strike

    # PCR history in session state for sparkline
    if "pcr_history" not in st.session_state:
        st.session_state.pcr_history = []
    st.session_state.pcr_history.append({"time": datetime.now().strftime("%H:%M:%S"), "pcr": pcr})
    if len(st.session_state.pcr_history) > 60:
        st.session_state.pcr_history = st.session_state.pcr_history[-60:]

    # Sentiment
    if pcr >= 1.25:
        sentiment, sent_color = "STRONG BULLISH", "#15803d"
        card_bg = "linear-gradient(135deg, #15803d, #166534)"
    elif pcr > 1.05:
        sentiment, sent_color = "MILDLY BULLISH", "#22c55e"
        card_bg = "linear-gradient(135deg, #065f46, #064e3b)"
    elif pcr <= 0.75:
        sentiment, sent_color = "STRONG BEARISH", "#dc2626"
        card_bg = "linear-gradient(135deg, #991b1b, #7f1d1d)"
    elif pcr < 0.95:
        sentiment, sent_color = "MILDLY BEARISH", "#ef4444"
        card_bg = "linear-gradient(135deg, #b91c1c, #991b1b)"
    else:
        sentiment, sent_color = "NEUTRAL", "#64748b"
        card_bg = "linear-gradient(135deg, #475569, #334155)"

    # σ bounds — uses live ATM IV when available
    std_price = spot_price * iv_for_sigma * np.sqrt(tte_years)
    lo1 = spot_price - std_price
    hi1 = spot_price + std_price
    lo2 = spot_price - 2*std_price
    hi2 = spot_price + 2*std_price

    # Strategy strikes
    ic_sell_put = round(lo1 / diff) * diff
    ic_sell_call = round(hi1 / diff) * diff
    ic_buy_put = ic_sell_put - diff
    ic_buy_call = ic_sell_call + diff

    # ═══════════════════════════════════════════════
    #  TOP KPI ROW
    # ═══════════════════════════════════════════════

    k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
    k1.metric("Spot", f"₹{spot_price:,.2f}")
    k2.metric("ATM", f"{atm_strike:,.0f}")
    k3.metric("PCR", f"{pcr}")
    k4.metric("Max Pain", f"{max_pain:,.0f}")
    k5.metric("VWAS", f"{vwas:,.0f}")
    adx_val = adx_metrics["adx"] if adx_metrics else None
    k6.metric("ADX (14)", f"{adx_val}" if adx_val else "—",
              f"+DI {adx_metrics['plus_di']}/-DI {adx_metrics['minus_di']}" if adx_metrics else None)
    k7.metric("DTE", f"{tte_days:.1f} days")

    # ── IV Percentile Gauge ──
    iv_color = "#22c55e" if iv_pct < 30 else "#f59e0b" if iv_pct < 70 else "#ef4444"
    iv_source_tag = '<span style="font-size:9px; color:#4ade80; background:rgba(34,197,94,0.15); padding:1px 6px; border-radius:3px; margin-left:6px;">LIVE</span>' if atm_iv_is_live else '<span style="font-size:9px; color:#f59e0b; background:rgba(245,158,11,0.15); padding:1px 6px; border-radius:3px; margin-left:6px;">MANUAL</span>'
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:14px; margin:8px 0 4px 0;">
        <span style="font-size:11px; color:#64748b; text-transform:uppercase; letter-spacing:1.5px; font-weight:600; min-width:110px;">IV Percentile</span>
        <div class="iv-gauge-bar" style="flex:1;">
            <div class="iv-gauge-marker" style="left:{min(iv_pct, 100)}%;"></div>
        </div>
        <span class="iv-pct-label" style="color:{iv_color};">{iv_pct:.0f}%</span>
        <span style="font-size:11px; color:#e2e8f0; font-weight:600;">ATM IV: {atm_iv:.1f}%</span>
        {iv_source_tag}
    </div>
    """, unsafe_allow_html=True)

    # ── Sentiment Card ──
    st.markdown(f"""
    <div class="signal-card" style="background:{card_bg};">
        <div class="signal-label">Structural Trend Signal</div>
        <div class="signal-value">{sentiment}</div>
        <div class="signal-sub">
            Put OI: {total_pe_oi:,.0f} &nbsp;·&nbsp; Call OI: {total_ce_oi:,.0f}
            &nbsp;&nbsp;
            <span class="badge badge-green">Support {support:,.0f}</span>
            <span class="badge badge-red">Resistance {resistance:,.0f}</span>
            <span class="badge badge-purple">Max Pain {max_pain:,.0f}</span>
            <span class="badge badge-amber">IV Pct {iv_pct:.0f}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── VWAS Highlight Bar ──
    vwas_diff = vwas - spot_price
    vwas_dir = "above" if vwas_diff > 0 else "below" if vwas_diff < 0 else "at"
    vwas_arrow = "▲" if vwas_diff > 0 else "▼" if vwas_diff < 0 else "●"
    vwas_clr = "#4ade80" if vwas_diff > 0 else "#f87171" if vwas_diff < 0 else "#e2e8f0"
    st.markdown(f"""
    <div style="display:flex; align-items:center; justify-content:space-between; padding:14px 20px; margin:10px 0 14px 0;
                border:1px solid #1e293b; border-radius:10px; background:rgba(17,24,39,0.7);">
        <div style="display:flex; align-items:center; gap:14px;">
            <span style="font-size:10px; color:#94a3b8; text-transform:uppercase; letter-spacing:1.8px; font-weight:600;">
                Volume-Weighted Avg Strike
            </span>
            <span style="font-family:'JetBrains Mono',monospace; font-size:26px; font-weight:800; color:#ffffff;">
                {vwas:,.0f}
            </span>
        </div>
        <div style="display:flex; align-items:center; gap:10px;">
            <span style="font-family:'JetBrains Mono',monospace; font-size:14px; font-weight:700; color:{vwas_clr};">
                {vwas_arrow} {abs(vwas_diff):,.1f} pts {vwas_dir} spot
            </span>
            <span style="font-size:11px; color:#64748b;">
                (money flow bias)
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════
    #  OI-CHANGE DIRECTIONAL FORECAST
    # ═══════════════════════════════════════════════

    # Analyze OI change patterns to predict direction and magnitude
    ce_chg_total = df["CE OI Chg"].sum()
    pe_chg_total = df["PE OI Chg"].sum()

    # Weighted OI change — strikes closer to ATM matter more
    df_tmp = df.copy()
    df_tmp["dist"] = abs(df_tmp["Strike"] - atm_strike)
    max_dist = df_tmp["dist"].max()
    df_tmp["weight"] = 1.0 - (df_tmp["dist"] / (max_dist + 1)) if max_dist > 0 else 1.0

    w_ce_buildup = (df_tmp["CE OI Chg"] * df_tmp["weight"]).sum()
    w_pe_buildup = (df_tmp["PE OI Chg"] * df_tmp["weight"]).sum()

    # Find the highest CE OI change strike (new resistance forming)
    # and highest PE OI change strike (new support forming)
    ce_chg_positive = df[df["CE OI Chg"] > 0]
    pe_chg_positive = df[df["PE OI Chg"] > 0]

    new_resistance = float(ce_chg_positive.loc[ce_chg_positive["CE OI Chg"].idxmax(), "Strike"]) if not ce_chg_positive.empty else resistance
    new_support = float(pe_chg_positive.loc[pe_chg_positive["PE OI Chg"].idxmax(), "Strike"]) if not pe_chg_positive.empty else support

    # OI interpretation matrix:
    # Call OI build-up = writers selling calls = bearish (they don't think it'll go up)
    # Put OI build-up = writers selling puts = bullish (they don't think it'll go down)
    # Call OI unwinding = writers exiting calls = bullish (removing ceiling)
    # Put OI unwinding = writers exiting puts = bearish (removing floor)

    signals = []
    direction_score = 0  # positive = bullish, negative = bearish

    # Signal 1: Net OI change direction
    if pe_chg_total > 0 and ce_chg_total > 0:
        if pe_chg_total > ce_chg_total * 1.3:
            direction_score += 25
            signals.append(("Put OI build-up dominates Call build-up", "bullish",
                           f"Put writers added {pe_chg_total:+,.0f} vs Call writers {ce_chg_total:+,.0f} — fresh support being built"))
        elif ce_chg_total > pe_chg_total * 1.3:
            direction_score -= 25
            signals.append(("Call OI build-up dominates Put build-up", "bearish",
                           f"Call writers added {ce_chg_total:+,.0f} vs Put writers {pe_chg_total:+,.0f} — fresh resistance being built"))
        else:
            signals.append(("Balanced OI build-up on both sides", "neutral",
                           f"Call Δ: {ce_chg_total:+,.0f}, Put Δ: {pe_chg_total:+,.0f} — no clear directional bias from OI change"))
    elif pe_chg_total > 0 and ce_chg_total <= 0:
        direction_score += 30
        signals.append(("Put build-up + Call unwinding", "bullish",
                       f"Writers adding puts ({pe_chg_total:+,.0f}) and exiting calls ({ce_chg_total:+,.0f}) — strong bullish conviction"))
    elif ce_chg_total > 0 and pe_chg_total <= 0:
        direction_score -= 30
        signals.append(("Call build-up + Put unwinding", "bearish",
                       f"Writers adding calls ({ce_chg_total:+,.0f}) and exiting puts ({pe_chg_total:+,.0f}) — strong bearish conviction"))
    elif ce_chg_total < 0 and pe_chg_total < 0:
        signals.append(("Both Call and Put OI unwinding", "neutral",
                       f"Writers exiting both sides — potential for big move in either direction (breakout likely)"))

    # Signal 2: Where is the fresh build-up relative to spot
    if not ce_chg_positive.empty:
        ce_buildup_above = ce_chg_positive[ce_chg_positive["Strike"] > spot_price]["CE OI Chg"].sum()
        ce_buildup_below = ce_chg_positive[ce_chg_positive["Strike"] <= spot_price]["CE OI Chg"].sum()
        if ce_buildup_above > ce_buildup_below * 2:
            direction_score -= 10
            signals.append(("Call writing concentrated above spot", "bearish",
                           f"Fresh Call OI build-up at {new_resistance:,.0f} — resistance wall forming"))
    if not pe_chg_positive.empty:
        pe_buildup_below = pe_chg_positive[pe_chg_positive["Strike"] < spot_price]["PE OI Chg"].sum()
        pe_buildup_above = pe_chg_positive[pe_chg_positive["Strike"] >= spot_price]["PE OI Chg"].sum()
        if pe_buildup_below > pe_buildup_above * 2:
            direction_score += 10
            signals.append(("Put writing concentrated below spot", "bullish",
                           f"Fresh Put OI build-up at {new_support:,.0f} — support floor forming"))

    # Signal 3: Near-ATM weighted bias
    if abs(w_pe_buildup) + abs(w_ce_buildup) > 0:
        near_atm_ratio = w_pe_buildup / (abs(w_pe_buildup) + abs(w_ce_buildup))
        if near_atm_ratio > 0.6:
            direction_score += 15
            signals.append(("Near-ATM OI skews toward Put writing", "bullish",
                           "Writers near ATM are predominantly selling puts — immediate upside expected"))
        elif near_atm_ratio < 0.4:
            direction_score -= 15
            signals.append(("Near-ATM OI skews toward Call writing", "bearish",
                           "Writers near ATM are predominantly selling calls — immediate downside expected"))

    # Compute expected move from OI structure
    upside_target = new_resistance  # Ceiling
    downside_target = new_support   # Floor
    upside_pts = upside_target - spot_price
    downside_pts = spot_price - downside_target

    # Direction determination
    if direction_score >= 20:
        dir_label = "BULLISH"
        dir_color = "#4ade80"
        dir_bg = "linear-gradient(135deg, rgba(34,197,94,0.15), rgba(17,24,39,0.8))"
        dir_icon = "🟢"
        dir_arrow = "▲"
        primary_target = upside_target
        primary_pts = upside_pts
        secondary_target = downside_target
    elif direction_score <= -20:
        dir_label = "BEARISH"
        dir_color = "#f87171"
        dir_bg = "linear-gradient(135deg, rgba(239,68,68,0.15), rgba(17,24,39,0.8))"
        dir_icon = "🔴"
        dir_arrow = "▼"
        primary_target = downside_target
        primary_pts = downside_pts
        secondary_target = upside_target
    else:
        dir_label = "SIDEWAYS"
        dir_color = "#fbbf24"
        dir_bg = "linear-gradient(135deg, rgba(245,158,11,0.12), rgba(17,24,39,0.8))"
        dir_icon = "🟡"
        dir_arrow = "↔"
        primary_target = max_pain
        primary_pts = abs(spot_price - max_pain)
        secondary_target = max_pain

    # Build signals HTML
    signals_html = ""
    for title, bias, detail in signals:
        if bias == "bullish":
            dot = '<span style="color:#4ade80;">●</span>'
        elif bias == "bearish":
            dot = '<span style="color:#f87171;">●</span>'
        else:
            dot = '<span style="color:#fbbf24;">●</span>'
        signals_html += f"""
        <div style="padding:6px 0; border-bottom:1px solid rgba(30,41,59,0.5);">
            <div style="font-size:13px; font-weight:600; color:#e2e8f0;">{dot} {title}</div>
            <div style="font-size:11px; color:#94a3b8; margin-top:2px; padding-left:16px;">{detail}</div>
        </div>"""

    st.markdown(f"""
    <div style="border:1px solid #1e293b; border-radius:12px; padding:0; margin:12px 0 14px 0;
                background:{dir_bg}; overflow:hidden;">
        <!-- Header -->
        <div style="display:flex; align-items:center; justify-content:space-between;
                    padding:16px 20px; border-bottom:1px solid #1e293b;">
            <div style="display:flex; align-items:center; gap:10px;">
                <span style="font-size:10px; color:#94a3b8; text-transform:uppercase; letter-spacing:2px; font-weight:600;">
                    OI Change Directional Forecast
                </span>
            </div>
            <div style="display:flex; align-items:center; gap:8px;">
                <span style="font-size:22px;">{dir_icon}</span>
                <span style="font-size:20px; font-weight:800; color:{dir_color}; font-family:'JetBrains Mono',monospace;">
                    {dir_label}
                </span>
            </div>
        </div>
        <!-- Targets Row -->
        <div style="display:flex; padding:14px 20px; gap:0; border-bottom:1px solid #1e293b;">
            <div style="flex:1; text-align:center; border-right:1px solid #1e293b;">
                <div style="font-size:9px; color:#64748b; text-transform:uppercase; letter-spacing:1.5px; font-weight:600;">
                    Downside Target
                </div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:20px; font-weight:800; color:#f87171; margin-top:4px;">
                    {downside_target:,.0f}
                </div>
                <div style="font-size:11px; color:#f87171;">▼ {downside_pts:,.0f} pts</div>
            </div>
            <div style="flex:1; text-align:center; border-right:1px solid #1e293b;">
                <div style="font-size:9px; color:#64748b; text-transform:uppercase; letter-spacing:1.5px; font-weight:600;">
                    Spot Now
                </div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:20px; font-weight:800; color:#ffffff; margin-top:4px;">
                    {spot_price:,.1f}
                </div>
                <div style="font-size:11px; color:#94a3b8;">ATM {atm_strike:,.0f}</div>
            </div>
            <div style="flex:1; text-align:center;">
                <div style="font-size:9px; color:#64748b; text-transform:uppercase; letter-spacing:1.5px; font-weight:600;">
                    Upside Target
                </div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:20px; font-weight:800; color:#4ade80; margin-top:4px;">
                    {upside_target:,.0f}
                </div>
                <div style="font-size:11px; color:#4ade80;">▲ {upside_pts:,.0f} pts</div>
            </div>
        </div>
        <!-- OI Change Summary -->
        <div style="display:flex; padding:10px 20px; gap:0; border-bottom:1px solid #1e293b;">
            <div style="flex:1; text-align:center;">
                <span style="font-size:10px; color:#64748b; text-transform:uppercase; letter-spacing:1px;">Call OI Δ</span><br>
                <span style="font-family:'JetBrains Mono',monospace; font-size:15px; font-weight:700;
                             color:{'#f87171' if ce_chg_total > 0 else '#4ade80'};">
                    {ce_chg_total:+,.0f}
                </span>
                <span style="font-size:10px; color:#64748b;"> {'(resistance ↑)' if ce_chg_total > 0 else '(ceiling removed)'}</span>
            </div>
            <div style="flex:1; text-align:center;">
                <span style="font-size:10px; color:#64748b; text-transform:uppercase; letter-spacing:1px;">Put OI Δ</span><br>
                <span style="font-family:'JetBrains Mono',monospace; font-size:15px; font-weight:700;
                             color:{'#4ade80' if pe_chg_total > 0 else '#f87171'};">
                    {pe_chg_total:+,.0f}
                </span>
                <span style="font-size:10px; color:#64748b;"> {'(support ↑)' if pe_chg_total > 0 else '(floor removed)'}</span>
            </div>
        </div>
        <!-- Signals -->
        <div style="padding:12px 20px;">
            <div style="font-size:9px; color:#64748b; text-transform:uppercase; letter-spacing:1.5px; font-weight:600; margin-bottom:6px;">
                Signal Breakdown
            </div>
            {signals_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════
    #  TABBED LAYOUT
    # ═══════════════════════════════════════════════

    tab_dash, tab_chain, tab_strat, tab_charts = st.tabs([
        "📊 Dashboard", "📋 Options Chain", "🛡️ Strategy Builder", "📈 Charts"
    ])

    # ──────────────────────────────────
    #  TAB 1: DASHBOARD
    # ──────────────────────────────────
    with tab_dash:

        # Normal distribution
        st.markdown("#### 🎯 Statistical Price Forecast")
        pc1, pc2, pc3, pc4 = st.columns(4)
        pc1.metric("2σ Low (95.4%)", f"₹{lo2:,.0f}")
        pc2.metric("1σ Low (68.2%)", f"₹{lo1:,.0f}")
        pc3.metric("1σ High (68.2%)", f"₹{hi1:,.0f}")
        pc4.metric("2σ High (95.4%)", f"₹{hi2:,.0f}")

        # Bell curve with Plotly
        x = np.linspace(spot_price - 3.5*std_price, spot_price + 3.5*std_price, 400)
        y = norm.pdf(x, spot_price, std_price)

        fig_bell = go.Figure()
        # 2σ fill
        mask_2s = (x >= lo2) & (x <= hi2)
        fig_bell.add_trace(go.Scatter(x=x[mask_2s], y=y[mask_2s], fill="tozeroy",
            fillcolor="rgba(59,130,246,0.08)", line=dict(width=0), name="95.4% (2σ)", showlegend=True))
        # 1σ fill
        mask_1s = (x >= lo1) & (x <= hi1)
        fig_bell.add_trace(go.Scatter(x=x[mask_1s], y=y[mask_1s], fill="tozeroy",
            fillcolor="rgba(56,189,248,0.22)", line=dict(width=0), name="68.2% (1σ)", showlegend=True))
        # PDF line
        fig_bell.add_trace(go.Scatter(x=x, y=y, mode="lines",
            line=dict(color="#e2e8f0", width=2), name="PDF"))
        # Vertical lines
        for val, clr, nm, dash in [
            (spot_price, "#f1f5f9", "Spot", "solid"),
            (max_pain, "#8b5cf6", "Max Pain", "dashdot"),
            (support, "#22c55e", "Support", "dot"),
            (resistance, "#ef4444", "Resistance", "dot"),
        ]:
            fig_bell.add_vline(x=val, line=dict(color=clr, width=1.5, dash=dash),
                               annotation_text=nm, annotation_position="top")

        fig_bell.update_layout(**PLOTLY_LAYOUT, height=320,
            title=dict(text=f"Expiry Forecast — {selected_index_name} ({selected_expiry})", font=dict(size=13)),
            yaxis_title="Probability Density", xaxis_title="Settlement Price")
        st.plotly_chart(fig_bell, use_container_width=True)

        # PCR Trend sparkline
        if len(st.session_state.pcr_history) > 1:
            st.markdown("#### 📉 PCR Trend (Session)")
            pcr_df = pd.DataFrame(st.session_state.pcr_history)
            fig_pcr = go.Figure()
            fig_pcr.add_trace(go.Scatter(x=pcr_df["time"], y=pcr_df["pcr"], mode="lines+markers",
                line=dict(color="#3b82f6", width=2), marker=dict(size=4), name="PCR"))
            fig_pcr.add_hline(y=1.0, line=dict(color="#64748b", dash="dash", width=1), annotation_text="Neutral")
            fig_pcr.update_layout(**PLOTLY_LAYOUT, height=200,
                yaxis_title="PCR", xaxis_title="Time")
            st.plotly_chart(fig_pcr, use_container_width=True)

        # OI Change summary
        st.markdown("#### 🔄 OI Change (Current vs Previous)")
        oc1, oc2 = st.columns(2)
        with oc1:
            top_ce_buildup = df.nlargest(5, "CE OI Chg")[["Strike", "CE OI", "CE OI Chg"]]
            st.markdown("**Top Call OI Build-up**")
            st.dataframe(top_ce_buildup.style.format({"Strike":"{:,.0f}","CE OI":"{:,.0f}","CE OI Chg":"{:+,.0f}"}),
                         use_container_width=True, hide_index=True)
        with oc2:
            top_pe_buildup = df.nlargest(5, "PE OI Chg")[["Strike", "PE OI", "PE OI Chg"]]
            st.markdown("**Top Put OI Build-up**")
            st.dataframe(top_pe_buildup.style.format({"Strike":"{:,.0f}","PE OI":"{:,.0f}","PE OI Chg":"{:+,.0f}"}),
                         use_container_width=True, hide_index=True)

    # ──────────────────────────────────
    #  TAB 2: OPTIONS CHAIN
    # ──────────────────────────────────
    with tab_chain:
        st.markdown("#### Live Options Chain — Greeks & Market Data")

        # Build a color-mapped HTML table for better visual density
        display_cols = ["CE OI","CE OI Chg","CE Vol","CE IV","CE Delta","CE Theta","CE LTP",
                        "Strike",
                        "PE LTP","PE Theta","PE Delta","PE IV","PE Vol","PE OI Chg","PE OI"]
        df_display = df[display_cols].copy()

        def oi_bg(val, max_val):
            if max_val == 0: return ""
            intensity = min(abs(val)/max_val, 1.0)
            return f"rgba(59,130,246,{intensity*0.25})"

        def chg_color(val):
            if val > 0: return "color: #4ade80;"
            elif val < 0: return "color: #f87171;"
            return ""

        max_oi = max(df["CE OI"].max(), df["PE OI"].max(), 1)

        def style_chain(row):
            s = row["Strike"]
            styles = [""] * len(row)
            si = list(row.index)

            # ATM highlight
            if s == atm_strike:
                styles = ["background-color: rgba(245,158,11,0.15); font-weight:700;"] * len(row)

            # OI intensity on CE OI and PE OI columns
            for col_name in ["CE OI", "PE OI"]:
                idx = si.index(col_name)
                intensity = min(abs(row[col_name])/max_oi, 1.0) if max_oi > 0 else 0
                bg = f"rgba(59,130,246,{intensity*0.3})"
                styles[idx] += f" background-color: {bg};"

            # OI Change coloring
            for col_name in ["CE OI Chg", "PE OI Chg"]:
                idx = si.index(col_name)
                v = row[col_name]
                if v > 0:
                    styles[idx] += " color: #4ade80;"
                elif v < 0:
                    styles[idx] += " color: #f87171;"

            return styles

        styled = df_display.style.apply(style_chain, axis=1).format({
            "CE OI":"{:,.0f}", "CE OI Chg":"{:+,.0f}", "CE Vol":"{:,.0f}", "CE IV":"{:.1f}",
            "CE Delta":"{:.3f}", "CE Theta":"{:.2f}", "CE LTP":"₹{:.2f}",
            "Strike":"{:,.0f}",
            "PE LTP":"₹{:.2f}", "PE Theta":"{:.2f}", "PE Delta":"{:.3f}", "PE IV":"{:.1f}",
            "PE Vol":"{:,.0f}", "PE OI Chg":"{:+,.0f}", "PE OI":"{:,.0f}",
        })
        st.dataframe(styled, use_container_width=True, height=500)

    # ──────────────────────────────────
    #  TAB 3: STRATEGY BUILDER
    # ──────────────────────────────────
    with tab_strat:
        st.markdown("#### 🛡️ Strategy Playbook")

        # ── Smart Strategy Scoring Engine ──
        adx_v = adx_metrics["adx"] if adx_metrics else 15
        plus_di = adx_metrics["plus_di"] if adx_metrics else 0
        minus_di = adx_metrics["minus_di"] if adx_metrics else 0
        is_trending = adx_v > 25
        is_bullish_trend = plus_di > minus_di
        is_range_bound = adx_v < 20
        spot_vs_maxpain = spot_price - max_pain  # positive = spot above max pain
        spot_near_support = abs(spot_price - support) <= 2 * diff
        spot_near_resistance = abs(spot_price - resistance) <= 2 * diff

        # Score each strategy (higher = better fit for current conditions)
        scores = {}
        reasons = {}

        # ── Iron Condor ──
        ic_score = 0
        ic_why = []
        if is_range_bound:
            ic_score += 30
            ic_why.append(f"ADX is {adx_v} (below 20) — market is range-bound, ideal for condors")
        elif not is_trending:
            ic_score += 15
            ic_why.append(f"ADX is {adx_v} — no strong trend detected")
        if 0.85 <= pcr <= 1.15:
            ic_score += 20
            ic_why.append(f"PCR is {pcr} — balanced put/call sentiment supports range expectation")
        if 30 <= iv_pct <= 70:
            ic_score += 15
            ic_why.append(f"IV Percentile at {iv_pct:.0f}% — moderate, good for selling with wings")
        elif iv_pct > 70:
            ic_score += 10
            ic_why.append(f"IV Percentile at {iv_pct:.0f}% — high IV boosts premium collected")
        if tte_days <= 7:
            ic_score += 10
            ic_why.append(f"Only {tte_days:.1f} DTE — short expiry favors premium sellers")
        if abs(spot_vs_maxpain) < 2 * diff:
            ic_score += 10
            ic_why.append(f"Spot is near Max Pain ({max_pain:,.0f}) — likely to stay pinned")
        scores["Iron Condor"] = ic_score
        reasons["Iron Condor"] = ic_why

        # ── Short Straddle ──
        ss_score = 0
        ss_why = []
        if is_range_bound:
            ss_score += 25
            ss_why.append(f"ADX at {adx_v} — range-bound market suits straddle selling")
        if iv_pct > 60:
            ss_score += 30
            ss_why.append(f"IV Percentile at {iv_pct:.0f}% — elevated IV means fat premiums to collect")
        elif iv_pct > 40:
            ss_score += 10
            ss_why.append(f"IV Percentile at {iv_pct:.0f}% — decent premium available")
        if 0.90 <= pcr <= 1.10:
            ss_score += 15
            ss_why.append(f"PCR at {pcr} — very balanced, no directional pressure")
        if tte_days <= 5:
            ss_score += 15
            ss_why.append(f"{tte_days:.1f} DTE — rapid theta decay maximizes straddle income")
        if is_trending:
            ss_score -= 20
            ss_why.append(f"⚠️ ADX at {adx_v} — trending market is dangerous for naked straddles")
        scores["Short Straddle"] = ss_score
        reasons["Short Straddle"] = ss_why

        # ── Iron Butterfly ──
        ib_score = 0
        ib_why = []
        if iv_pct > 65:
            ib_score += 30
            ib_why.append(f"IV Percentile at {iv_pct:.0f}% — high IV is the butterfly's best friend")
        if is_range_bound:
            ib_score += 20
            ib_why.append(f"ADX at {adx_v} — low trend strength supports pinning at ATM")
        if abs(spot_vs_maxpain) < diff:
            ib_score += 20
            ib_why.append(f"Spot is very close to Max Pain ({max_pain:,.0f}) — high pin probability")
        elif abs(spot_vs_maxpain) < 2 * diff:
            ib_score += 10
            ib_why.append(f"Spot is near Max Pain — moderate pin probability")
        if tte_days <= 3:
            ib_score += 15
            ib_why.append(f"Only {tte_days:.1f} DTE — expiry-week pinning effect strongest now")
        if is_trending:
            ib_score -= 10
            ib_why.append(f"⚠️ ADX at {adx_v} — trending market makes ATM pinning less likely")
        scores["Iron Butterfly"] = ib_score
        reasons["Iron Butterfly"] = ib_why

        # ── Bull Put Spread ──
        bps_score = 0
        bps_why = []
        if pcr > 1.10:
            bps_score += 25
            bps_why.append(f"PCR at {pcr} — heavy put writing signals support below, bullish")
        elif pcr > 1.0:
            bps_score += 10
            bps_why.append(f"PCR at {pcr} — slightly bullish lean")
        if is_bullish_trend and is_trending:
            bps_score += 25
            bps_why.append(f"+DI ({plus_di}) > -DI ({minus_di}) with ADX {adx_v} — confirmed bullish trend")
        elif is_bullish_trend:
            bps_score += 10
            bps_why.append(f"+DI ({plus_di}) > -DI ({minus_di}) — directional bias is upward")
        if spot_price > max_pain:
            bps_score += 10
            bps_why.append(f"Spot ({spot_price:,.0f}) above Max Pain ({max_pain:,.0f}) — bullish positioning")
        if spot_near_support:
            bps_score += 15
            bps_why.append(f"Spot near strong support at {support:,.0f} — low risk for put selling")
        if iv_pct > 40:
            bps_score += 5
            bps_why.append(f"IV Percentile at {iv_pct:.0f}% — adequate premium for credit spread")
        if pcr < 0.85:
            bps_score -= 15
            bps_why.append(f"⚠️ PCR at {pcr} — bearish OI skew argues against bullish bets")
        scores["Bull Put Spread"] = bps_score
        reasons["Bull Put Spread"] = bps_why

        # ── Bear Call Spread ──
        bcs_score = 0
        bcs_why = []
        if pcr < 0.90:
            bcs_score += 25
            bcs_why.append(f"PCR at {pcr} — heavy call writing signals resistance above, bearish")
        elif pcr < 1.0:
            bcs_score += 10
            bcs_why.append(f"PCR at {pcr} — slightly bearish lean")
        if not is_bullish_trend and is_trending:
            bcs_score += 25
            bcs_why.append(f"-DI ({minus_di}) > +DI ({plus_di}) with ADX {adx_v} — confirmed bearish trend")
        elif not is_bullish_trend:
            bcs_score += 10
            bcs_why.append(f"-DI ({minus_di}) > +DI ({plus_di}) — directional bias is downward")
        if spot_price < max_pain:
            bcs_score += 10
            bcs_why.append(f"Spot ({spot_price:,.0f}) below Max Pain ({max_pain:,.0f}) — bearish positioning")
        if spot_near_resistance:
            bcs_score += 15
            bcs_why.append(f"Spot near strong resistance at {resistance:,.0f} — ceiling likely to hold")
        if iv_pct > 40:
            bcs_score += 5
            bcs_why.append(f"IV Percentile at {iv_pct:.0f}% — adequate premium for credit spread")
        if pcr > 1.15:
            bcs_score -= 15
            bcs_why.append(f"⚠️ PCR at {pcr} — bullish OI skew argues against bearish bets")
        scores["Bear Call Spread"] = bcs_score
        reasons["Bear Call Spread"] = bcs_why

        # ── Rank strategies ──
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_name, best_score = ranked[0]
        second_name, second_score = ranked[1]

        # Strategy icons and colors
        strat_meta = {
            "Iron Condor":     {"icon": "📊", "color": "#3b82f6", "risk": "Defined", "type": "Neutral"},
            "Short Straddle":  {"icon": "🔥", "color": "#f59e0b", "risk": "Unlimited", "type": "Neutral"},
            "Iron Butterfly":  {"icon": "🦋", "color": "#8b5cf6", "risk": "Defined", "type": "Neutral"},
            "Bull Put Spread": {"icon": "📈", "color": "#22c55e", "risk": "Defined", "type": "Bullish"},
            "Bear Call Spread":{"icon": "📉", "color": "#ef4444", "risk": "Defined", "type": "Bearish"},
        }

        bm = strat_meta[best_name]
        best_reasons_html = "".join(
            f'<div style="font-size:13px; color:#e2e8f0; line-height:1.7; padding:2px 0;">✓ {r}</div>'
            for r in reasons[best_name]
        )

        # Confidence label
        if best_score >= 60:
            conf_label = "HIGH CONFIDENCE"
            conf_color = "#4ade80"
        elif best_score >= 40:
            conf_label = "MODERATE CONFIDENCE"
            conf_color = "#fbbf24"
        else:
            conf_label = "LOW CONFIDENCE"
            conf_color = "#f87171"

        st.markdown(f"""
        <div style="border:2px solid {bm['color']}; border-radius:12px; padding:20px; margin:8px 0 16px 0;
                    background:rgba(17,24,39,0.8);">
            <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:12px;">
                <div style="display:flex; align-items:center; gap:10px;">
                    <span style="font-size:10px; color:#94a3b8; text-transform:uppercase; letter-spacing:2px; font-weight:600;">
                        Recommended Strategy
                    </span>
                </div>
                <span style="font-size:10px; font-weight:700; color:{conf_color};
                             background:rgba(0,0,0,0.3); padding:3px 10px; border-radius:4px;
                             letter-spacing:1.5px;">
                    {conf_label}
                </span>
            </div>
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:14px;">
                <span style="font-size:32px;">{bm['icon']}</span>
                <div>
                    <div style="font-size:22px; font-weight:800; color:#ffffff; font-family:'JetBrains Mono',monospace;">
                        {best_name}
                    </div>
                    <div style="font-size:12px; color:#94a3b8; margin-top:2px;">
                        {bm['type']} · {bm['risk']} Risk · Score: {best_score}/100
                    </div>
                </div>
            </div>
            <div style="border-top:1px solid #1e293b; padding-top:12px; margin-top:4px;">
                <div style="font-size:10px; color:#64748b; text-transform:uppercase; letter-spacing:1.5px; font-weight:600; margin-bottom:8px;">
                    Why This Strategy Now
                </div>
                {best_reasons_html}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Show runner-up as a smaller card
        sm = strat_meta[second_name]
        second_reasons_short = reasons[second_name][:2]
        second_reasons_html = "".join(
            f'<span style="font-size:11px; color:#94a3b8;">✓ {r}</span><br>'
            for r in second_reasons_short
        )
        st.markdown(f"""
        <div style="border:1px solid #1e293b; border-radius:8px; padding:12px 16px; margin:0 0 16px 0;
                    background:rgba(17,24,39,0.5); display:flex; align-items:center; gap:14px;">
            <span style="font-size:20px;">{sm['icon']}</span>
            <div style="flex:1;">
                <div style="font-size:13px; font-weight:700; color:#cbd5e1;">
                    Runner-up: {second_name}
                    <span style="font-size:11px; color:#64748b; font-weight:400; margin-left:6px;">
                        Score: {second_score}/100
                    </span>
                </div>
                {second_reasons_html}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Strategy selector — user still picks
        strat_choice = st.selectbox("Select Strategy", ["Iron Condor", "Short Straddle", "Iron Butterfly", "Bull Put Spread", "Bear Call Spread"])

        def get_ltp(strike):
            """
            Look up CE & PE LTP for a strike with 3-tier fallback:
            1. Live LTP from batch API call (already fetched at load time)
            2. Chain DataFrame LTP
            3. Black-Scholes theoretical price (last resort, marked with '(theo)')
            """
            ce_ltp = live_ltp_map.get((strike, "CE"), 0.0)
            pe_ltp = live_ltp_map.get((strike, "PE"), 0.0)

            # Fallback to chain DataFrame
            if ce_ltp == 0 or pe_ltp == 0:
                m = df[df["Strike"] == strike]
                if not m.empty:
                    if ce_ltp == 0:
                        ce_ltp = float(m.iloc[0]["CE LTP"])
                    if pe_ltp == 0:
                        pe_ltp = float(m.iloc[0]["PE LTP"])

            # Last resort: BS theoretical price using live ATM IV
            if ce_ltp == 0:
                ce_ltp = bs_greeks(spot_price, strike, tte_years, risk_free_rate, iv_for_sigma, "CE")["price"]
            if pe_ltp == 0:
                pe_ltp = bs_greeks(spot_price, strike, tte_years, risk_free_rate, iv_for_sigma, "PE")["price"]

            return ce_ltp, pe_ltp

        def price_source_label(strike, opt_type):
            """Returns a label indicating price source."""
            if live_ltp_map.get((strike, opt_type), 0) > 0:
                return ""  # Live — no label needed
            m = df[df["Strike"] == strike]
            if not m.empty:
                col = "CE LTP" if opt_type == "CE" else "PE LTP"
                if float(m.iloc[0][col]) > 0:
                    return ""  # Chain — no label needed
            return " <span style='font-size:10px; color:#94a3b8;'>(theo)</span>"

        legs = []

        if strat_choice == "Iron Condor":
            c_sell_ce, _ = get_ltp(ic_sell_call)
            _, p_sell_pe = get_ltp(ic_sell_put)
            c_buy_ce, _ = get_ltp(ic_buy_call)
            _, p_buy_pe = get_ltp(ic_buy_put)
            net = max((c_sell_ce + p_sell_pe) - (c_buy_ce + p_buy_pe), 0)
            max_risk = diff - net if net > 0 else diff
            legs = [
                {"strike":ic_buy_put,"type":"PE","action":"BUY","premium":p_buy_pe},
                {"strike":ic_sell_put,"type":"PE","action":"SELL","premium":p_sell_pe},
                {"strike":ic_sell_call,"type":"CE","action":"SELL","premium":c_sell_ce},
                {"strike":ic_buy_call,"type":"CE","action":"BUY","premium":c_buy_ce},
            ]
            st.markdown(f"""
            <div class="strat-card">
                <div class="strat-title" style="color:#3b82f6;">📊 Iron Condor</div>
                <div class="strat-leg">
                    BUY 1× <b>{ic_buy_put} PE</b> @ ₹{p_buy_pe:.2f}{price_source_label(ic_buy_put, "PE")}<br>
                    SELL 1× <b>{ic_sell_put} PE</b> @ ₹{p_sell_pe:.2f}{price_source_label(ic_sell_put, "PE")}<br>
                    SELL 1× <b>{ic_sell_call} CE</b> @ ₹{c_sell_ce:.2f}{price_source_label(ic_sell_call, "CE")}<br>
                    BUY 1× <b>{ic_buy_call} CE</b> @ ₹{c_buy_ce:.2f}{price_source_label(ic_buy_call, "CE")}
                </div>
                <div class="strat-profit" style="color:#4ade80;">
                    💰 Net Credit: ₹{net:,.2f}/lot &nbsp;(₹{net*lot_size:,.0f} total)
                </div>
                <div style="color:#fca5a5; font-size:14px; margin-top:6px; font-weight:700;">
                    ⚠️ Max Risk: ₹{max_risk:,.2f}/lot &nbsp;(₹{max_risk*lot_size:,.0f} total)
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif strat_choice == "Short Straddle":
            c_atm, p_atm = get_ltp(atm_strike)
            net = c_atm + p_atm
            upper_be = atm_strike + net
            lower_be = atm_strike - net
            legs = [
                {"strike":atm_strike,"type":"CE","action":"SELL","premium":c_atm},
                {"strike":atm_strike,"type":"PE","action":"SELL","premium":p_atm},
            ]
            st.markdown(f"""
            <div class="strat-card">
                <div class="strat-title" style="color:#f59e0b;">🔥 Short Straddle</div>
                <div class="strat-leg">
                    SELL 1× <b>{atm_strike} CE</b> @ ₹{c_atm:.2f}{price_source_label(atm_strike, "CE")}<br>
                    SELL 1× <b>{atm_strike} PE</b> @ ₹{p_atm:.2f}{price_source_label(atm_strike, "PE")}
                </div>
                <div class="strat-profit" style="color:#4ade80;">
                    💰 Net Credit: ₹{net:,.2f}/lot &nbsp;(₹{net*lot_size:,.0f} total)
                </div>
                <div style="font-size:14px; color:#e2e8f0; margin-top:6px; font-weight:600;">
                    Breakevens: ₹{lower_be:,.0f} – ₹{upper_be:,.0f} &nbsp;⚠️ Unlimited risk
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif strat_choice == "Iron Butterfly":
            c_atm, p_atm = get_ltp(atm_strike)
            c_buy_ce, _ = get_ltp(ic_buy_call)
            _, p_buy_pe = get_ltp(ic_buy_put)
            net = max((c_atm + p_atm) - (c_buy_ce + p_buy_pe), 0)
            legs = [
                {"strike":ic_buy_put,"type":"PE","action":"BUY","premium":p_buy_pe},
                {"strike":atm_strike,"type":"PE","action":"SELL","premium":p_atm},
                {"strike":atm_strike,"type":"CE","action":"SELL","premium":c_atm},
                {"strike":ic_buy_call,"type":"CE","action":"BUY","premium":c_buy_ce},
            ]
            st.markdown(f"""
            <div class="strat-card">
                <div class="strat-title" style="color:#8b5cf6;">🦋 Iron Butterfly</div>
                <div class="strat-leg">
                    BUY 1× <b>{ic_buy_put} PE</b> @ ₹{p_buy_pe:.2f}{price_source_label(ic_buy_put, "PE")}<br>
                    SELL 1× <b>{atm_strike} PE</b> @ ₹{p_atm:.2f}{price_source_label(atm_strike, "PE")}<br>
                    SELL 1× <b>{atm_strike} CE</b> @ ₹{c_atm:.2f}{price_source_label(atm_strike, "CE")}<br>
                    BUY 1× <b>{ic_buy_call} CE</b> @ ₹{c_buy_ce:.2f}{price_source_label(ic_buy_call, "CE")}
                </div>
                <div class="strat-profit" style="color:#4ade80;">
                    💰 Net Credit: ₹{net:,.2f}/lot &nbsp;(₹{net*lot_size:,.0f} total)
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif strat_choice == "Bull Put Spread":
            sell_strike = ic_sell_put
            buy_strike = ic_buy_put
            _, p_sell = get_ltp(sell_strike)
            _, p_buy = get_ltp(buy_strike)
            net = max(p_sell - p_buy, 0)
            max_risk = diff - net
            legs = [
                {"strike":buy_strike,"type":"PE","action":"BUY","premium":p_buy},
                {"strike":sell_strike,"type":"PE","action":"SELL","premium":p_sell},
            ]
            st.markdown(f"""
            <div class="strat-card">
                <div class="strat-title" style="color:#22c55e;">📈 Bull Put Spread</div>
                <div class="strat-leg">
                    SELL 1× <b>{sell_strike} PE</b> @ ₹{p_sell:.2f}{price_source_label(sell_strike, "PE")}<br>
                    BUY 1× <b>{buy_strike} PE</b> @ ₹{p_buy:.2f}{price_source_label(buy_strike, "PE")}
                </div>
                <div class="strat-profit" style="color:#4ade80;">
                    💰 Net Credit: ₹{net:,.2f}/lot &nbsp;(₹{net*lot_size:,.0f} total)
                </div>
                <div style="color:#fca5a5; font-size:14px; margin-top:6px; font-weight:700;">
                    ⚠️ Max Risk: ₹{max_risk:,.2f}/lot
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif strat_choice == "Bear Call Spread":
            sell_strike = ic_sell_call
            buy_strike = ic_buy_call
            c_sell, _ = get_ltp(sell_strike)
            c_buy, _ = get_ltp(buy_strike)
            net = max(c_sell - c_buy, 0)
            max_risk = diff - net
            legs = [
                {"strike":sell_strike,"type":"CE","action":"SELL","premium":c_sell},
                {"strike":buy_strike,"type":"CE","action":"BUY","premium":c_buy},
            ]
            st.markdown(f"""
            <div class="strat-card">
                <div class="strat-title" style="color:#ef4444;">📉 Bear Call Spread</div>
                <div class="strat-leg">
                    SELL 1× <b>{sell_strike} CE</b> @ ₹{c_sell:.2f}{price_source_label(sell_strike, "CE")}<br>
                    BUY 1× <b>{buy_strike} CE</b> @ ₹{c_buy:.2f}{price_source_label(buy_strike, "CE")}
                </div>
                <div class="strat-profit" style="color:#4ade80;">
                    💰 Net Credit: ₹{net:,.2f}/lot &nbsp;(₹{net*lot_size:,.0f} total)
                </div>
                <div style="color:#fca5a5; font-size:14px; margin-top:6px; font-weight:700;">
                    ⚠️ Max Risk: ₹{max_risk:,.2f}/lot
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── P&L Heatmap ──
        if legs:
            st.markdown("#### 🗺️ P&L Heatmap — What You Make or Lose")
            st.caption("Each cell shows your total profit (green) or loss (red) in ₹, based on where the index lands (columns) and how many days remain until expiry (rows). Hover any cell for details.")

            spot_arr, dte_arr, pnl_mat = compute_pnl_heatmap(legs, lot_size, spot_price, diff, iv_for_sigma, risk_free_rate)

            # Format labels
            spot_labels = [f"{int(s):,}" for s in spot_arr]
            dte_labels = [f"{int(d)}d left" if d > 0 else "Expiry Day" for d in dte_arr]

            # Smart text: show ₹ values in K for large numbers
            def fmt_cell(v):
                v = int(round(v))
                if abs(v) >= 1000:
                    return f"₹{v/1000:+.1f}K"
                return f"₹{v:+,}"

            text_mat = [[fmt_cell(pnl_mat[i][j]) for j in range(len(spot_arr))] for i in range(len(dte_arr))]

            # Color: red for loss, green for profit, dark neutral at zero
            colorscale = [
                [0.0,  "#991b1b"],   # deep red — max loss
                [0.3,  "#ef4444"],   # red
                [0.45, "#fca5a5"],   # light red
                [0.5,  "#1e293b"],   # dark neutral — breakeven
                [0.55, "#86efac"],   # light green
                [0.7,  "#22c55e"],   # green
                [1.0,  "#15803d"],   # deep green — max profit
            ]

            fig_hm = go.Figure(data=go.Heatmap(
                z=pnl_mat,
                x=spot_labels,
                y=dte_labels,
                colorscale=colorscale,
                zmid=0,
                text=text_mat,
                texttemplate="%{text}",
                textfont=dict(size=11, family="JetBrains Mono, monospace"),
                hovertemplate=(
                    "<b>If index at %{x}</b><br>"
                    "With %{y}<br>"
                    "Your P&L: <b>₹%{z:,.0f}</b>"
                    "<extra></extra>"
                ),
                colorbar=dict(
                    title=dict(text="P&L (₹)", font=dict(size=11)),
                    tickformat=",",
                    tickprefix="₹",
                    len=0.9,
                ),
                xgap=2, ygap=2,
            ))

            fig_hm.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(17,24,39,0.6)",
                font=dict(family="Inter, sans-serif", size=11, color="#94a3b8"),
                margin=dict(l=50, r=30, t=40, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
                height=380,
                title=dict(text=f"{strat_choice} — {lot_size} lot size", font=dict(size=14)),
                xaxis=dict(
                    title="Index Level ➜",
                    tickangle=-45,
                    gridcolor="rgba(30,41,59,0.4)",
                    side="bottom",
                    zeroline=False,
                ),
                yaxis=dict(
                    title="",
                    gridcolor="rgba(30,41,59,0.4)",
                    autorange="reversed",
                    zeroline=False,
                ),
            )
            # Mark current spot on categorical axis
            current_label = f"{int(round(spot_price / diff) * diff):,}"
            if current_label in spot_labels:
                idx = spot_labels.index(current_label)
                fig_hm.add_shape(
                    type="line", x0=idx, x1=idx, y0=-0.5, y1=len(dte_arr)-0.5,
                    xref="x", yref="y",
                    line=dict(color="#fbbf24", width=2.5, dash="solid"),
                )
                fig_hm.add_annotation(
                    x=idx, y=-0.5, xref="x", yref="y",
                    text="▼ You Are Here", showarrow=False,
                    font=dict(size=11, color="#fbbf24", family="Inter"),
                    yshift=-16,
                )
            st.plotly_chart(fig_hm, use_container_width=True)

            # ── Payoff at Expiry — simplified ──
            st.markdown("#### 📐 Payoff at Expiry — Your Profit/Loss If You Hold Until End")
            st.caption("This shows your final P&L based on where the index closes on expiry day. The flat green zone in the middle is your 'safe zone' where you keep the premium collected.")

            expiry_pnl = pnl_mat[np.where(dte_arr == 0)[0][0]] if 0 in dte_arr else pnl_mat[0]

            # Check if all zeros (premiums were 0)
            has_real_data = np.any(expiry_pnl != 0)

            if not has_real_data:
                st.warning(
                    "⚠️ All strategy leg premiums are ₹0.00 — this usually means the market is closed "
                    "or the selected strikes don't have active quotes. The payoff chart will appear flat. "
                    "Try again during market hours (9:15 AM – 3:30 PM IST)."
                )

            fig_payoff = go.Figure()

            # Profit zone (green fill above zero)
            profit_y = np.where(expiry_pnl >= 0, expiry_pnl, 0)
            fig_payoff.add_trace(go.Scatter(
                x=spot_arr, y=profit_y, mode="lines",
                line=dict(color="rgba(0,0,0,0)", width=0),
                fill="tozeroy", fillcolor="rgba(34,197,94,0.25)",
                name="Profit Zone", showlegend=True,
            ))

            # Loss zone (red fill below zero)
            loss_y = np.where(expiry_pnl <= 0, expiry_pnl, 0)
            fig_payoff.add_trace(go.Scatter(
                x=spot_arr, y=loss_y, mode="lines",
                line=dict(color="rgba(0,0,0,0)", width=0),
                fill="tozeroy", fillcolor="rgba(239,68,68,0.25)",
                name="Loss Zone", showlegend=True,
            ))

            # Main P&L line
            fig_payoff.add_trace(go.Scatter(
                x=spot_arr, y=expiry_pnl, mode="lines+markers",
                line=dict(color="#f1f5f9", width=2.5),
                marker=dict(size=5, color="#f1f5f9"),
                name="Your P&L",
                hovertemplate="Index at %{x:,.0f}<br>P&L: <b>₹%{y:,.0f}</b><extra></extra>",
            ))

            # Breakeven line
            fig_payoff.add_hline(y=0, line=dict(color="#fbbf24", width=1.5, dash="dash"),
                                 annotation_text="Breakeven", annotation_position="bottom right",
                                 annotation=dict(font=dict(color="#fbbf24", size=10)))

            # Current spot marker
            fig_payoff.add_vline(x=spot_price, line=dict(color="#3b82f6", dash="dash", width=1.5),
                                 annotation_text="Current Spot",
                                 annotation=dict(font=dict(color="#60a5fa", size=10)))

            # Mark max profit and max loss
            max_p = np.max(expiry_pnl)
            max_l = np.min(expiry_pnl)
            if max_p > 0:
                best_spot = spot_arr[np.argmax(expiry_pnl)]
                fig_payoff.add_annotation(
                    x=best_spot, y=max_p,
                    text=f"Max Profit ₹{max_p:,.0f}",
                    showarrow=True, arrowhead=2, arrowcolor="#4ade80",
                    font=dict(color="#4ade80", size=11, family="JetBrains Mono"),
                    bgcolor="rgba(17,24,39,0.8)", bordercolor="#4ade80", borderwidth=1,
                )
            if max_l < 0:
                worst_spot = spot_arr[np.argmin(expiry_pnl)]
                fig_payoff.add_annotation(
                    x=worst_spot, y=max_l,
                    text=f"Max Loss ₹{max_l:,.0f}",
                    showarrow=True, arrowhead=2, arrowcolor="#f87171",
                    font=dict(color="#f87171", size=11, family="JetBrains Mono"),
                    bgcolor="rgba(17,24,39,0.8)", bordercolor="#f87171", borderwidth=1,
                )

            fig_payoff.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(17,24,39,0.6)",
                font=dict(family="Inter, sans-serif", size=11, color="#94a3b8"),
                margin=dict(l=50, r=30, t=40, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
                height=340,
                title=dict(text="What Happens on Expiry Day", font=dict(size=14)),
                xaxis=dict(title="Where the Index Closes ➜", tickformat=",", gridcolor="rgba(30,41,59,0.6)", zeroline=False),
                yaxis=dict(title="Your Profit / Loss (₹)", tickprefix="₹", tickformat=",", gridcolor="rgba(30,41,59,0.6)", zeroline=False),
            )
            st.plotly_chart(fig_payoff, use_container_width=True)

            # ── Quick Summary Box ──
            if has_real_data:
                net_credit = sum(
                    leg["premium"] * (1 if leg["action"] == "SELL" else -1)
                    for leg in legs
                )
                net_credit_total = net_credit * lot_size

                summary_color = "#4ade80" if net_credit > 0 else "#f87171"
                rr_ratio = abs(max_p / max_l) if max_l != 0 else 0

                if max_l != 0:
                    summary_html = f"""
                    <div style="border:1px solid #1e293b; border-radius:10px; padding:16px; margin:10px 0;
                                background:rgba(17,24,39,0.6); display:flex; flex-wrap:wrap; gap:24px; align-items:center;">
                        <div>
                            <div style="font-size:10px; color:#64748b; text-transform:uppercase; letter-spacing:1.5px;">You Collect</div>
                            <div style="font-size:22px; font-weight:800; color:{summary_color}; font-family:'JetBrains Mono',monospace;">
                                ₹{net_credit_total:,.0f}
                            </div>
                        </div>
                        <div>
                            <div style="font-size:10px; color:#64748b; text-transform:uppercase; letter-spacing:1.5px;">Best Case</div>
                            <div style="font-size:22px; font-weight:800; color:#4ade80; font-family:'JetBrains Mono',monospace;">
                                ₹{max_p:,.0f}
                            </div>
                        </div>
                        <div>
                            <div style="font-size:10px; color:#64748b; text-transform:uppercase; letter-spacing:1.5px;">Worst Case</div>
                            <div style="font-size:22px; font-weight:800; color:#f87171; font-family:'JetBrains Mono',monospace;">
                                ₹{max_l:,.0f}
                            </div>
                        </div>
                        <div>
                            <div style="font-size:10px; color:#64748b; text-transform:uppercase; letter-spacing:1.5px;">Risk:Reward</div>
                            <div style="font-size:22px; font-weight:800; color:#e2e8f0; font-family:'JetBrains Mono',monospace;">
                                1 : {rr_ratio:.1f}
                            </div>
                        </div>
                    </div>
                    """
                    st.markdown(summary_html, unsafe_allow_html=True)

    # ──────────────────────────────────
    #  TAB 4: CHARTS
    # ──────────────────────────────────
    with tab_charts:

        # OI Distribution
        st.markdown("#### 📊 Open Interest Distribution")
        fig_oi = go.Figure()
        fig_oi.add_trace(go.Bar(
            x=df["Strike"], y=df["CE OI"], name="Call OI",
            marker=dict(color="#ef4444", opacity=0.85, line=dict(color="#b91c1c", width=0.5)),
            offsetgroup=0,
        ))
        fig_oi.add_trace(go.Bar(
            x=df["Strike"], y=df["PE OI"], name="Put OI",
            marker=dict(color="#22c55e", opacity=0.85, line=dict(color="#15803d", width=0.5)),
            offsetgroup=1,
        ))
        fig_oi.add_vline(x=atm_strike, line=dict(color="#f1f5f9", dash="dash", width=1), annotation_text="ATM")
        fig_oi.add_vline(x=max_pain, line=dict(color="#8b5cf6", dash="dashdot", width=1), annotation_text="Max Pain")
        fig_oi.update_layout(**PLOTLY_LAYOUT, height=380, barmode="group",
            title=dict(text=f"OI Distribution — {selected_index_name} ({selected_expiry})", font=dict(size=13)),
            xaxis_title="Strike", yaxis_title="Open Interest")
        st.plotly_chart(fig_oi, use_container_width=True)

        # OI Change chart
        st.markdown("#### 🔄 OI Change (Build-up / Unwinding)")
        fig_oichg = make_subplots(rows=1, cols=1)
        ce_chg_colors = ["#ef4444" if v >= 0 else "#7f1d1d" for v in df["CE OI Chg"]]
        pe_chg_colors = ["#22c55e" if v >= 0 else "#14532d" for v in df["PE OI Chg"]]
        fig_oichg.add_trace(go.Bar(x=df["Strike"]-diff*0.15, y=df["CE OI Chg"], name="CE OI Δ",
            marker_color=ce_chg_colors, width=diff*0.3))
        fig_oichg.add_trace(go.Bar(x=df["Strike"]+diff*0.15, y=df["PE OI Chg"], name="PE OI Δ",
            marker_color=pe_chg_colors, width=diff*0.3))
        fig_oichg.add_hline(y=0, line=dict(color="#475569", width=1))
        fig_oichg.update_layout(**PLOTLY_LAYOUT, height=320,
            title=dict(text="OI Change — Positive = Build-up, Negative = Unwinding", font=dict(size=13)),
            xaxis_title="Strike", yaxis_title="OI Change")
        st.plotly_chart(fig_oichg, use_container_width=True)

        ch1, ch2 = st.columns(2)

        with ch1:
            # Delta Skew
            st.markdown("#### 📉 Delta Skew")
            fig_d = go.Figure()
            fig_d.add_trace(go.Scatter(x=df["Strike"], y=df["CE Delta"], mode="lines+markers",
                line=dict(color="#3b82f6", width=2), marker=dict(size=4), name="CE Δ"))
            fig_d.add_trace(go.Scatter(x=df["Strike"], y=df["PE Delta"], mode="lines+markers",
                line=dict(color="#ef4444", width=2), marker=dict(size=4), name="PE Δ"))
            fig_d.add_hline(y=0, line=dict(color="#475569", width=1))
            fig_d.add_vline(x=atm_strike, line=dict(color="#64748b", dash="dash", width=1))
            fig_d.update_layout(**PLOTLY_LAYOUT, height=320,
                title=dict(text="Delta Across Strikes", font=dict(size=12)),
                xaxis_title="Strike", yaxis_title="Delta")
            st.plotly_chart(fig_d, use_container_width=True)

        with ch2:
            # IV Smile
            st.markdown("#### 😊 IV Smile")
            fig_iv = go.Figure()
            fig_iv.add_trace(go.Scatter(x=df["Strike"], y=df["CE IV"], mode="lines+markers",
                line=dict(color="#3b82f6", width=2), marker=dict(size=4, symbol="triangle-up"), name="CE IV"))
            fig_iv.add_trace(go.Scatter(x=df["Strike"], y=df["PE IV"], mode="lines+markers",
                line=dict(color="#ef4444", width=2), marker=dict(size=4, symbol="triangle-down"), name="PE IV"))
            fig_iv.add_vline(x=atm_strike, line=dict(color="#64748b", dash="dash", width=1))
            fig_iv.update_layout(**PLOTLY_LAYOUT, height=320,
                title=dict(text="IV Smile / Skew", font=dict(size=12)),
                xaxis_title="Strike", yaxis_title="IV (%)")
            st.plotly_chart(fig_iv, use_container_width=True)

        # Cumulative OI Pressure
        st.markdown("#### ⚖️ Cumulative OI Pressure")
        ds = df.sort_values("Strike")
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(x=ds["Strike"], y=ds["CE OI"].cumsum(), mode="lines",
            fill="tozeroy", fillcolor="rgba(239,68,68,0.12)",
            line=dict(color="#ef4444", width=2), name="Cumul. Call OI"))
        fig_cum.add_trace(go.Scatter(x=ds["Strike"], y=ds["PE OI"].cumsum(), mode="lines",
            fill="tozeroy", fillcolor="rgba(34,197,94,0.12)",
            line=dict(color="#22c55e", width=2), name="Cumul. Put OI"))
        fig_cum.add_vline(x=atm_strike, line=dict(color="#64748b", dash="dash", width=1))
        fig_cum.update_layout(**PLOTLY_LAYOUT, height=320,
            title=dict(text="Cumulative OI Build-up", font=dict(size=13)),
            xaxis_title="Strike", yaxis_title="Cumulative OI")
        st.plotly_chart(fig_cum, use_container_width=True)

    # ═══════════════════════════════════════════════
    #  FOOTER
    # ═══════════════════════════════════════════════

    st.markdown("---")
    f1, f2, f3 = st.columns(3)
    f1.caption(f"🕐 {datetime.now().strftime('%H:%M:%S IST')}")
    f2.caption(f"📅 Expiry: {selected_expiry} · DTE: {tte_days:.1f}d")
    f3.caption(f"📊 {len(df)} strikes loaded · Lot: {lot_size}")

    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

except ValueError as ve:
    st.error(f"🚫 **Data Error**: {ve}")
except requests.exceptions.HTTPError as he:
    code = he.response.status_code if he.response is not None else "?"
    st.error(f"🚫 **HTTP {code}**: Token likely expired. Re-authenticate via Upstox OAuth.")
except requests.exceptions.ConnectionError:
    st.error("🚫 **Connection Error**: Can't reach Upstox API.")
except requests.exceptions.Timeout:
    st.error("🚫 **Timeout**: API didn't respond in time.")
except Exception as e:
    st.error(f"🚫 {type(e).__name__}: {e}")
    st.exception(e)
