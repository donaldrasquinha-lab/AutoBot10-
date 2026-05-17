import streamlit as st
import time
import requests
import pandas as pd
import pandas_ta as ta
import upstox_client
from upstox_client.rest import ApiException

# ==============================================================================
# STREAMLIT UI CONFIGURATION & STATE
# ==============================================================================
st.set_page_config(page_title="Upstox Algo Scalper", layout="wide")

# Initialize persistent session states across UI refreshes
if "bot_active" not in st.session_state:
    st.session_state.bot_active = False
if "current_position" not in st.session_state:
    st.session_state.current_position = None
if "trade_logs" not in st.session_state:
    st.session_state.trade_logs = []
if "session_pnl" not in st.session_state:
    st.session_state.session_pnl = 0.0

# ==============================================================================
# BACKEND API HELPERS
# ==============================================================================
def fetch_ltp(token, instrument_key):
    """Retrieves Last Traded Price (LTP) from Upstox API."""
    try:
        url = f"https://upstox.com{instrument_key}"
        headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}
        res = requests.get(url, headers=headers, timeout=5).json()
        return float(res['data'][instrument_key]['last_price'])
    except Exception:
        return None

def fetch_historical_candles(token, instrument_key):
    """Fetches real-time candles to run indicator logic."""
    try:
        url = f"https://upstox.com{instrument_key}/1minute/{time.strftime('%Y-%m-%d')}"
        headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}
        res = requests.get(url, headers=headers, timeout=5).json()
        candles = res['data']['candles']
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
        df = df.iloc[::-1].reset_index(drop=True)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        return df
    except Exception:
        return pd.DataFrame()

# ==============================================================================
# SIDEBAR CONTROL PANEL
# ==============================================================================
st.sidebar.header("🔌 Authentication & Modes")
ACCESS_TOKEN = st.sidebar.text_input("Upstox Access Token", type="password", help="Generate via developer console")
TRADE_MODE = st.sidebar.radio("🎯 Execution Mode", ["📄 Paper Trading (Simulated)", "⚡ Live Trading (Real Money)"])

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Strategy Parameters")
INITIAL_CAPITAL = st.sidebar.number_input("Starting Capital (₹)", value=50000)
STOP_LOSS_PCT = st.sidebar.slider("Stop Loss (%)", 1, 10, 6) / 100.0
TARGET_PCT = st.sidebar.slider("Target Profit (%)", 1, 25, 12) / 100.0

# Dynamic calculations for local capital display
current_capital = INITIAL_CAPITAL + st.session_state.session_pnl

# ==============================================================================
# MAIN DASHBOARD INTERFACE
# ==============================================================================
st.title("🦅 Upstox Options Advanced Scalping Engine")

# Row 1: Key Performance Metrics
m1, m2, m3, m4 = st.columns(4)

if ACCESS_TOKEN:
    live_index = fetch_ltp(ACCESS_TOKEN, "NSE_INDEX|Nifty 50") or 0.0
else:
    live_index = 0.0

m1.metric(label="📊 Nifty 50 Spot Index", value=f"₹{live_index:,.2f}" if live_index else "Disconnected")
m2.metric(label="💰 Current Account Balance", value=f"₹{current_capital:,.2f}")
m3.metric(label="📈 Today's Net PnL", value=f"₹{st.session_state.session_pnl:,.2f}", 
          delta=f"{((st.session_state.session_pnl/INITIAL_CAPITAL)*100):.2f}%" if INITIAL_CAPITAL else "0%")
m4.metric(label="🛡️ Operational Mode", value="LIVE" if "Live" in TRADE_MODE else "PAPER")

st.markdown("---")

# Row 2: Bot Core Operations Layout
col_ctrl, col_pos = st.columns([1, 2])

with col_ctrl:
    st.subheader("🕹️ Engine Telemetry")
    if not ACCESS_TOKEN:
        st.warning("Please provide a valid Upstox Access Token to activate systems.")
    else:
        if not st.session_state.bot_active:
            if st.button("▶️ START AUTO-BOT", type="primary", use_container_width=True):
                st.session_state.bot_active = True
                st.rerun()
        else:
            if st.button("🛑 EMERGENCY HALT / SHUTDOWN", type="secondary", use_container_width=True):
                st.session_state.bot_active = False
                st.session_state.current_position = None
                st.rerun()
        
        status_color = "🟢 Active & Scanning" if st.session_state.bot_active else "🔴 Core Switched Off"
        st.info(f"System State: **{status_color}**")

with col_pos:
    st.subheader("📦 Active Market Position")
    pos = st.session_state.current_position
    if pos:
        pos_ltp = fetch_ltp(ACCESS_TOKEN, pos['key']) or pos['entry_price']
        unrealized_pnl = (pos_ltp - pos['entry_price']) * 25 # Lot size 25
        pnl_color = "inverse" if unrealized_pnl < 0 else "normal"
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Symbol", pos['symbol'])
        c2.metric("Entry Price", f"₹{pos['entry_price']:.2f}")
        c3.metric("Live Price", f"₹{pos_ltp:.2f}")
        c4.metric("Unrealized PnL", f"₹{unrealized_pnl:.2f}")
        
        # Display underlying safeguards
        st.caption(f"🛡️ Target Price: **₹{pos['target_price']:.2f}** | Trailing SL Floor: **₹{pos['sl_price']:.2f}**")
    else:
        st.write("*No open positions found. The system is flat.*")

# ==============================================================================
# RUNNING ALGORITHMIC LOOP ENGINE
# ==============================================================================
if st.session_state.bot_active and ACCESS_TOKEN:
    # 1. Indicator processing block
    df = fetch_historical_candles(ACCESS_TOKEN, "NSE_INDEX|Nifty 50")
    
    if not df.empty and len(df) >= 30:
        # Tech Indicator Confluences
        df['EMA_9'] = ta.ema(df['close'], length=9)
        df['EMA_21'] = ta.ema(df['close'], length=21)
        df['Vol_SMA'] = ta.sma(df['volume'], length=20)
        df['RSI_14'] = ta.rsi(df['close'], length=14)
        df_adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        df['ADX'] = df_adx['ADX_14']
        df_st = ta.supertrend(df['high'], df['low'], df['close'], length=7, multiplier=3)
        df['ST_Direction'] = df_st['SUPERTd_7_3.0']
        
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        
        # 2. Check for Signals if Flat
        if st.session_state.current_position is None:
            ema_call_cross = (prev_row['EMA_9'] <= prev_row['EMA_21']) and (last_row['EMA_9'] > last_row['EMA_21'])
            ema_put_cross = (prev_row['EMA_9'] >= prev_row['EMA_21']) and (last_row['EMA_9'] < last_row['EMA_21'])
            
            is_trending = last_row['ADX'] > 25
            is_high_volume = last_row['volume'] > last_row['Vol_SMA']
            
            if ema_call_cross and is_trending and is_high_volume and last_row['ST_Direction'] == 1 and last_row['RSI_14'] > 50:
                # Simulated placeholder mapping for demo context. Replace key fetching logic in production
                st.session_state.current_position = {
                    "key": "NSE_OPTION|NIFTY24MAY24200CE", "symbol": "NIFTY 24200 CE",
                    "entry_price": 100.0, "highest_price": 100.0, "sl_price": 100.0 * (1 - STOP_LOSS_PCT), "target_price": 100.0 * (1 + TARGET_PCT)
                }
                st.success("🚨 CALL Entry Position triggered!")
                st.rerun()
                
            elif ema_put_cross and is_trending and is_high_volume and last_row['ST_Direction'] == -1 and last_row['RSI_14'] < 50:
                st.session_state.current_position = {
                    "key": "NSE_OPTION|NIFTY24MAY24200PE", "symbol": "NIFTY 24200 PE",
                    "entry_price": 100.0, "highest_price": 100.0, "sl_price": 100.0 * (1 - STOP_LOSS_PCT), "target_price": 100.0 * (1 + TARGET_PCT)
                }
                st.success("🚨 PUT Entry Position triggered!")
                st.rerun()

        # 3. Handle Active Trailing Monitoring & Streak Enforcement
        else:
            pos = st.session_state.current_position
            pos_ltp = fetch_ltp(ACCESS_TOKEN, pos['key'])
            
            if pos_ltp:
                # Trailing SL Mechanics
                if pos_ltp > pos['highest_price']:
                    gain = pos_ltp - pos['highest_price']
                    st.session_state.current_position['sl_price'] += gain
                    st.session_state.current_position['highest_price'] = pos_ltp
                
                # Check Exit Boundaries
                hit_sl = pos_ltp <= pos['sl_price']
                hit_target = pos_ltp >= pos['target_price']
                
                if hit_sl or hit_target:
                    trade_pnl = (pos_ltp - pos['entry_price']) * 25
                    st.session_state.session_pnl += trade_pnl
                    
                    # Log metrics to operational memory state
                    st.session_state.trade_logs.append({
                        "Timestamp": time.strftime("%H:%M:%S"), "Symbol": pos['symbol'],
                        "Mode": TRADE_MODE, "Entry": pos['entry_price'], "Exit": pos_ltp, "PnL": trade_pnl
                    })
                    
                    # Log structural data block to Excel automatically
                    try:
                        pd.DataFrame(st.session_state.trade_logs).to_excel("scalping_trade_logs.xlsx", index=False)
                    except Exception:
                        pass
                    
                    # STREAK RISK MANAGEMENT RULE
                    if trade_pnl <= 0:
                        st.session_state.bot_active = False
                        st.error("🚨 Streak broken by a loss! Safety circuit breaker tripped. System deactivated.")
                    
                    st.session_state.current_position = None
                    st.rerun()

    # Dynamic refreshing trick inside Streamlit context for 2-second telemetry refreshes
    time.sleep(2)
    st.rerun()

# ==============================================================================
# Row 3: HISTORICAL SESSION LOGS DISPLAY
# ==============================================================================
st.markdown("---")
st.subheader("📑 Session Trade Analytics (Auto-Logged to Excel)")
if st.session_state.trade_logs:
    st.dataframe(pd.DataFrame(st.session_state.trade_logs), use_container_width=True)
else:
    st.info("No trades executed in this specific dashboard UI loop container yet.")
