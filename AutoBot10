import time
import requests
import pandas as pd
import pandas_ta as ta
import upstox_client
from upstox_client.rest import ApiException

# ==============================================================================
# 1. CONFIGURATION & ACCESS TOKEN
# ==============================================================================
ACCESS_TOKEN = "YOUR_UPSTOX_ACCESS_TOKEN"  # Replace with valid token

# Trading Settings
UNDERLYING_INDEX = "NSE_INDEX|Nifty 50" 
TIMEFRAME = "1minute"                   
LOT_SIZE = 25                           # Current Nifty lot size
TRADE_LOTS = 1                          

# Strict Risk Management Parameters (Based on ₹50,000 Capital)
STOP_LOSS_PERCENT = 0.06                # Strict 6% initial stop loss
TARGET_PERCENT = 0.12                   # Strict 12% profit target (1:2 Risk-Reward)

# Initial State Variables
current_position = None  
bot_active = True                       # Controls global execution line

# Upstox API Clients Setup
config = upstox_client.Configuration()
config.access_token = ACCESS_TOKEN
api_client = upstox_client.ApiClient(config)
order_api = upstox_client.OrderApi(api_client)     
market_api = upstox_client.HistoryApi(api_client)  

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================
def get_historical_candles():
    """Fetches real-time historical data from Upstox to calculate technicals."""
    try:
        response = market_api.get_historical_candle_data_v3(
            instrument_key=UNDERLYING_INDEX,
            interval=TIMEFRAME,
            to_date=time.strftime("%Y-%m-%d")
        )
        candles = response.data.candles
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
        df = df.iloc[::-1].reset_index(drop=True) 
        df['close'] = pd.to_numeric(df['close'])
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        df['volume'] = pd.to_numeric(df['volume'])
        return df
    except Exception as e:
        print(f"Error fetching candle data: {e}")
        return pd.DataFrame()

def fetch_atm_strike(current_spot):
    """Dynamically calculates the At-The-Money (ATM) contract strike."""
    base = 50 
    return int(base * round(current_spot / base))

def fetch_option_contract_key(strike, option_type):
    """Retrieves exact Upstox Instrument Key for active contracts using Option Chain API."""
    try:
        url = "https://upstox.com"
        headers = {"Accept": "application/json", "Authorization": f"Bearer {ACCESS_TOKEN}"}
        params = {"instrument_key": UNDERLYING_INDEX, "expiry_date": time.strftime("%Y-%m-%d")} 
        
        response = requests.get(url, params=params, headers=headers).json()
        chain_data = response.get('data', [])
        
        for item in chain_data:
            if int(item.get('strike_price')) == strike:
                return item.get(option_type.lower(), {}).get('instrument_key')
    except Exception as e:
        print(f"Error resolving option chain keys: {e}")
    return None

def place_market_order(instrument_key, transaction_type, quantity):
    """Executes instant market orders via Upstox Execution Engine."""
    try:
        body = {
            "quantity": quantity,
            "product": "I",              # Intraday MIS
            "validity": "DAY",
            "price": 0.0,                # Market Order
            "tag": "ScalpBot",
            "instrument_key": instrument_key,
            "order_type": "MARKET",
            "transaction_type": transaction_type
        }
        api_response = order_api.place_order(body, api_version="v2")
        print(f"Order Executed! Type: {transaction_type} | Order ID: {api_response.data.order_id}")
        return api_response.data.order_id
    except ApiException as e:
        print(f"Critical Execution Error: {e.body}")
        return None

def fetch_ltp(instrument_key):
    """Retrieves Last Traded Price (LTP)."""
    try:
        url = f"https://upstox.com{instrument_key}"
        headers = {"Accept": "application/json", "Authorization": f"Bearer {ACCESS_TOKEN}"}
        res = requests.get(url, headers=headers).json()
        return res['data'][instrument_key]['last_price']
    except:
        return None

# ==============================================================================
# 3. CORE BOT LOOP LOGIC
# ==============================================================================
print("Initializing Upstox Scalping Bot Engine...")

while bot_active:
    df = get_historical_candles()
    if df.empty or len(df) < 30:
        time.sleep(10)
        continue

    # 1. Advanced Indicator Suite Calculations
    df['EMA_9'] = ta.ema(df['close'], length=9)
    df['EMA_21'] = ta.ema(df['close'], length=21)
    df['Vol_SMA'] = ta.sma(df['volume'], length=20)
    df['RSI_14'] = ta.rsi(df['close'], length=14)
    
    df_adx = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['ADX'] = df_adx['ADX_14']
    
    # Supertrend calculation (Length: 7, Multiplier: 3)
    df_st = ta.supertrend(df['high'], df['low'], df['close'], length=7, multiplier=3)
    df['ST_Direction'] = df_st['SUPERTd_7_3.0'] # Returns 1 for Bullish, -1 for Bearish
    
    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    spot_price = last_row['close']

    # 2. Multi-Filter Signal Entry Verification
    if current_position is None:
        ema_call_cross = (prev_row['EMA_9'] <= prev_row['EMA_21']) and (last_row['EMA_9'] > last_row['EMA_21'])
        ema_put_cross = (prev_row['EMA_9'] >= prev_row['EMA_21']) and (last_row['EMA_9'] < last_row['EMA_21'])
        
        # Validation Pipeline
        is_trending = last_row['ADX'] > 25
        is_high_volume = last_row['volume'] > last_row['Vol_SMA']
        is_st_bullish = last_row['ST_Direction'] == 1
        is_st_bearish = last_row['ST_Direction'] == -1
        
        # Combined Confluence Triggers
        call_signal = ema_call_cross and is_trending and is_high_volume and is_st_bullish and (last_row['RSI_14'] > 50)
        put_signal = ema_put_cross and is_trending and is_high_volume and is_st_bearish and (last_row['RSI_14'] < 50)

        if call_signal or put_signal:
            strike = fetch_atm_strike(spot_price)
            opt_type = 'CE' if call_signal else 'PE'
            print(f"Confluence Signal Verified! Spot: {spot_price} | ATM: {strike} {opt_type} | RSI: {last_row['RSI_14']:.1f}")
            
            target_key = fetch_option_contract_key(strike, opt_type)
            if target_key:
                entry_price = fetch_ltp(target_key)
                if entry_price:
                    order_id = place_market_order(target_key, "BUY", TRADE_LOTS * LOT_SIZE)
                    if order_id:
                        current_position = {
                            "key": target_key,
                            "entry_price": entry_price,
                            "highest_price": entry_price, # Tracks trailing mechanics
                            "sl_price": entry_price * (1 - STOP_LOSS_PERCENT),
                            "target_price": entry_price * (1 + TARGET_PERCENT)
                        }
                        print(f"Position Open: Entry ₹{entry_price:.2f} | SL: ₹{current_position['sl_price']:.2f} | Target: ₹{current_position['target_price']:.2f}")

    # 3. Dynamic Trailing Monitoring & Streak Management
    elif current_position is not None:
        ltp = fetch_ltp(current_position['key'])
        if ltp:
            # Trailing Stop-Loss Engine (1:1 Trailing Mechanic)
            if ltp > current_position['highest_price']:
                price_gain = ltp - current_position['highest_price']
                current_position['sl_price'] += price_gain # Move SL up by the exact gain amount
                current_position['highest_price'] = ltp    # Update highest observed price boundary
                print(f" Trailing SL Adjusted Upwards! New SL: ₹{current_position['sl_price']:.2f}")

            # Exit Boundaries Check
            hit_sl = ltp <= current_position['sl_price']
            hit_target = ltp >= current_position['target_price']

            if hit_sl or hit_target:
                exit_reason = "TARGET HIT" if hit_target else "STOP LOSS / TRAILING SL HIT"
                print(f"Exiting Position: {exit_reason} | Current Price: ₹{ltp:.2f}")
                
                place_market_order(current_position['key'], "SELL", TRADE_LOTS * LOT_SIZE)
                
                trade_pnl = (ltp - current_position['entry_price']) * (TRADE_LOTS * LOT_SIZE)
                print(f"Trade Closed. PnL: ₹{trade_pnl:.2f}")
                
                # Streak Logic Evaluation
                if trade_pnl > 0:
                    print("Winning trade! Resetting logic to wait for next setup...")
                    current_position = None  
                else:
                    print("Loss encountered. Capital protection filter engaged. Shutting down bot for today.")
                    bot_active = False       

    time.sleep(2)
