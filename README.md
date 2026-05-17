
Overview — what the app does in one paragraph
Architecture — the five layers (constants → state → API helpers → sidebar → algo loop) in a table
API Layer — each of the five helper functions explained individually, including the | → : key normalisation quirk and the batch-50 pattern
Trading Strategy — the six-factor confluence model (EMA cross, ADX, Volume, Supertrend, RSI, VWAP) laid out as a CE vs PE comparison table
Position Management — entry gate, trailing SL ratchet logic, and exit conditions
Risk Management — circuit breaker mechanics and all tunable parameters with defaults
Session Trade Log — what's captured per trade and the CSV download approach (and why not disk write)
Installation — pip install, local run, Streamlit Cloud deploy, Python version caveat for pandas-ta
Known Limitations — instrument key resolution, no rate-limit retry, no OI data, no order routing wired in
