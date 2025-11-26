import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np

def fetch_crypto_data(ticker="BTC-USD", period="2y", interval="1d"):
    """
    Fetches historical data.
    """
    print(f"📥 Fetching {ticker} data ({period}, {interval})...")
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        
        # Handle MultiIndex columns (yfinance update)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        if df.empty:
            raise ValueError("Empty dataset.")
            
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        print(f"✅ Loaded {len(df)} rows.")
        return df
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return pd.DataFrame()

def add_technical_indicators(df):
    """
    Calculates Log-Returns and Technical Indicators.
    Returns: (df_features, df_close_prices)
    """
    df = df.copy()
    
    # 1. Target: Log Returns (Stationary Time Series)
    # ln(Price_t / Price_t-1)
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # 2. Log Volume
    df['Log_Vol'] = np.log(df['Volume'] + 1)
    
    # 3. Technical Indicators
    df['RSI'] = df.ta.rsi(length=14)
    macd = df.ta.macd(fast=12, slow=26, signal=9)
    df = pd.concat([df, macd], axis=1)
    
    bbands = df.ta.bbands(length=20, std=2)
    df = pd.concat([df, bbands], axis=1)
    
    # Drop NaN values (first ~26 rows are NaN due to MACD)
    df.dropna(inplace=True)
    
    # Cleanup Column Names
    cols_to_rename = {}
    for col in df.columns:
        if 'MACD_' in col and 'h' not in col and 's' not in col: cols_to_rename[col] = 'MACD'
        elif 'MACDs_' in col: cols_to_rename[col] = 'MACD_Signal'
        elif 'BBU_' in col: cols_to_rename[col] = 'BB_Upper'
        elif 'BBL_' in col: cols_to_rename[col] = 'BB_Lower'
    
    df.rename(columns=cols_to_rename, inplace=True)
    
    # Select final features
    # We do NOT include raw 'Close' in features, only returns and indicators
    feature_cols = ['Log_Ret', 'Log_Vol', 'RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower']
    
    # Return features AND the raw close prices (needed for rebuilding price later)
    return df[feature_cols], df['Close']

def create_sequences(data, seq_length=60):
    """
    Creates (X, y) pairs for LSTM.
    """
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length, 0] # Index 0 is Log_Ret
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)