# app/ml_logic.py

import pandas as pd
import numpy as np
import requests
import pandas_ta as ta
import warnings
import os
from dotenv import load_dotenv
from datetime import datetime

# Loads the FMP_API_KEY from your .env file or Render environment
load_dotenv()
warnings.filterwarnings('ignore')

# --- Configuration for the Financial Modeling Prep (FMP) API ---
FMP_API_KEY = os.getenv("FMP_API_KEY")
BASE_URL = "https://financialmodelingprep.com/api/v3"

if not FMP_API_KEY:
    print("🔥🔥🔥 CRITICAL ERROR: FMP_API_KEY is not set in environment variables.")

def _format_symbol_for_fmp(symbol):
    """Converts the app's internal symbol format to FMP's format."""
    # Forex: EURUSD=X -> EURUSD
    if "=X" in symbol:
        return symbol.replace('=X', '')
    # Crypto: BTC-USD -> BTCUSD
    if "-USD" in symbol:
        return symbol.replace('-USD', 'USD')
    # Stocks and indices (e.g., AAPL, ^GSPC) are usually fine
    return symbol

def fetch_fmp_data(symbol, period='90d', interval='1h'):
    """Fetches historical data from the Financial Modeling Prep (FMP) API."""
    if not FMP_API_KEY:
        print("   ❌ FMP API key is missing. Cannot fetch data.")
        return None

    print(f"--- Starting FMP API fetch for {symbol} (Interval: {interval}) ---")

    api_symbol = _format_symbol_for_fmp(symbol)
    
    if interval in ['1h', '4h']:
        fmp_interval_string = '1hour' if interval == '1h' else '4hour'
        url = f"{BASE_URL}/historical-chart/{fmp_interval_string}/{api_symbol}"
        params = {'apikey': FMP_API_KEY}
    elif interval == '1m':
        url = f"{BASE_URL}/historical-chart/1min/{api_symbol}"
        params = {'apikey': FMP_API_KEY}
    elif interval in ['1d', '1wk']:
        url = f"{BASE_URL}/historical-price-full/{api_symbol}"
        params = {'apikey': FMP_API_KEY, 'from': '2000-01-01'}
    else:
        print(f"   ❌ Unsupported interval for FMP: {interval}")
        return None

    try:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()

        if isinstance(data, dict) and "Error Message" in data:
            error_message = data["Error Message"]
            print(f"   ⚠️ FMP API returned an error for {symbol}: {error_message}")
            return None

        if 'historical' in data:
            df = pd.DataFrame(data['historical'])
        elif isinstance(data, list) and len(data) > 0:
            df = pd.DataFrame(data)
        else:
            print(f"   ⚠️ FMP API returned no or unexpected data for {symbol}")
            return None

        if df.empty:
            print(f"   ⚠️ FMP API returned an empty DataFrame for {symbol}")
            return None
            
        df['datetime'] = pd.to_datetime(df['date'])
        df = df.set_index('datetime')
        
        df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
        
        ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in ohlcv_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col])
            
        df = df.iloc[::-1]

        if interval == '1wk':
            agg_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
            df = df.resample('W-FRI').agg(agg_dict).dropna()

        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df[[col for col in required_cols if col in df.columns]].dropna()

        print(f"   ✅ Success with FMP API for {symbol}! Got {len(df)} rows")
        return df

    except requests.exceptions.RequestException as e:
        print(f"   ❌ Network error fetching from FMP for {symbol}: {e}")
        return None
    except Exception as e:
        print(f"   ❌ Unexpected error processing FMP response for {symbol}: {e}")
        return None

def create_features_for_prediction(data, feature_columns_list):
    df = data.copy()
    if df.empty or len(df) < 20:
        return pd.DataFrame()
    try:
        df.ta.rsi(length=14, append=True)
        df.ta.ema(length=200, append=True)
        df.ta.atr(length=14, append=True)
        df.rename(columns={'RSI_14': 'rsi_14', 'ATRr_14': 'atr_14'}, inplace=True)
        
        channel_period = 20
        df['channel_high'] = df['High'].rolling(window=channel_period).max()
        df['channel_low'] = df['Low'].rolling(window=channel_period).min()
        df['channel_mid'] = (df['channel_high'] + df['channel_low']) / 2
        
        def get_slope(array):
            y = np.array(array)
            x = np.arange(len(y))
            slope, _ = np.polyfit(x, y, 1)
            return slope
        
        df['channel_slope'] = df['channel_mid'].rolling(window=channel_period).apply(get_slope, raw=False)
        df['channel_width'] = df['channel_high'] - df['channel_low']
        df['channel_width_atr'] = df['channel_width'] / df['atr_14'].replace(0, 1)
        df['bars_outside_zone'] = (df['Close'] > df['channel_high']).rolling(10).sum() + (df['Close'] < df['channel_low']).rolling(10).sum()
        
        df['breakout_distance'] = df['Close'] - df['channel_mid']
        df['breakout_distance_norm'] = df['breakout_distance'] / df['atr_14'].replace(0, 1)
        df['breakout_candle_body_ratio'] = abs(df['Close'] - df['Open']) / (df['High'] - df['Low']).replace(0, 1)
        
        df['price_vs_ema200'] = df['Close'] / df['EMA_200'].replace(0, 1)
        volume_ma = df['Volume'].rolling(20, min_periods=1).mean()
        df['volume_ratio'] = df['Volume'] / volume_ma.replace(0, 1)
        
        df['hour_of_day'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        
        df['risk_reward_ratio'] = 2.0
        # --- THIS IS THE CORRECTED LINE ---
        df['stop_loss_in_atrs'] = 1.5
        # ------------------------------------
        df['entry_pos_in_channel_norm'] = (df['Close'] - df['channel_low']) / df['channel_width'].replace(0, 1)

        channel_dev = (df['Close'] - df['channel_mid']) / df['channel_width'].replace(0, 1)
        for i in range(24):
            df[f'hist_close_channel_dev_t_minus_{i}'] = channel_dev.shift(i)

        df['volume_rsi_interaction'] = df['volume_ratio'] * df['rsi_14']
        df['breakout_strength'] = df['breakout_distance_norm'] * df['volume_ratio']
        df['channel_efficiency'] = df['channel_slope'] / df['channel_width_atr'].replace(0, 1)

        df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
        df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
        df['price_above_ema'] = (df['Close'] > df['EMA_200']).astype(int)
        df['high_risk_trade'] = ((df['rsi_14'] > 75) | (df['rsi_14'] < 25)).astype(int)
        
        df['trade_type_encoded'] = 0
        
        for col in feature_columns_list:
            if col not in df.columns:
                df[col] = 0.0
        
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        required_cols = feature_columns_list + ['Close', 'atr_14']
        available_cols = [col for col in required_cols if col in df.columns]
        
        return df[available_cols]
        
    except Exception as e:
        print(f"Error creating features: {e}")
        return pd.DataFrame()

def get_model_prediction(data, model, scaler, feature_columns):
    if data is None or data.empty:
        return {"error": "Cannot generate prediction, input data is missing."}
    try:
        features_df = create_features_for_prediction(data, feature_columns)
        if features_df.empty:
            return {"error": "Could not create features for prediction."}

        latest_features = features_df.iloc[-1].copy()
        last_price = latest_features['Close']
        latest_atr = latest_features.get('atr_14', 0)

        buy_features = latest_features.copy()
        buy_features['trade_type_encoded'] = 0
        sell_features = latest_features.copy()
        sell_features['trade_type_encoded'] = 1

        buy_df = pd.DataFrame([buy_features])[feature_columns]
        sell_df = pd.DataFrame([sell_features])[feature_columns]

        buy_scaled = scaler.transform(buy_df)
        sell_scaled = scaler.transform(sell_df)
        
        buy_prob = model.predict_proba(buy_scaled)[0][1]
        sell_prob = model.predict_proba(sell_scaled)[0][1]

        confidence_threshold = 0.55
        signal_type = "HOLD"
        confidence = max(buy_prob, sell_prob)

        if buy_prob > sell_prob and buy_prob > confidence_threshold:
            signal_type = "BUY"
            confidence = buy_prob
        elif sell_prob > buy_prob and sell_prob > confidence_threshold:
            signal_type = "SELL"
            confidence = sell_prob
        else:
            confidence = 0.5

        return {
            "signal": signal_type,
            "confidence": confidence,
            "latest_price": last_price,
            "latest_atr": latest_atr,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

def fetch_data_via_proxies(symbol, period='90d', interval='1h'):
    """Alias for backward compatibility in helpers.py."""
    return fetch_fmp_data(symbol, period, interval)
