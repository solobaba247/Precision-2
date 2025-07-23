# app/routes.py

from flask import current_app, request, jsonify
from functools import wraps
import concurrent.futures
import os
from .ml_logic import get_model_prediction, fetch_fmp_data
from .helpers import calculate_stop_loss_value

# --- MODIFIED: API Key Authentication from URL Parameter ---
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        secret_key = os.getenv("SECRET_API_KEY")
        if not secret_key:
            return jsonify({"error": "API service is not configured."}), 500
        
        # CHANGED: Get the API key from the URL query parameter `?apikey=`
        api_key = request.args.get('apikey')
        
        if not api_key or api_key != secret_key:
            return jsonify({"error": "Unauthorized. Invalid or missing API Key in URL."}), 401
        
        return f(*args, **kwargs)
    return decorated_function

# --- The rest of the file remains the same ---

def _get_and_format_signal(symbol, timeframe):
    try:
        data = fetch_fmp_data(symbol, period='90d', interval=timeframe)
        if data is None or len(data) < 50:
            return {"error": f"Insufficient data for {symbol}. Need at least 50 data points."}

        prediction = get_model_prediction(
            data,
            current_app.model,
            current_app.scaler,
            current_app.feature_columns
        )
        if "error" in prediction:
            return prediction

        signal = prediction['signal']
        confidence = prediction['confidence']
        latest_price = prediction['latest_price']
        latest_atr = prediction['latest_atr']
        
        atr_multiplier_sl = 1.5
        atr_multiplier_tp = 3.0
        
        if signal == "BUY":
            entry_price = latest_price
            stop_loss = latest_price - (atr_multiplier_sl * latest_atr)
            exit_price = latest_price + (atr_multiplier_tp * latest_atr)
        elif signal == "SELL":
            entry_price = latest_price
            stop_loss = latest_price + (atr_multiplier_sl * latest_atr)
            exit_price = latest_price - (atr_multiplier_tp * latest_atr)
        else:
            entry_price, stop_loss, exit_price = latest_price, latest_price, latest_price

        return {
            "symbol": symbol,
            "signal": signal,
            "confidence": f"{confidence:.2%}",
            "entry_price": f"{entry_price:.5f}",
            "exit_price": f"{exit_price:.5f}",
            "stop_loss": f"{stop_loss:.5f}",
            "stop_loss_value": calculate_stop_loss_value(symbol, entry_price, stop_loss),
            "timestamp": prediction['timestamp']
        }
    except Exception as e:
        print(f"Error in _get_and_format_signal for {symbol}: {e}")
        return {"error": f"Failed to generate signal for {symbol}: {str(e)}"}

@current_app.route('/')
def root():
    return jsonify({
        "status": "online",
        "message": "Welcome to the Trading Signal API. Use the /v1 endpoints with an 'apikey' parameter to access signals.",
        "model_status": "loaded" if current_app.config.get('MODELS_LOADED', False) else "error"
    })

@current_app.route('/v1/signal')
@require_api_key
def generate_signal_route():
    symbol = request.args.get('symbol')
    timeframe = request.args.get('timeframe', '1h')

    if not symbol:
        return jsonify({"error": "Symbol parameter is required."}), 400
    
    response = _get_and_format_signal(symbol, timeframe)
    
    if "error" in response:
        return jsonify(response), 500
    
    return jsonify(response)


def get_prediction_for_symbol_sync(symbol, timeframe):
    result = _get_and_format_signal(symbol, timeframe)
    if result and "error" not in result and result.get("signal") in ["BUY", "SELL"]:
        return result
    return None

@current_app.route('/v1/scan', methods=['POST'])
@require_api_key
def scan_market_route():
    try:
        data = request.get_json()
        if not data or 'symbols' not in data:
            return jsonify({"error": "Request body must be JSON and contain a 'symbols' list."}), 400

        symbols_to_scan = data.get('symbols')
        timeframe = data.get('timeframe', '1h')
        
        if not isinstance(symbols_to_scan, list):
            return jsonify({"error": "'symbols' must be a list of asset tickers."}), 400
            
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_symbol = {
                executor.submit(get_prediction_for_symbol_sync, symbol, timeframe): symbol for symbol in symbols_to_scan
            }
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol_name = future_to_symbol[future]
                try:
                    result = future.result(timeout=60)
                    if result:
                        results.append(result)
                except concurrent.futures.TimeoutError:
                    print(f"⏰ Timeout processing {symbol_name} after 60 seconds. Skipping.")
                except Exception as e:
                    print(f"Error processing {symbol_name}: {e}")

        results.sort(key=lambda x: float(x['confidence'].strip('%')), reverse=True)
        
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": f"Failed to scan market: {str(e)}"}), 500
