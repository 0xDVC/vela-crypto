from core.collect import Collector
from core.features import FeatureEngineer
from core.model import HMMMarketCycle
from core.backtest import BacktestStrategy, BacktestConfig


def debug_print(msg: str, data=None):
    """Helper function for debug printing"""
    print("\n" + "="*20 + f" {msg} " + "="*20)
    if data is not None:
        print(data)

def prepare_dashboard_data():
    """Prepare data for the Streamlit dashboard"""
    debug_print("Preparing Dashboard Data")
    
    symbols_data = {}
    predictions_data = {}
    backtest_results = {}
    
    collector = Collector()
    collection_results = collector.collect_batch(batch_size=50, delay=1)
    
    if collection_results is None or collection_results['candidates'].empty:
        debug_print("No candidates found")
        return None, None, None
    
    feature_eng = FeatureEngineer()
    
    for symbol in collection_results['candidates'].index:
        try:
            data = collector.get_latest_data(symbol)
            if data is None or len(data) < 100:
                continue
                
            # Feature engineering
            train_features, test_features = feature_eng.prepare_train_test(data)
            
            if train_features is None or test_features is None:
                debug_print(f"Failed to prepare features for {symbol}")
                continue
            
            # Model training and prediction
            model = HMMMarketCycle()
            model.fit(train_features)
            predictions = model.predict(test_features)
            
            # Backtest
            config = BacktestConfig(
                initial_balance=10000.0,
                position_size=0.1,
                base_stop_loss=0.02,
                base_take_profit=0.03
            )
            
            backtest = BacktestStrategy(config)
            results = backtest.run_backtest(data[-len(test_features):], predictions)
            
            # Store results
            symbols_data[symbol] = data
            predictions_data[symbol] = predictions
            backtest_results[symbol] = results
            
        except Exception as e:
            debug_print(f"Error processing {symbol}", str(e))
            continue
    
    return symbols_data, predictions_data, backtest_results

if __name__ == "__main__":
    # For running the Streamlit dashboard
    symbols_data, predictions, backtest_results = prepare_dashboard_data()