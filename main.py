from core.collect import Collector
from core.features import FeatureEngineer
from core.model import HMMMarketCycle
from core.backtest import BacktestStrategy, BacktestConfig
import pandas as pd
import matplotlib.pyplot as plt
import time
import os

def debug_print(msg: str, data=None):
    """Helper function for debug printing"""
    print("\n" + "="*20 + f" {msg} " + "="*20)
    if data is not None:
        print(data)

def run_full_pipeline():
    debug_print("STARTING PIPELINE")
    start_time = time.time()

    debug_print("Collecting Data")
    collector = Collector()
    collection_results = collector.collect_batch(batch_size=50, delay=1)
    
    if collection_results is None or collection_results['candidates'].empty:
        debug_print("No candidates found")
        return
    
    debug_print("Collection Complete", 
                f"Found {len(collection_results['candidates'])} candidates")
    
    for symbol in collection_results['candidates'].index:
        try:
            debug_print(f"Processing {symbol}")
            
            data = collector.get_latest_data(symbol)
            if data is None or len(data) < 100:
                debug_print(f"Insufficient data for {symbol}")
                continue
                
            debug_print("Data loaded", 
                       f"Shape: {data.shape}, Date range: {data.index[0]} to {data.index[-1]}")

            debug_print("Feature Engineering")
            feature_eng = FeatureEngineer()
            train_features, test_features = feature_eng.prepare_train_test(data)
            debug_print("Features created", 
                       f"Train shape: {train_features.shape}, Test shape: {test_features.shape}")
            
            debug_print("Training Model")
            model = HMMMarketCycle()
            train_results = model.train(train_features)
            debug_print("Model trained", f"Log likelihood: {train_results['log_likelihood']:.2f}")
            
            debug_print("Generating Predictions")
            predictions = model.predict(test_features)
            debug_print("Predictions generated", f"Shape: {predictions.shape}")
            
            debug_print("Running Backtest")
            config = BacktestConfig(
                initial_balance=10000.0,
                position_size=0.1,
                stop_loss=0.02,
                take_profit=0.03
            )
            
            backtest = BacktestStrategy(config)
            results = backtest.run_backtest(data[-len(test_features):], predictions)
            
            debug_print("Saving Results")
            os.makedirs("results", exist_ok=True)

            os.makedirs("results/portfolio", exist_ok=True)
            results['portfolio'].to_csv(f"results/portfolio/{symbol}_portfolio.csv")
            os.makedirs("results/predictions", exist_ok=True)
            predictions.to_csv(f"results/predictions/{symbol}_predictions.csv")
            
            perf = results['performance']
            debug_print("Backtest Results", {
                'Total Return': f"{perf['total_return']:.2%}",
                'Sharpe Ratio': f"{perf['sharpe_ratio']:.2f}",
                'Max Drawdown': f"{perf['max_drawdown']:.2%}",
                'Win Rate': f"{perf['win_rate']:.2%}",
                'Total Trades': perf['total_trades']
            })
            
            plot_results(results['portfolio'], symbol)
            
        except Exception as e:
            debug_print(f"Error processing {symbol}", str(e))
            import traceback
            traceback.print_exc()

    end_time = time.time()
    debug_print("Pipeline Complete", f"Total Runtime: {end_time - start_time:.2f} seconds")

def plot_results(portfolio: pd.DataFrame, symbol: str):
    plt.figure(figsize=(15, 8))
    
    # Plot portfolio value and buy & hold
    plt.plot(portfolio.index, portfolio['value'], label='Strategy', linewidth=2)
    plt.plot(portfolio.index, portfolio['buy_hold'], label='Buy & Hold', linewidth=2, alpha=0.7)
    
    # Mark trades
    entries = portfolio[portfolio['trade'] == 'entry'].index
    exits = portfolio[portfolio['trade'] == 'exit'].index
    
    plt.scatter(entries, portfolio.loc[entries, 'value'], 
               marker='^', color='g', label='Entry', s=100)
    plt.scatter(exits, portfolio.loc[exits, 'value'], 
               marker='v', color='r', label='Exit', s=100)
    
    plt.title(f'{symbol} Trading Performance')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (USDT)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_full_pipeline()