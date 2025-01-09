import streamlit as st
from core.collect import MarketData
from core.features import FeatureEngineer
from core.model import HMMMarketCycle
from core.trading import TradingStrategy
from core.report import create_streamlit_app
import pandas as pd
from pathlib import Path


# extract symbols from data directory
def get_symbols_from_directory():
    historical_dir = Path("data/historical")
    return [f.stem.replace("historical_", "") for f in historical_dir.glob("historical_*.csv")]


def main():
    st.title("Crypto Market Analysis")
    
    # ensure we have data
    collector = MarketData()
    if not get_symbols_from_directory():
        found_pairs = len(collector.get_symbols())
        with st.spinner(f'Collecting historical data for {found_pairs} pairs...'):
            st.write("Initial data collection might take a while...")
            collector.get_historical_data()
            st.success("Data collection complete!")

    # get available symbols
    available_symbols = get_symbols_from_directory()
    if not available_symbols:
        st.error("No data files found in data/historical directory")
        return

    # initialize processing components
    engineer = FeatureEngineer()
    model = HMMMarketCycle()
    simulator = TradingStrategy()

    # process selected symbol
    selected_symbol = st.selectbox("Select symbol to analyze:", available_symbols)
    
    if selected_symbol:
        try:
            with st.spinner(f"Processing {selected_symbol}..."):
                # load and process data
                df = pd.read_csv(f"data/historical/historical_{selected_symbol}.csv", 
                               index_col='timestamp', parse_dates=True)
                
                # process single symbol
                features = engineer.calculate_features(df)
                train_data, test_data = engineer.prepare_train_test(df)
                model.train(train_data)
                predictions = model.predict(test_data)
                
                # get the corresponding price data for the test period
                test_price_data = df.loc[test_data.index]
                
                # fixed backtest call using price data
                portfolio_values, trade_history = simulator.backtest(
                    test_price_data,  # original price data for test period
                    predictions  # predictions DataFrame
                )
                
                # calculate buy & hold values for comparison
                initial_price = test_price_data['close'].iloc[0]
                buy_hold_values = pd.Series(
                    test_price_data['close'] * simulator.trading_config.initial_balance / initial_price,
                    index=test_price_data.index
                )
                
                metrics = simulator.calculate_metrics(portfolio_values, buy_hold_values)

                # create report with single symbol data
                create_streamlit_app(
                    feature_dfs={selected_symbol: features},
                    price_data={selected_symbol: df},
                    test_data={selected_symbol: test_data},
                    predictions={selected_symbol: predictions},
                    results={selected_symbol: {
                        'portfolio_values': portfolio_values,
                        'buy_hold_values': buy_hold_values
                    }},
                    metrics={selected_symbol: metrics}
                )

        except Exception as e:
            st.error(f"Error processing {selected_symbol}: {str(e)}")


if __name__ == "__main__":
    main()
