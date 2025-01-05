from dataclasses import dataclass
from binance.client import Client
from joblib import Memory
import pandas as pd
import requests
import os
import json
import certifi

@dataclass
class MarketFilters:
    """market filtering criteria"""
    max_days: int = 90  # 3 months
    min_volume: int = 50_000  # 50k
    max_volume: int = 10_000_000  # 10M
    min_market_cap: int = 1_000_000  # 1M
    max_market_cap: int = 100_000_000  # 100M


class Collector:
    """binance data collection and filtering"""
    
    VALID_INTERVALS = ['1m', '3m', '5m', '15m', '30m',
                       '1h', '2h', '4h', '6h', '8h', '12h',
                       '1d', '3d', '1w', '1M']
    
    def __init__(self, cache_dir="./.cache", data_dir="./data"):
        self.cache_dir = cache_dir
        self.data_dir = data_dir
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        
        self.client = Client()
        self.memory = Memory(cache_dir, verbose=0)
        
        self.filters = MarketFilters()
        
        self.exchange_info = self.client.get_exchange_info()
        tickers = self.client.get_ticker()
        self.tickers_df = pd.DataFrame(tickers).set_index('symbol')
        self.tickers_df['quoteVolume'] = pd.to_numeric(self.tickers_df['quoteVolume'])

    @property
    def cached_get_klines(self):
        """klines getter with caching"""    # memoized approach
        @self.memory.cache
        def _get_klines(symbol, interval='15m', limit=100):
            if interval not in self.VALID_INTERVALS:
                raise ValueError(f"Invalid interval. Must be one of {self.VALID_INTERVALS}")
            return self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
        return _get_klines

    # def get_binance_usdt_pairs(self, max_market_cap=100_000_000):
    #     """fetch USDT token pairs with market cap below max_market_cap from binance."""
    #     url = "https://www.binance.com/bapi/apex/v1/friendly/apex/marketing/complianceSymbolList"
        
    #     try:
    #         response = requests.get(
    #             url,
    #             headers={"Accept-Encoding": "gzip"},
    #             verify=certifi.where()
    #         )
    #         response.raise_for_status()
    #         data = response.json()
            
    #         usdt_pairs = []
    #         for asset in data.get("data", []):
    #             symbol = asset.get("symbol", "")
    #             market_cap = asset.get("marketCap")
                
    #             if "USDT" in symbol and market_cap is not None:
    #                 if market_cap <= max_market_cap:
    #                     usdt_pairs.append(symbol)
            
    #         return set(usdt_pairs)
            
    #     except requests.RequestException as e:
    #         print(f"fetch error: {e}")
    #         return set()
    #     except json.JSONDecodeError:
    #         print("failed to parse json")
    #         return set()

    def filter_usdt_pairs(self):
        """filter USDT trading pairs based on volume and market cap."""
        url = "https://www.binance.com/bapi/apex/v1/friendly/apex/marketing/complianceSymbolList"
        
        try:
            response = requests.get(
                url,
                headers={"Accept-Encoding": "gzip"},
                verify=certifi.where()
            )
            response.raise_for_status()
            data = response.json()
            
            # let's create a dataframe with market cap data
            symbol_data = []
            for asset in data.get("data", []):
                symbol = asset.get("symbol", "")
                market_cap = asset.get("marketCap")
                
                
                if "USDT" in symbol and market_cap is not None:
                    symbol_data.append({
                        'symbol': symbol,
                        'marketCap': market_cap
                    })
            
            market_cap_df = pd.DataFrame(symbol_data).set_index('symbol')
            
            # get active trading pairs and filter tickers
            trading_pairs = set(s['symbol'] for s in self.exchange_info['symbols']
                              if s['symbol'].endswith('USDT') and s['status'] == 'TRADING')
            
            # include tickers that are in trading_pairs
            self.tickers_df = self.tickers_df[self.tickers_df.index.isin(trading_pairs)]
            
            # apply volume filter
            volume_mask = self.tickers_df['quoteVolume'].between(
                self.filters.min_volume,
                self.filters.max_volume
            )
            volume_filtered_df = self.tickers_df[volume_mask]
            
            # merge volume and market cap data
            combined_df = volume_filtered_df.join(market_cap_df)
            
            # Apply market cap filter
            market_cap_mask = combined_df['marketCap'].between(
                self.filters.min_market_cap,
                self.filters.max_market_cap
            )
            
            return combined_df[market_cap_mask]
            
        except requests.RequestException as e:
            print(f"fetch error: {e}")
            return pd.DataFrame()
        except json.JSONDecodeError:
            print("failed to parse json")
            return pd.DataFrame()

    def find_candidates(self, filtered_pairs):
        """find trading candidates based on age and market cap with data quality checks"""
        candidates_data = []

        symbols = filtered_pairs.index
        
        for symbol in symbols:
            try:
                # get historical data
                klines = self.cached_get_klines(symbol, "1d", 90)
                
                # data quality checks
                if len(klines) < 7:  # minimum 7 days of data
                    print(f"Skipping {symbol}: Insufficient historical data")
                    continue
                    
                # convert klines to DataFrame with all columns
                df = pd.DataFrame(klines, columns=[
                   'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_volume', 'trades',
                        'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'
                ])
                
                # convert numeric columns
                numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
                df[numeric_cols] = df[numeric_cols].astype(float)
                
                # check for gaps in data
                if df['volume'].isna().any() or (df['volume'] == 0).any():
                    print(f"Skipping {symbol}: Contains missing or zero volume data")
                    continue
                    
                # check for consistent trading (volume stability)
                vol_mean = df['volume'].mean()
                vol_std = df['volume'].std()
                if vol_std / vol_mean > 3:  # coefficient of variation > 3
                    print(f"Skipping {symbol}: Highly irregular trading volume")
                    continue

                market_cap = filtered_pairs.loc[symbol, 'marketCap']
                volume_24h = filtered_pairs.loc[symbol, 'quoteVolume']

                candidates_data.append({
                    'symbol': symbol,
                    'days_listed': len(klines),
                    'market_cap': market_cap,
                    'volume_24h': volume_24h,
                    'avg_daily_volume': vol_mean,
                    'volume_stability': vol_std / vol_mean
                })

            except Exception as e:
                print(f"Error processing {symbol}: {str(e)}")

        if candidates_data:
            return pd.DataFrame(candidates_data).set_index('symbol')
        else:
            return pd.DataFrame(columns=['days_listed', 'market_cap', 'volume_24h', 
                                       'avg_daily_volume', 'volume_stability'])

    def get_latest_data(self, symbol: str) -> pd.DataFrame:
        """get latest historical data from cache if available"""
        files = [f for f in os.listdir(self.data_dir) 
                 if f.startswith(f'historical_{symbol}_')]
        
        if not files:
            return None
        
        latest_file = sorted(files)[-1]
        return pd.read_csv(
            f"{self.data_dir}/{latest_file}", 
            index_col='timestamp', 
            parse_dates=True
        )

    def fetch_historical_data(self, candidates_df, interval="15m"):
        """fetch historical price data for candidates with caching"""
        historical_data = {}

        for symbol in candidates_df.index:
            try:
                cached_df = self.get_latest_data(symbol)
                
                if cached_df is not None:
                    last_timestamp = cached_df.index[-1]
                    new_klines = self.cached_get_klines(
                        symbol=symbol,
                        interval=interval,
                        limit=100  
                    )
                    
                    new_df = pd.DataFrame(new_klines, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_volume', 'trades',
                        'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'
                    ])
                    
                    new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], unit='ms')
                    new_df.set_index('timestamp', inplace=True)
                    
                    numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                                     'quote_volume', 'trades', 'taker_buy_volume', 
                                     'taker_buy_quote_volume']
                    for col in numeric_columns:
                        new_df[col] = pd.to_numeric(new_df[col])
                    
                    historical_data[symbol] = pd.concat([
                        cached_df,
                        new_df[new_df.index > last_timestamp]
                    ]).drop_duplicates()
                    
                else:
                    klines = self.cached_get_klines(
                        symbol=symbol,
                        interval=interval,
                        limit=self.filters.max_days * 24
                    )
                    
                    df = pd.DataFrame(klines, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_volume', 'trades',
                        'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'
                    ])
                    
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    
                    numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                                     'quote_volume', 'trades', 'taker_buy_volume', 
                                     'taker_buy_quote_volume']
                    for col in numeric_columns:
                        df[col] = pd.to_numeric(df[col])
                            
                    historical_data[symbol] = df

            except Exception as e:
                print(f"Error fetching history for {symbol}: {str(e)}")

        return historical_data

    def collect_batch(self, batch_size=50, delay=1):
        """run full data collection process in batches and save results"""
        os.makedirs('./data', exist_ok=True)
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        
        filtered_pairs = self.filter_usdt_pairs()
        print(f"found {len(filtered_pairs)} pairs meeting volume criteria")
        
        os.makedirs('./data/pairs', exist_ok=True)
        filtered_pairs.to_csv(f'./data/pairs/filtered_pairs_{timestamp}.csv')

        candidates_df = self.find_candidates(filtered_pairs)
        print(f"found {len(candidates_df)} candidates meeting all criteria")

        os.makedirs('./data/candidates', exist_ok=True)
        candidates_df.to_csv(f'./data/candidates/candidates_{timestamp}.csv')

        symbols = candidates_df.index.tolist()
        
        for i in range(0, len(symbols), batch_size):
            batch_symbols = symbols[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1} ({len(batch_symbols)} symbols)")
            
            batch_candidates = candidates_df.loc[batch_symbols]
            historical_data = self.fetch_historical_data(batch_candidates)
            
            for symbol, df in historical_data.items():
                os.makedirs('./data/historical', exist_ok=True)
                df.to_csv(f'./data/historical/historical_{symbol}_{timestamp}.csv')
                
            
            if i + batch_size < len(symbols):
                time.sleep(delay)
        
        return {
            "candidates": candidates_df,
            "timestamp": timestamp
        }


if __name__ == '__main__':
    import time
    
    print("Starting data collection...")
    start_time = time.time()
    
    # initialize collector
    collector = Collector(
        cache_dir="./.cache",
        data_dir="./data"
    )
    
    # run batch collection with progress tracking
    results = collector.collect_batch(batch_size=50, delay=1)
    
    # print summary
    end_time = time.time()
    duration = end_time - start_time
    
    print("\ncollection:")
    print(f"total time: {duration:.2f} seconds")
    print(f"timestamp: {results['timestamp']}")
    print(f"total candidates: {len(results['candidates'])}")
    
    # check sample candidates
    candidates_df = results['candidates']
    if not candidates_df.empty:
        print("\nsample:")
        print(candidates_df.head())
        
        # summary stats
        summary_stats = {
            'total_pairs': len(candidates_df),
            'avg_market_cap': candidates_df['market_cap'].mean(),
            'avg_volume': candidates_df['volume_24h'].mean(),
            'avg_age': candidates_df['days_listed'].mean()
        }
        
        print("\mrkt stats:")
        for key, value in summary_stats.items():
            print(f"{key}: {value:,.2f}")
        