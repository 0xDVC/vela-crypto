from dataclasses import dataclass
from typing import List
from binance.client import Client
from joblib import Memory
import requests
import certifi
import pandas as pd
from tqdm import tqdm
import time
import os



@dataclass
class MarketFilters():
    MIN_MARKET_CAP: float = 1_000_000.00  # 1M USDT min
    MAX_MARKET_CAP: float = 100_000_000.00  # 100M USDT max
    MIN_VOLUME: float = 50_000.00  # Minimum daily volume
    KLINE_INTERVAL: str = "15m"  # For short-term analysis
    DAILY_BARS: int = 96  # 96 bars per day
    BATCH_SIZE: int = 50
    SLEEP_TIME: int = 5


class MarketData():
    """Collect market data from Binance"""
    def __init__(self):
        self.filters = MarketFilters()
        self.client = Client()
        self.memory = Memory(location='.cache', verbose=0)


    def cached_get_klines(self, symbol, interval="15m") -> List[List[float]]:
        """Cache historical data requests"""
        # Memoize the function to cache the results
        @self.memory.cache
        def _get_klines(symbol, interval):
            """Get klines from Binance API"""
            end_time = int(pd.Timestamp.now().timestamp() * 1000)
            start_time = int((pd.Timestamp.now() - pd.Timedelta(days=97)).timestamp() * 1000)
            all_klines = []
            current_start = start_time
            
            while current_start < end_time:
                klines = self.client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=1000,
                    startTime=current_start,
                    endTime=end_time
                )
                
                if not klines:
                    break
                    
                all_klines.extend(klines)
                current_start = klines[-1][0] + 1
                
            return all_klines

        # call the cached function with the parameters
        return _get_klines(symbol=symbol, interval=interval)

    def get_symbols(self) -> List[str]:
        url = "https://www.binance.com/bapi/apex/v1/friendly/apex/marketing/complianceSymbolList"

        try:
            # get market caps
            response = requests.get(
                        url, headers={"Accept-Encoding": "gzip"}, verify=certifi.where()
                    )
            data = response.json()

            volumes = {  t["symbol"]: float(t["quoteVolume"]) for t in self.client.get_ticker() if t["symbol"].endswith("USDT") }

            # find valid coins
            small_coins = []
            for token in data.get("data", []):
                symbol = token.get("symbol", "")
                mcap = token.get("marketCap")
                vol = volumes.get(symbol, 0)

                    # check market cap and volume criteria
                if (
                    symbol.endswith("USDT")
                        and mcap is not None
                        and MarketFilters.MIN_MARKET_CAP  # Access the value
                        <= float(mcap)
                        <= MarketFilters.MAX_MARKET_CAP  # Access the value
                        and vol >= MarketFilters.MIN_VOLUME  # Access the value
                    ):
                    small_coins.append(symbol)

        except Exception as e:
                    print(f"Error finding small caps: {str(e)}")

        return small_coins
    
    def get_historical_data(self) -> None:
        """Collect historical data for multiple symbols"""
        print("\nCollecting historical data...")
        symbols = self.get_symbols()
        
        for i in range(0, len(symbols), MarketFilters.BATCH_SIZE):
            batch = symbols[i : i + MarketFilters.BATCH_SIZE]
            
            for symbol in tqdm(batch, desc=f"Batch {i//MarketFilters.BATCH_SIZE + 1}\n"):
                try:
                    # Get klines using the cached method
                    klines = self.cached_get_klines(
                        symbol=symbol, 
                        interval=MarketFilters.KLINE_INTERVAL
                    )

                    if len(klines) < MarketFilters.DAILY_BARS * 97:  # (training period + 1 week) * daily bars
                        continue

                    # Extract only OHLCV data from klines
                    df = pd.DataFrame([
                        [
                            k[0],  # timestamp
                            float(k[1]),  # open
                            float(k[2]),  # high
                            float(k[3]),  # low
                            float(k[4]),  # close
                            float(k[5])   # volume
                        ] 
                        for k in klines
                    ], columns=["timestamp", "open", "high", "low", "close", "volume"])

                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                    df.set_index("timestamp", inplace=True)

                    # Create directory if it doesn't exist
                    os.makedirs("data/historical", exist_ok=True)
                    df.to_csv(f"data/historical/historical_{symbol}.csv")

                except Exception as e:
                    print(f"Error collecting {symbol}: {e}")

                if i + MarketFilters.BATCH_SIZE < len(symbols):
                    time.sleep(MarketFilters.SLEEP_TIME)  # avoid rate limits
        

