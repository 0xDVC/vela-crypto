from dataclasses import dataclass
import pandas as pd
import numpy as np
import ta
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple

@dataclass
class Features:
    # Adjusted for 15-minute intervals
    rsi_period: int = 14        # 3.5 hours (14 * 15min)
    vol_window: int = 24        # 6 hours (24 * 15min)
    ma_fast: int = 20          # 5 hours
    ma_slow: int = 80          # 20 hours
    volume_window: int = 24     # 6 hours
    
    # For 3 months of 15min data
    lookback_bars: int = 8640   # 90 days * 24 hours * 4 (15-min bars)
    test_bars: int = 672       # 7 days * 24 hours * 4 (15-min bars)

class FeatureEngineer:
    def __init__(self, params: Features = None):
        self.params = params or Features()
        self.feature_columns = [
            'returns',           # 15-min returns
            'volatility',        # Rolling volatility
            'mean_reversion',    # Oversold/overbought
            'rsi'               # RSI indicator
        ]
    
    def _calculate_volatility(self, data: pd.DataFrame) -> pd.Series:
        """Volatility adjusted for 15-min intervals"""
        if isinstance(data, pd.DataFrame):
            prices = data['close']
      
        
        returns = np.log(prices / prices.shift(1))
        # scale to 15-min volatility 
        volatility = returns.ewm(span=self.params.vol_window).std() * np.sqrt(4)
        return volatility.fillna(0)
    
    def _detect_volume_surges(self, volume: pd.Series) -> pd.Series:
        """Volume surges on 15-min timeframe"""
        # use shorter window for more reactive volume detection
        vol_ema = volume.ewm(span=self.params.volume_window).mean()
        vol_std = volume.ewm(span=self.params.volume_window).std()
        z_score = (volume - vol_ema) / (vol_std + 1e-8)
        return z_score.fillna(0)
    
    def prepare_train_test(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data accounting for 15-min bars"""
        if len(data) < (self.params.lookback_bars + self.params.test_bars):
            raise ValueError(
                f"Insufficient data: need {self.params.lookback_bars + self.params.test_bars} "
                f"15-min bars, got {len(data)}"
            )
        
        # train-test split data
        train_data = data[:-self.params.test_bars]
        test_data = data[-self.params.test_bars:]
        
        print(f"\nData splits (15-minute intervals):")
        print(f"Training: {train_data.index[0]} to {train_data.index[-1]} ({len(train_data)} bars)")
        print(f"Testing:  {test_data.index[0]} to {test_data.index[-1]} ({len(test_data)} bars)")
        
        return self.process_features(train_data, is_training=True), \
               self.process_features(test_data, is_training=False)
    
    def process_features(self, data: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Process and engineer features from price data"""
        features = pd.DataFrame(index=data.index)
        
        # calculate features
        features['returns'] = self._calculate_returns(data['close'])
        features['volatility'] = self._calculate_volatility(data)
        features['volume_surge'] = self._detect_volume_surges(data['volume'])
        features['mean_reversion'] = self._calculate_mean_reversion(data['close'])
        features['rsi'] = self._calculate_rsi(data)
        
        if is_training:
            self._analyze_correlations(features)
        
        return features.dropna()
    
    def _calculate_returns(self, prices: pd.Series) -> pd.Series:
        """Log returns with forward-looking bias check"""
        returns = np.log(prices / prices.shift(1))
        return returns.fillna(0)
    
    def _calculate_mean_reversion(self, prices: pd.Series) -> pd.Series:
        """oversold/overbought indicator"""
        rsi = ta.momentum.rsi(prices, window=self.params.rsi_period)
        mean_rev = -((rsi - 50) / 50)
        return mean_rev.fillna(0)
    
    def _calculate_rsi(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """calculate RSI"""
        # ensure we're working with a DataFrame
        if isinstance(data, pd.Series):
            close_prices = data
        else:
            if 'close' not in data.columns:
                raise ValueError("DataFrame must contain 'close' column")
            close_prices = data['close']
        
        # clculate RSI
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Fill NaN with neutral value
    
    def _analyze_correlations(self, features: pd.DataFrame) -> None:
        """analyze and plot feature correlations"""
        corr_matrix = features.corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.show()
        
        print("\nFeature correlations:")
        print(corr_matrix.round(3))
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for model"""
        if isinstance(data, pd.DataFrame):
            prices = data['close']
            
        features = pd.DataFrame(index=data.index)
        features['returns'] = self._calculate_returns(prices)
        features['volatility'] = self._calculate_volatility(data)
        features['mean_reversion'] = self._calculate_mean_reversion(prices)
        features['rsi'] = self._calculate_rsi(data, self.params.rsi_period)
        
        return features