from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class FeatureEngineer:
    def __init__(self):
        self.feature_columns = [
            'returns',
            'volatility',
            'ma_trend',
            'rsi',
            'adx',
            'trend_strength'
        ]
        
    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical features for market regime detection"""
        features = pd.DataFrame(index=data.index)
        
        # Price-based features
        features['returns'] = data['close'].pct_change()
        features['volatility'] = features['returns'].rolling(20).std()
        
        # Trend features
        features['ma_trend'] = self._calculate_ma_trend(data['close'])
        features['rsi'] = self._calculate_rsi(data['close'])
        features['adx'] = self._calculate_adx(data)
        features['trend_strength'] = self._calculate_trend_strength(data)
        
        # Forward fill any NaN values
        features = features.ffill().fillna(0)
        
        return features
    
    def prepare_train_test(self, data: pd.DataFrame, train_size: float = 0.7) -> tuple:
        """Prepare training and testing datasets"""
        features = self.calculate_features(data)
        split_idx = int(len(features) * train_size)
        return features[:split_idx], features[split_idx:]
    
    def _calculate_ma_trend(self, prices: pd.Series, fast: int = 10, slow: int = 30) -> pd.Series:
        """Calculate moving average trend"""
        fast_ma = prices.rolling(fast).mean()
        slow_ma = prices.rolling(slow).mean()
        return (fast_ma - slow_ma) / slow_ma
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate +DM and -DM
        high_diff = high.diff()
        low_diff = low.diff()
        
        pos_dm = high_diff.where((high_diff > 0) & (high_diff > -low_diff), 0)
        neg_dm = (-low_diff).where((low_diff > 0) & (low_diff > high_diff), 0)
        
        tr = pd.DataFrame({
            'hl': high - low,
            'hc': (high - close.shift()).abs(),
            'lc': (low - close.shift()).abs()
        }).max(axis=1)
        
        # Smooth with Wilder's moving average
        smoothing = 1 - 1/period
        tr_smooth = tr.ewm(alpha=1/period).mean()
        pos_di = 100 * pos_dm.ewm(alpha=1/period).mean() / tr_smooth
        neg_di = 100 * neg_dm.ewm(alpha=1/period).mean() / tr_smooth
        
        # Calculate ADX
        dx = 100 * (pos_di - neg_di).abs() / (pos_di + neg_di)
        adx = dx.ewm(alpha=1/period).mean()
        
        return adx
    
    def _calculate_trend_strength(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate trend strength using price momentum and volume"""
        momentum = data['close'].pct_change(period)
        volume_change = data['volume'].pct_change(period)
        
        # Combine momentum and volume signals
        return momentum * np.sign(volume_change)