from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass
class Features:
    volatility_window: int = 14
    volume_ma_window: int = 14
    rsi_window: int = 14

class HMMFeatureEngineer:
    def __init__(self, params: Features = None):
        self.params = params or Features()
    
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """calculate core features for HMM model"""
        features = pd.DataFrame(index=df.index)
        
        # 1. Returns - normally distributed, captures price movement
        features['returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # 2. Volatility - right skewed, captures market turbulence
        features['volatility'] = self._calculate_volatility(df)
        
        # 3. Volume Intensity - right skewed, captures trading activity
        features['volume_intensity'] = self._calculate_volume_features(df)
        
        # 4. RSI - symmetrically distributed, captures momentum
        features['rsi'] = self._calculate_rsi(df)
        
        return features.dropna()
    
    def _calculate_volatility(self, df: pd.DataFrame) -> pd.Series:
        """calculate price volatility using true range"""
        ranges = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift(1)).abs(),
            (df['low'] - df['close'].shift(1)).abs()
        ], axis=1)
        
        return ranges.max(axis=1).rolling(self.params.volatility_window).mean()
    
    def _calculate_volume_features(self, df: pd.DataFrame) -> pd.Series:
        """calculate volume intensity relative to moving average"""
        volume_ma = df['volume'].rolling(self.params.volume_ma_window).mean()
        return df['volume'] / volume_ma
    
    def _calculate_rsi(self, df: pd.DataFrame) -> pd.Series:
        """calculate RSI for momentum indication"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(self.params.rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.params.rsi_window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))