import pandas as pd
import numpy as np
from dataclasses import dataclass, field

@dataclass
class FeatureConfig():
    """feature configuration for short-term HMM on small caps"""
    FEATURE_COLUMNS: list = field(default_factory=lambda: ["returns", "rsi", "bb_position", "volatility", "momentum"])
    RSI_PERIOD: int = 10
    BB_PERIOD: int = 15
    BB_STD: float = 1.8
    MOMENTUM_SPAN: int = 3


class FeatureEngineer:
    def __init__(self):
        self.config = FeatureConfig()

    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate features optimized for short-term HMM on small caps"""
        features = pd.DataFrame(index=data.index)

        # core features for detection of regimes
        features["returns"] = np.log(data["close"]).diff()
        features["volatility"] = (data["high"] - data["low"]) / data["close"]
        features["volatility"] = np.log1p(features["volatility"])

        # oscillators with shorter period   s for small caps
        features["rsi"] = self.calculate_rsi(
            data["close"], period=self.config.RSI_PERIOD
        )

        # Modified BB position with tighter bands for small caps
        bb_period = self.config.BB_PERIOD
        bb_std = self.config.BB_STD

        features["bb_position"] = self.calculate_bb_position(
            data["close"], bb_period, bb_std
        )

        # short-term momentum 
        features["momentum"] = (
            (data["close"] / data["close"].shift(self.config.MOMENTUM_SPAN) - 1)
            .ewm(span=self.config.MOMENTUM_SPAN)
            .mean()
        )

        # drop NaN values
        features = features.dropna()

        features = self.normalize(features)

        return features

    def calculate_rsi(self, data: pd.Series, period: int) -> pd.Series:
        """calculate the Relative Strength Index (RSI)."""
        delta = data.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))

    def calculate_bb_position(
        self, data: pd.Series, period: int, std: float
    ) -> pd.Series:
        """calculate the Bollinger Bands position."""
        middle_band = data.rolling(period).mean()
        std_dev = data.rolling(period).std()
        upper_bb = middle_band + (std * std_dev)
        lower_bb = middle_band - (std * std_dev)
        return (data - lower_bb) / (upper_bb - lower_bb)

    def normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        # normalize features
        for col in data.columns:
            if col in ["rsi", "bb_position"]:  # keep these in original scale
                continue
            elif col == "volatility":
                data[col] = np.abs(data[col])
            elif col == "returns":
                data[col] = data[col].clip(-0.15, 0.15)  # wider clips for small caps
            else:
                max_abs_val = np.abs(data[col]).max()
                if max_abs_val > 0:
                    data[col] = data[col] / max_abs_val

        return data

    def prepare_train_test(self, data: pd.DataFrame) -> tuple:
        """Prepare training and testing datasets with fixed 3-month training period"""
        features = self.calculate_features(data)

        train_size = 90 * 24 * 4  # 90 days * 24 hours * 4 (15-min intervals)

        if len(features) < train_size:
            raise ValueError(
                f"Not enough data. Need at least {train_size} rows, but got {len(features)}"
            )

        train_data = features[:train_size]
        test_data = features[train_size:]

        print(f"\nData Split Info:")
        print(f"Training period: {train_data.index[0]} to {train_data.index[-1]}")
        print(f"Testing period: {test_data.index[0]} to {test_data.index[-1]}")
        print(f"Training samples: {len(train_data)}")
        print(f"Testing samples: {len(test_data)}")

        return train_data, test_data
