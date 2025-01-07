from dataclasses import dataclass
import numpy as np
import pandas as pd
from hmmlearn import hmm
from typing import Dict
from sklearn.preprocessing import StandardScaler
import warnings


@dataclass
class HMMConfig:
    n_states: int = 3
    n_iter: int = 2000        # Maximum number of iterations for EM algorithm
    tol: float = 1e-5         # Convergence threshold
    random_state: int = 42
    min_prob: float = 0.50    # Increased confidence threshold
    min_duration: int = 12    # 3 hours minimum (15m bars)
    volatility_threshold: float = 2.0  # Maximum volatility multiplier

class HMMMarketCycle:
    def __init__(self):
        self.config = HMMConfig()
        self.model = hmm.GaussianHMM(
            n_components=self.config.n_states,
            covariance_type="diag",
            n_iter=self.config.n_iter,
            random_state=self.config.random_state
        )
        self.scaler = StandardScaler()
        self.trained = False
        self.state_labels = {}  # Initialize state_labels
        self.feature_columns = [
            'returns',
            'volatility',
            'ma_trend',
            'rsi',
            'adx',
            'trend_strength'
        ]

    def fit(self, features: pd.DataFrame) -> Dict:
        """Fit the HMM model with monitoring"""
        X = self._prepare_features(features)
        
        # Initialize model
        self.model = hmm.GaussianHMM(
            n_components=self.config.n_states,
            n_iter=self.config.n_iter,
            tol=self.config.tol,
            random_state=self.config.random_state
        )
        
        # Fit and monitor
        self.model.fit(X)
        
        if self.model.monitor_.iter == self.config.n_iter:
            warnings.warn(
                f"HMM did not converge after {self.config.n_iter} iterations. "
                f"Consider increasing n_iter or adjusting tol (current: {self.config.tol})"
            )
        
        # Set state labels after fitting
        self.state_labels = self._analyze_states(X)
        self.trained = True
        
        return {
            'log_likelihood': float(self.model.score(X)),
            'state_labels': self.state_labels
        }

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Predict market regimes with dynamic thresholds"""
        if not self.trained:
            raise RuntimeError("Model must be trained before making predictions")
            
        # Validate features
        self._validate_features(features)
        
        # Transform features using fitted scaler
        X = features[self.feature_columns].values
        X_scaled = self.scaler.transform(X)
        
        # Get predictions
        state_probs = self.model.predict_proba(X_scaled)
        states = self.model.predict(X_scaled)
        
        # Create results DataFrame
        results = pd.DataFrame(
            state_probs, 
            columns=[f"{label}_prob" for label in ['bull', 'bear', 'neutral']],
            index=features.index
        )
        results['state'] = [self.state_labels[state] for state in states]
        
        # Calculate dynamic threshold
        bull_probs = results['bull_prob']
        results['bull_threshold'] = bull_probs.rolling(window=48).mean()  # 12-hour moving average
        results['bull_threshold'] = results['bull_threshold'].fillna(bull_probs.mean())
        
        return results

    def _validate_features(self, features: pd.DataFrame) -> None:
        """Ensure required features are present"""
        missing_features = set(self.feature_columns) - set(features.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
    
    def _prepare_features(self, features: pd.DataFrame) -> np.ndarray:
        """Validate and scale the feature data"""
        self._validate_features(features)
        X = features[self.feature_columns].values
        return self.scaler.fit_transform(X) if not self.trained else self.scaler.transform(X)

    def _analyze_states(self, features: pd.DataFrame) -> Dict[int, str]:
        """Enhanced market regime classification with clearer thresholds"""
        means = self.model.means_
        
        # Create state characteristics
        state_chars = pd.DataFrame({
            'returns': means[:, 0],
            'volatility': means[:, 1],
            'ma_trend': means[:, 2],
            'rsi': means[:, 3],
            'adx': means[:, 4],
            'trend_strength': means[:, 5]
        })
        
        labels = {}
        for state in range(len(means)):
            returns = state_chars.loc[state, 'returns']
            ma_trend = state_chars.loc[state, 'ma_trend']
            rsi = state_chars.loc[state, 'rsi']
            adx = state_chars.loc[state, 'adx']
            trend_strength = state_chars.loc[state, 'trend_strength']
            
            # Bull Market Conditions
            if ((returns > 0) and (ma_trend > 0 or trend_strength > 0)):
                labels[state] = 'bull'
            # Bear Market Conditions    
            elif ((returns < 0) and (ma_trend < 0 or trend_strength < 0)):
                labels[state] = 'bear'
            # Ranging/Neutral Market
            else:
                labels[state] = 'neutral'
            
            print(f"\nState {state} Analysis:")
            print(f"Returns: {returns:.6f}")
            print(f"MA Trend: {ma_trend:.6f}")
            print(f"RSI: {rsi:.6f}")
            print(f"ADX: {adx:.6f}")
            print(f"Trend Strength: {trend_strength:.6f}")
            print(f"Label: {labels[state]}")
        
        return labels
