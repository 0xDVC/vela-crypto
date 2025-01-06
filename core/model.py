from dataclasses import dataclass
import numpy as np
import pandas as pd
from hmmlearn import hmm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict

@dataclass
class HMMConfig:
    n_states: int = 3          # bull, bear, neutral
    n_iter: int = 100         # max iterations for training
    random_state: int = 42
    min_prob: float = 0.15    # threshold for 15m data
    min_duration: int = 12    # min_state duration (3 hours in 15m bars)

class HMMMarketCycle:
    def __init__(self, config: HMMConfig = None):
        self.config = config or HMMConfig()

        # initialize gaussian hmm model        
        self.model = hmm.GaussianHMM(
            n_components=self.config.n_states,
            covariance_type="diag",
            n_iter=self.config.n_iter,
            random_state=self.config.random_state,
            init_params="kmeans",
            params="stmc",
            min_covar=1e-5  # seemed more stable for higher frequency data
        )
        
        self.trained = False
        self.state_labels = None
        self.feature_means = None
    
    def train(self, features: pd.DataFrame) -> Dict:
        """Train HMM model on features"""
        print("\nTraining HMM model...")
        
            # feature matrix
        X = features[['returns', 'volatility', 'volume_surge', 'mean_reversion']].values
        
        # train HMM
        self.model.fit(X)
        
        # get predictions
        states = self.model.predict(X)
        
        # add states to features for analysis
        features_with_states = features.copy()
        features_with_states['state'] = states
        
        # analyze states and assign labels
        self.state_labels = self._analyze_states(features_with_states)
        
        return {
            'log_likelihood': self.model.score(X),
            'n_iter': self.model.monitor_.iter,
            'states': states
        }
    
    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """generate predictions for features"""
        X = features[['returns', 'volatility', 'volume_surge', 'mean_reversion']].values
        
        # Get state predictions and probabilities
        states = self.model.predict(X)
        state_probs = self.model.predict_proba(X)
        
        # Create predictions DataFrame
        predictions = pd.DataFrame(index=features.index)
        predictions['state'] = states
        predictions['label'] = [self.state_labels.get(s, "unknown") for s in states]
        predictions['confidence'] = [max(probs) for probs in state_probs]  # Add confidence scores
        
        return predictions
    
    def _analyze_states(self, features: pd.DataFrame) -> Dict[int, str]:
        """analyze features to determine state characteristics"""
        state_metrics = features.groupby('state').agg({
            'returns': ['mean', 'std'],
            'volatility': 'mean',
            'mean_reversion': 'mean',
            'rsi': 'mean'
        })
        
        returns_mean = state_metrics[('returns', 'mean')]
        rsi_means = state_metrics[('rsi', 'mean')]
        
        # sort states by return for classification
        sorted_states = returns_mean.sort_values(ascending=False)
        
        labels = {}
        
        # top state: check for bull conditions
        top_state = sorted_states.index[0]
        if returns_mean[top_state] > 0.00005 and rsi_means[top_state] > 50:
            labels[top_state] = 'bull'
        else:
            labels[top_state] = 'neutral'
        
        # bottom state: check for bear conditions
        bottom_state = sorted_states.index[-1]
        if returns_mean[bottom_state] < -0.00005 and rsi_means[bottom_state] < 50:
            labels[bottom_state] = 'bear'
        else:
            labels[bottom_state] = 'neutral'
        
        # middle state is always neutral
        middle_state = sorted_states.index[1]
        labels[middle_state] = 'neutral'
        


        ## just for debugging, and verbose logs
        # Print detailed state analysis
        print("\nDetailed State Analysis:")
        for state in range(self.config.n_states):
            print(f"\nState {state} ({labels[state]}):")
            print(f"Returns: mean={returns_mean[state]:.6f}, std={state_metrics.loc[state, ('returns', 'std')]:.6f}")
            print(f"RSI: {rsi_means[state]:.2f}")
            print(f"Mean Reversion: {state_metrics.loc[state, ('mean_reversion', 'mean')]:.4f}")
        
        return labels
    
    def _get_feature_means(self, features: pd.DataFrame) -> pd.DataFrame:
        """calculate mean feature values for each state"""
        states = self.model.predict(features.values)
        means = pd.DataFrame(index=self.state_labels, columns=features.columns)
        
        for i, label in enumerate(self.state_labels):
            state_data = features[states == i]
            means.loc[label] = state_data.mean()
        
        return means
    
    def _plot_state_characteristics(self, features: pd.DataFrame) -> None:
        """visualize state characteristics"""
        states = self.model.predict(features.values)
        
        # create state-wise feature distribution plot
        plt.figure(figsize=(12, 6))
        for i, feature in enumerate(features.columns):
            plt.subplot(2, 2, i+1)
            for j, label in enumerate(self.state_labels):
                state_data = features[states == j][feature]
                sns.kdeplot(state_data, label=label)
            plt.title(f'{feature} Distribution by State')
            plt.legend()
        plt.tight_layout()
        plt.show()
        
        # plot state transitions over time
        plt.figure(figsize=(15, 4))
        plt.plot(features.index, [self.state_labels[s] for s in states])
        plt.title('Market States Over Time')
        plt.xlabel('Date')
        plt.ylabel('State')
        plt.grid(True)
        plt.show()
    
    def _smooth_states(self, states: np.ndarray) -> np.ndarray:
        """Smooth state predictions to avoid rapid switching"""
        smoothed = states.copy()
        n = len(states)
        
        # frward pass
        i = 0
        while i < n:
            state_start = i
            current_state = states[i]
            
            # find state duration
            while i < n and states[i] == current_state:
                i += 1
            duration = i - state_start
            
            # if duration is too short, extend previous state
            if duration < self.config.min_duration and state_start > 0:
                smoothed[state_start:i] = smoothed[state_start - 1]
        
        return smoothed
