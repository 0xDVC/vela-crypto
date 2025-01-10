from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
from .features import FeatureConfig
from sklearn.cluster import KMeans


@dataclass
class HMMConfig:
    n_states: int = 3  # no. of hidden states
    n_iter: int = 1000  # maximum iterations for training
    tol: float = 1e-5  # convergence tolerance
    random_seed: int = 42
    init_params: str = "kmeans"
    min_states: int = 2  # minimum states for CV (cross-validation)
    max_states: int = 5  # maximum states for CV
    cv_splits: int = 5  # number of CV splits


class HMMMarketCycle:
    """Gaussian Hidden Markov Model (HMM) for market cycle detection."""

    def __init__(self, config: HMMConfig = None):
        self.config = config or HMMConfig()
        self.model = None
        self.scaler = StandardScaler()
        self.trained = False
        self.state_labels = []
        self.feature_columns = FeatureConfig().FEATURE_COLUMNS
        # add cache for trained symbols
        self._trained_symbols = {}

    def _validate_features(self, features: pd.DataFrame) -> None:
        """ensure the required features are present in the DataFrame"""
        missing_features = set(self.feature_columns) - set(features.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

    def _prepare_features(self, features: pd.DataFrame) -> np.ndarray:
        """validate and scale the feature data"""
        self._validate_features(features)
        X = features[self.feature_columns].values

        # handle any infinite or NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)

        # Scale the features
        scaled_X = (
            self.scaler.fit_transform(X)
            if not self.trained
            else self.scaler.transform(X)
        )

        # ensure numerical stability
        scaled_X = np.clip(scaled_X, -100, 100)

        return scaled_X

    def _initialize_hmm(self, n_states: int, n_features: int) -> hmm.GaussianHMM:
        """Initialize HMM with proper parameters"""
        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=self.config.n_iter,
            random_state=self.config.random_seed,
            tol=self.config.tol,
            init_params="",
            params="stmc",
            verbose=False,
        )

        # set required attributes
        model.n_features = n_features
        model.n_components = n_states

        # initialize parameters with correct shapes
        model.startprob_ = np.ones(n_states) / n_states
        model.transmat_ = np.ones((n_states, n_states)) / n_states
        model.means_ = np.zeros((n_states, n_features))
        
        # create diagonal covariance matrices
        covars = np.ones((n_states, n_features))
        # force the attribute to be the correct shape
        model.__dict__['_covars_'] = covars.copy()          # AI assistant suggested this

        # double check the shape
        assert len(model._covars_.shape) == 2, f"Covariance must be 2D for diagonal type, got shape {model._covars_.shape}"
        assert model._covars_.shape == (n_states, n_features), f"Covariance shape mismatch: expected {(n_states, n_features)}, got {model._covars_.shape}"
        assert model.means_.shape == (n_states, n_features), f"Means shape mismatch: expected {(n_states, n_features)}, got {model.means_.shape}"

        return model

    def _assign_state_labels(self, X: np.ndarray) -> List[str]:
        """Assign labels to states based on their features."""
        states = self.model.predict(X)
        X_original = self.scaler.inverse_transform(X)
        state_means = np.array([X_original[states == i].mean(axis=0) for i in range(self.config.n_states)])
        
        # Get feature indices
        returns_idx = self.feature_columns.index("returns")
        vol_idx = self.feature_columns.index("volatility")
        momentum_idx = self.feature_columns.index("momentum")
        rsi_idx = self.feature_columns.index("rsi")
        
        labels = []
        state_info = []
        
        # Revised classification logic with stricter thresholds
        for i in range(self.config.n_states):
            returns = state_means[i, returns_idx]
            volatility = state_means[i, vol_idx]
            momentum = state_means[i, momentum_idx]
            rsi = state_means[i, rsi_idx]
            
            # Bull market: Strong positive signals required
            if returns > 0.0005 and momentum > 0 and rsi > 55 and volatility < 0.015:
                label = "bull"
            # Bear market: Any strong negative signal
            elif returns < -0.0003 or (volatility > 0.02 and rsi < 45):
                label = "bear"
            else:
                label = "neutral"
            
            labels.append(label)
            state_info.append((i, label, returns, volatility, momentum, rsi))
        
        print("\nState Classification:")
        for state, label, ret, vol, mom, rsi in state_info:
            print(f"State {state}: {label:<6} (Returns: {ret:8.6f}, Vol: {vol:8.6f}, "
                  f"Momentum: {mom:8.6f}, RSI: {rsi:8.2f})")
        
        return labels

    def _calculate_aic(self, X: np.ndarray) -> float:
        """Calculate the Akaike Information Criterion (AIC)."""
        n_params = self.config.n_states * (
            len(self.feature_columns) + self.config.n_states - 1
        )
        return -2 * self.model.score(X) + 2 * n_params

    def _calculate_bic(self, X: np.ndarray) -> float:
        """Calculate the Bayesian Information Criterion (BIC)."""
        n_params = self.config.n_states * (
            len(self.feature_columns) + self.config.n_states - 1
        )
        return -2 * self.model.score(X) + n_params * np.log(len(X))

    def cross_validate(
        self, 
        features: pd.DataFrame
    ) -> Tuple[int, float, Dict[int, float]]:
        """find optimal number of HMM states"""
        print("Starting cross-validation...")
        X = self._prepare_features(features)
        n_features = X.shape[1]

        # calculate minimum samples per split
        min_samples = len(X) // self.config.cv_splits
        if min_samples < 30:
            raise ValueError(f"Not enough samples per split. Got {min_samples}, need at least 30")

        best_score = float("-inf")
        best_n_states = self.config.min_states
        scores_by_state = {}


        # NOTE: this is quite slow, takes an awful amount of time to run. 
        # TODO: try to optimize this, or use a different approach
        for n_states in range(self.config.min_states, self.config.max_states + 1):
            cv_scores = []

            for i in range(2, self.config.cv_splits + 1):
                split_point = (len(X) * i) // self.config.cv_splits

                X_train = X[:split_point - min_samples]
                X_val = X[split_point - min_samples:split_point]

                if len(X_val) > 0:
                    # initialize model outside try block
                    model = self._initialize_hmm(n_states, n_features)
                    
                    try:
                        # initialize means using k-means
                        kmeans = KMeans(
                            n_clusters=n_states, 
                            random_state=self.config.random_seed
                        )
                        kmeans.fit(X_train)
                        model.means_ = kmeans.cluster_centers_

                        # Verify shapes before fitting
                        assert model._covars_.shape == (n_states, n_features)
                        assert model.means_.shape == (n_states, n_features)

                        model.fit(X_train)
                        score = model.score(X_val)
                        cv_scores.append(score)

                    except Exception as e:
                        print(f"Warning: CV failed for {n_states} states: {str(e)}")
                        print(f"Shapes - X_train: {X_train.shape}, means: {model.means_.shape}, covars: {model._covars_.shape}")
                        continue

            if cv_scores:
                mean_score = np.mean(cv_scores)
                scores_by_state[n_states] = mean_score
                print(f"States: {n_states}, Average Score: {mean_score:.2f}")

                if mean_score > best_score:
                    best_score = mean_score
                    best_n_states = n_states

        return best_n_states, best_score, scores_by_state

    def train(self, data: pd.DataFrame, symbol: str = None, optimize_states: bool = True) -> Dict[str, Any]:
        """Train the HMM model with optional state optimization and caching"""
        # if we have cached results for this symbol, return them
        if symbol and symbol in self._trained_symbols:
            self.model = self._trained_symbols[symbol]['model']
            self.state_labels = self._trained_symbols[symbol]['labels']
            self.trained = True
            return self._trained_symbols[symbol]['metrics']

        # proceed with training if no cache exists
        if optimize_states:
            best_n_states, best_score, _ = self.cross_validate(data)
            self.config.n_states = best_n_states
            print(f"\nSelected optimal states: {best_n_states} (score: {best_score:.2f})")

        # prepare features
        X = self._prepare_features(data)
        n_features = X.shape[1]

        # initialize model with current config and custom initialization
        self.model = hmm.GaussianHMM(
            n_components=self.config.n_states,
            covariance_type="diag",
            n_iter=self.config.n_iter,
            random_state=self.config.random_seed,
            tol=self.config.tol,
            init_params="",
            params="stmc",
            verbose=False,
        )
        
        # set n_features attribute
        self.model.n_features = n_features

        # multiple random initializations to avoid local optima
        best_score = float("-inf")
        best_params = None

        np.random.seed(self.config.random_seed)

        for i in range(5):
            try:
                # manual initialization
                self.model.startprob_ = np.ones(self.config.n_states) / self.config.n_states
                self.model.transmat_ = np.ones((self.config.n_states, self.config.n_states)) / self.config.n_states

                # initialize means using k-means
                kmeans = KMeans(
                    n_clusters=self.config.n_states,
                    random_state=self.config.random_seed + i,
                )
                kmeans.fit(X)
                self.model.means_ = kmeans.cluster_centers_

                # simple covariance initializatio
                self.model._covars_ = np.ones((self.config.n_states, n_features))

                # fit the model
                self.model.fit(X)
                score = self.model.score(X)

                if score > best_score:
                    best_score = score
                    best_params = {
                        "startprob_": self.model.startprob_.copy(),
                        "transmat_": self.model.transmat_.copy(),
                        "means_": self.model.means_.copy(),
                        "_covars_": self.model._covars_.copy(),
                    }

            except Exception as e:
                print(f"Warning: Model fitting attempt {i+1} failed: {str(e)}")
                continue

        if best_params is None:
            raise RuntimeError("Failed to fit model after multiple attempts")

        # create a new model with the best parameters
        self.model = hmm.GaussianHMM(
            n_components=self.config.n_states,
            covariance_type="diag",
            n_iter=self.config.n_iter,
            random_state=self.config.random_seed,
            tol=self.config.tol,
            init_params="",
            params="stmc",
        )

        # set the best parameters
        for param, value in best_params.items():
            setattr(self.model, param, value)

        self.trained = True
        self.state_labels = self._assign_state_labels(X)

        # store results in cache if symbol is provided
        if symbol:
            self._trained_symbols[symbol] = {
                'model': self.model,
                'labels': self.state_labels,
                'metrics': {
                    "log_likelihood": self.model.score(X),
                    "n_iter": self.model.monitor_.iter,
                    "converged": self.model.monitor_.converged,
                    "aic": self._calculate_aic(X),
                    "bic": self._calculate_bic(X),
                    "n_states": self.config.n_states,
                }
            }
            return self._trained_symbols[symbol]['metrics']

        return {
            "log_likelihood": self.model.score(X),
            "n_iter": self.model.monitor_.iter,
            "converged": self.model.monitor_.converged,
            "aic": self._calculate_aic(X),
            "bic": self._calculate_bic(X),
            "n_states": self.config.n_states,
        }

    def predict(self, data: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """Predict the hidden states for the given data"""
        if not self.trained:
            raise RuntimeError("Train the model first.")
        
        # use cached model if available
        if symbol and symbol in self._trained_symbols:
            self.model = self._trained_symbols[symbol]['model']
            self.state_labels = self._trained_symbols[symbol]['labels']
        
        if not self.state_labels:
            raise RuntimeError("No state labels available. Model may not be properly trained.")
        
        try:
            X = self._prepare_features(data)
            state_probs = self.model.predict_proba(X)
            states = self.model.predict(X)
            
            # verify states are within expected range
            if np.any(states >= len(self.state_labels)):
                raise ValueError(f"Invalid state index detected. Max state: {states.max()}, Available labels: {len(self.state_labels)}")

            results = pd.DataFrame(
                state_probs,
                columns=[f"{label}_prob" for label in self.state_labels],
                index=data.index,
            )
            results["state"] = [self.state_labels[state] for state in states]
            return results
            
        except Exception as e:
            print(f"Error in predict: {str(e)}")
            print(f"State labels: {self.state_labels}")
            print(f"Model states: {self.config.n_states}")
            raise RuntimeError(f"Prediction failed: {str(e)}")
