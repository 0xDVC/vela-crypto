from dataclasses import dataclass
# from .model import HMMMarketCycle
import pandas as pd
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class TradingConfig:
    initial_balance: float = 10000.0
    trading_fee: float = 0.001
    position_size: float = 0.95
    stop_loss_pct: float = 0.05     
    trailing_stop_pct: float = 0.07
    max_drawdown_pct: float = 0.15  # Maximum drawdown before stopping


class TradingStrategy:
    def __init__(self, config: TradingConfig = None):
        self.config = config or TradingConfig()
        self.reset()
        
    def reset(self):
        """Reset trading state"""
        self.in_position = False
        self.entry_price = 0.0
        self.stop_loss_price = 0.0
        self.trailing_stop_price = 0.0
        self.peak_portfolio_value = self.config.initial_balance
        self.current_balance = self.config.initial_balance
        self.position_size = 0.0
        self.units = 0.0  # track number of units held
        self.price_history = []  # store recent prices for volatility calculation
        self.trade_history = []  # initialize trade history list
        
    def calculate_volatility(self, price: float) -> float:
        """Calculate rolling volatility"""
        # Add new price to history
        self.price_history.append(price)
        
        # Maintain window size
        if len(self.price_history) > self.config.volatility_window:
            self.price_history = self.price_history[-self.config.volatility_window:]
            
        # Need at least 2 prices to calculate volatility
        if len(self.price_history) < 2:
            return 0.0
            
        # Calculate returns and volatility
        prices = np.array(self.price_history)
        returns = np.diff(prices) / prices[:-1]
        return np.std(returns)
        
    def _calculate_portfolio_value(self, current_price: float) -> float:
        """Calculate current portfolio value"""
        if self.in_position:
            return self.units * current_price
        return self.current_balance
        
    def execute_trade(self, date: pd.Timestamp, price: float, state: str) -> Dict:
        """Execute trading logic with simplified risk management"""
        current_portfolio_value = self._calculate_portfolio_value(price)
        
        # Calculate drawdown from peak
        drawdown = (self.peak_portfolio_value - current_portfolio_value) / self.peak_portfolio_value
        
        # Update peak portfolio value
        if current_portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = current_portfolio_value
        
        # Check maximum drawdown threshold
        if drawdown >= self.config.max_drawdown_pct:
            if self.in_position:
                return self._exit_position(date, price, "max_drawdown")
            return None  # Stop trading after hitting max drawdown
        
        # Position Management
        if self.in_position:
            # Exit on state change
            if state != "bull":
                return self._exit_position(date, price, "state_change")
                
            # Fixed stop-loss
            if price <= self.stop_loss_price:
                return self._exit_position(date, price, "stop_loss")
                
            # Standard trailing stop
            if price > self.trailing_stop_price / (1 - self.config.trailing_stop_pct):
                self.trailing_stop_price = price * (1 - self.config.trailing_stop_pct)
                self.stop_loss_price = max(self.stop_loss_price, self.trailing_stop_price)
                
        # Entry logic - trust the model's state prediction
        elif state == "bull":
            return self._enter_position(date, price)
            
        return None

    def _enter_position(self, date: pd.Timestamp, price: float) -> Dict:
        """Enter a new position"""
        self.position_size = self.current_balance * self.config.position_size
        self.units = self.position_size / price
        self.entry_price = price
        
        fee = self.position_size * self.config.trading_fee
        self.current_balance -= (self.position_size + fee)
        self.in_position = True
        
        # Set initial stop-loss and trailing stop
        self.stop_loss_price = price * (1 - self.config.stop_loss_pct)
        self.trailing_stop_price = price * (1 - self.config.trailing_stop_pct)
        
        trade = {
            'date': date,
            'type': 'enter',
            'price': price,
            'units': self.units,
            'fee': fee
        }
        self.trade_history.append(trade)
        return trade
        
    def _exit_position(self, date: pd.Timestamp, price: float, reason: str) -> Dict:
        """Exit current position"""
        if not self.in_position:
            return None
            
        exit_value = self.units * price
        fee = exit_value * self.config.trading_fee
        self.current_balance += (exit_value - fee)
        
        trade = {
            'date': date,
            'type': 'exit',
            'price': price,
            'units': self.units,
            'reason': reason,
            'fee': fee
        }
        
        # Reset position tracking
        self.in_position = False
        self.units = 0.0
        self.position_size = 0.0
        
        self.trade_history.append(trade)
        return trade

    def calculate_dynamic_stops(self, price: float, current_volatility: float):
        """calculate dynamic stop losses based on volatility"""
        # base stops
        base_stop = self.config.stop_loss_pct
        base_trailing = self.config.trailing_stop_pct
        
        # adjust stops based on volatility
        volatility_multiplier = min(3.0, max(1.0, current_volatility * 100))
        dynamic_stop = base_stop * volatility_multiplier
        dynamic_trailing = base_trailing * volatility_multiplier
        
        # set price levels
        self.stop_loss_price = price * (1 - dynamic_stop)
        self.trailing_stop_price = price * (1 - dynamic_trailing)
        
        return dynamic_stop, dynamic_trailing

    def backtest(self, test_data: pd.DataFrame, predictions: pd.DataFrame) -> Tuple[pd.Series, List[Dict]]:
        """Run backtest on test data with predictions"""
        self.reset()
        portfolio_values = []
        
        # We already have the predictions and features from the model
        # No need to recalculate anything
        
        for i in range(len(predictions)):
            date = predictions.index[i]
            price = test_data.loc[date, 'close']
            raw_state = predictions['state'].iloc[i]
            
            trade = self.execute_trade(date, price, raw_state)
            
            # calculate current portfolio value
            portfolio_value = self.current_balance
            if self.in_position:
                portfolio_value += self.units * price
            portfolio_values.append(portfolio_value)

        return pd.Series(portfolio_values, index=predictions.index), self.trade_history

    def calculate_metrics(self, portfolio_values: pd.Series, buy_hold_values: pd.Series) -> Dict:
        """calculate trading performance metrics"""
        portfolio_returns = portfolio_values.pct_change().dropna()
        benchmark_returns = buy_hold_values.pct_change().dropna()
        
        portfolio_metrics = {
            "Total Return (%)": ((portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1) * 100,
            "Sharpe Ratio": portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252),
            "Max Drawdown (%)": (1 - (portfolio_values / portfolio_values.cummax())).max() * 100,
            "Final Value": portfolio_values.iloc[-1]
        }
        
        benchmark_metrics = {
            "Total Return (%)": ((buy_hold_values.iloc[-1] / buy_hold_values.iloc[0]) - 1) * 100,
            "Sharpe Ratio": benchmark_returns.mean() / benchmark_returns.std() * np.sqrt(252),
            "Max Drawdown (%)": (1 - (buy_hold_values / buy_hold_values.cummax())).max() * 100,
            "Final Value": buy_hold_values.iloc[-1]
        }
        
        return {
            "Portfolio": portfolio_metrics,
            "Benchmark": benchmark_metrics
        }
