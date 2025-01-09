from dataclasses import dataclass
from .model import HMMMarketCycle
import pandas as pd
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class TradingConfig:
    initial_balance: float = 10000.0
    trading_fee: float = 0.001
    position_size: float = 0.65
    stop_loss_pct: float = 0.05  # 5% stop loss
    trailing_stop_pct: float = 0.07  # 7% trailing stop


class TradingStrategy:
    def __init__(self):
        self.trading_config = TradingConfig()
        self.model = HMMMarketCycle()
        self.reset()
        self.volatility_window = 20  # rolling window for volatility calculation

    def reset(self):
        """Reset trading state"""
        self.current_balance = self.trading_config.initial_balance
        self.units = 0
        self.in_position = False
        self.stop_loss_price = 0
        self.trailing_stop_price = 0
        self.portfolio_values = []
        self.trade_history = []

    def execute_trade(self, date: pd.Timestamp, price: float, state: str) -> Dict:
        """execute trading logic based on current state and price"""
        # check stop loss if in position
        if self.in_position:
            if price <= self.stop_loss_price:
                return self._exit_position(date, price, "stop_loss")

            # update trailing stop if price moves higher
            if price > self.trailing_stop_price / (1 - self.trading_config.trailing_stop_pct):
                self.trailing_stop_price = price * (1 - self.trading_config.trailing_stop_pct)
                self.stop_loss_price = max(self.stop_loss_price, self.trailing_stop_price)

        # only trade in bullish conditions
        if state == "bull":
            if not self.in_position:
                # enter with full position in bull market
                return self._enter_position(date, price, position_size=0.95)
        else:
            # exit if market turns non-bullish
            if self.in_position:
                return self._exit_position(date, price, "exit_non_bull")

        return None

    def _enter_position(self, date: pd.Timestamp, price: float, position_size: float = None) -> Dict:
        """enter a long position with optional position size override"""
        size = position_size if position_size is not None else self.trading_config.position_size
        position_value = self.current_balance * size
        self.units = position_value / price
        fee = position_value * self.trading_config.trading_fee
        self.current_balance -= position_value + fee
        self.in_position = True

        # set initial stops
        self.stop_loss_price = price * (1 - self.trading_config.stop_loss_pct)
        self.trailing_stop_price = price * (1 - self.trading_config.trailing_stop_pct)
        trade = {
            "date": date,
            "action": "buy",
            "price": price,
            "units": self.units,
            "fee": fee,
            "balance": self.current_balance,
        }
        self.trade_history.append(trade)
        return trade

    def _exit_position(self, date: pd.Timestamp, price: float, action: str) -> Dict:
        """Exit current position"""
        position_value = self.units * price
        fee = position_value * self.trading_config.trading_fee
        self.current_balance += position_value - fee

        trade = {
            "date": date,
            "action": action,
            "price": price,
            "units": self.units,
            "fee": fee,
            "balance": self.current_balance,
        }

        self.units = 0
        self.in_position = False
        self.trade_history.append(trade)
        return trade

    def calculate_dynamic_stops(self, price: float, current_volatility: float):
        """calculate dynamic stop losses based on volatility"""
        # base stops
        base_stop = self.trading_config.stop_loss_pct
        base_trailing = self.trading_config.trailing_stop_pct
        
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
        
        # calculate rolling volatility
        test_data['returns'] = np.log(test_data['close']).diff()
        test_data['volatility'] = test_data['returns'].rolling(self.volatility_window).std()
        
        # add state persistence to avoid frequent trading
        state_counter = 0
        current_state = None
        min_state_duration = 4  # minimum periods to hold a state
        
        for i in range(len(predictions)):
            date = predictions.index[i]
            price = test_data.loc[date, 'close']
            raw_state = predictions['state'].iloc[i]
            current_vol = test_data.loc[date, 'volatility']
            
            # state persistence logic
            if raw_state != current_state:
                state_counter += 1
                if state_counter >= min_state_duration:
                    current_state = raw_state
                    state_counter = 0
            else:
                state_counter = 0
            
            # execute trade with persisted state
            if self.in_position:
                # update stops dynamically based on current volatility
                self.calculate_dynamic_stops(price, current_vol)
                
            trade = self.execute_trade(date, price, current_state)
            
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
