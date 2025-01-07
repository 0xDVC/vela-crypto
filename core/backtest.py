from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import Dict


@dataclass
class BacktestConfig:
    initial_balance: float = 10000.0
    position_size: float = 0.2        # Base position size
    base_stop_loss: float = 0.02      # 2% base stop loss
    base_take_profit: float = 0.03    # 3% base take profit
    maker_fee: float = 0.001         # 0.1% maker fee
    taker_fee: float = 0.001         # 0.1% taker fee
    min_bull_prob: float = 0.65      # Minimum bull probability
    max_positions: int = 3           # Maximum concurrent positions
    min_regime_strength: float = 0.7  # Minimum regime strength
    min_volume: float = 1000000.0     # 1M USDT daily volume minimum

class BacktestStrategy:
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        
    def run_backtest(self, price_data: pd.DataFrame, predictions: pd.DataFrame) -> Dict:
        """Enhanced backtest with regime-based position sizing"""
        portfolio = self._initialize_portfolio(price_data)
        active_positions = 0
        entry_price = None  # Track entry price
        last_trade_idx = 0  # Track last trade index
        min_hold_bars = 24  # Minimum hold time in bars
        daily_trades = 0    # Initialize daily trades counter
        
        for i in range(1, len(portfolio)):
            # Get current state and probabilities
            state = predictions['state'].iloc[i]
            bull_prob = predictions['bull_prob'].iloc[i]
            regime_strength = max(predictions[['bull_prob', 'bear_prob', 'neutral_prob']].iloc[i])
            
            # Dynamic risk adjustment based on regime strength
            stop_loss = self.config.base_stop_loss * (1 - (regime_strength - 0.5))
            take_profit = self.config.base_take_profit * (1 + (regime_strength - 0.5))
            
            # Entry/Exit logic with minimum hold time
            if i - last_trade_idx >= min_hold_bars:
                # Entry conditions
                if portfolio['position'].iloc[i-1] == 0:  # Not in position
                    # Get dynamic threshold
                    current_threshold = predictions['bull_threshold'].iloc[i]
                    
                    # Bull regime entry with dynamic threshold
                    if (state == 'bull' and 
                        bull_prob > current_threshold * 1.1):  # Require 10% above average
                        
                        position_size = self.config.position_size
                        entry_amount = portfolio['cash'].iloc[i-1] * position_size
                        shares = entry_amount / portfolio['price'].iloc[i] * (1 - self.config.maker_fee)
                        
                        portfolio.loc[portfolio.index[i], 'position'] = shares
                        portfolio.loc[portfolio.index[i], 'cash'] = portfolio['cash'].iloc[i-1] - entry_amount
                        portfolio.loc[portfolio.index[i], 'trade'] = 'buy'
                        entry_price = portfolio['price'].iloc[i]
                        last_trade_idx = i
                        daily_trades += 1
                
                # Exit conditions
                elif portfolio['position'].iloc[i-1] > 0:
                    current_threshold = predictions['bull_threshold'].iloc[i]
                    
                    # Exit when probability falls significantly below average
                    if (state == 'bear' or 
                        bull_prob < current_threshold * 0.9):  # Exit when 10% below average
                        
                        portfolio.loc[portfolio.index[i], 'position'] = 0
                        portfolio.loc[portfolio.index[i], 'cash'] += (
                            portfolio['position'].iloc[i-1] * 
                            portfolio['price'].iloc[i] * 
                            (1 - self.config.taker_fee)
                        )
                        portfolio.loc[portfolio.index[i], 'trade'] = 'sell'
                        last_trade_idx = i
                        daily_trades += 1
            
            # Update portfolio values
            portfolio = self._update_portfolio_values(portfolio, i)
        
        return self._calculate_results(portfolio, predictions)

    def _can_enter_position(self, state: str, bull_prob: float, regime_strength: float) -> bool:
        """Check if we can enter a new position"""
        return (
            state == 'bull' and 
            bull_prob > self.config.min_bull_prob and
            regime_strength > self.config.min_regime_strength
        )

    def _should_exit_position(self, pnl: float, state: str, bull_prob: float, 
                            stop_loss: float, take_profit: float) -> bool:
        """Determine if we should exit the position"""
        return (
            pnl < -stop_loss or                  # Stop loss hit
            pnl > take_profit or                 # Take profit hit
            state == 'bear' or                   # Bear regime
            bull_prob < self.config.min_bull_prob * 0.9  # Weakening bull probability
        )

    def _initialize_portfolio(self, price_data: pd.DataFrame):
        """Initialize the portfolio"""
        portfolio = pd.DataFrame(index=price_data.index)
        portfolio['price'] = price_data['close']
        portfolio['position'] = 0.0
        portfolio['trade'] = ''
        portfolio['value'] = self.config.initial_balance
        portfolio['cash'] = self.config.initial_balance
        portfolio['holdings'] = 0.0
        portfolio['drawdown'] = 0.0
        
        # Add buy & hold calculation
        portfolio['buy_hold'] = self.config.initial_balance * (
            (1 + price_data['close'].pct_change().fillna(0)).cumprod()
        )
        
        return portfolio

    def _calculate_pnl(self, portfolio: pd.DataFrame, i: int, entry_price: float) -> float:
        """Calculate profit and loss"""
        return (portfolio['price'].iloc[i] - entry_price) / entry_price

    def _enter_position(self, portfolio: pd.DataFrame, i: int, entry_amount: float) -> pd.DataFrame:
        """Enter a new position"""
        shares = entry_amount / portfolio['price'].iloc[i] * (1 - self.config.maker_fee)
        portfolio.loc[portfolio.index[i], 'position'] = shares
        portfolio.loc[portfolio.index[i], 'cash'] = portfolio['cash'].iloc[i-1] - entry_amount
        portfolio.loc[portfolio.index[i], 'trade'] = 'buy'
        return portfolio

    def _exit_position(self, portfolio: pd.DataFrame, i: int, current_pnl: float) -> pd.DataFrame:
        """Exit the position"""
        portfolio.loc[portfolio.index[i], 'position'] = 0
        portfolio.loc[portfolio.index[i], 'cash'] += (
            portfolio['position'].iloc[i-1] * 
            portfolio['price'].iloc[i] * 
            (1 - self.config.taker_fee)
        )
        portfolio.loc[portfolio.index[i], 'trade'] = 'sell'
        return portfolio

    def _update_portfolio_values(self, portfolio: pd.DataFrame, i: int) -> pd.DataFrame:
        """Update portfolio values"""
        portfolio.loc[portfolio.index[i], 'value'] = (
            portfolio['cash'].iloc[i] + 
            portfolio['position'].iloc[i] * portfolio['price'].iloc[i]
        )
        return portfolio

    def _calculate_results(self, portfolio: pd.DataFrame, predictions: pd.DataFrame) -> Dict:
        """Calculate performance metrics"""
        returns = portfolio['value'].pct_change().dropna()
        total_return = (portfolio['value'].iloc[-1] / self.config.initial_balance) - 1
        buy_hold_return = (portfolio['buy_hold'].iloc[-1] / self.config.initial_balance) - 1
        
        trades = portfolio[portfolio['trade'] != '']
        win_trades = trades[trades['value'] > trades['value'].shift(1)].dropna()
        
        results = {
            'portfolio': portfolio,
            'performance': {
                'total_return': total_return,
                'buy_hold_return': buy_hold_return,
                'sharpe_ratio': self._calculate_sharpe(returns) if len(returns) > 1 else 0,
                'max_drawdown': self._calculate_max_drawdown(portfolio['value']),
                'win_rate': len(win_trades) / len(trades) if len(trades) > 0 else 0,
                'total_trades': len(trades)
            }
        }
        #    Print trade summary
        print("\nTrade Summary:")
        trades = portfolio[portfolio['trade'] != '']
        for idx, trade in trades.iterrows():
            print(f"{idx}: {trade['trade'].upper()} at {trade['price']:.4f}")

        return results
        

    def _calculate_sharpe(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio with proper error handling"""
        if len(returns) < 2:  # Need at least 2 points for std
            return 0
        
        std = returns.std()
        if std == 0 or np.isnan(std):  # Handle zero/nan std case
            return 0
        
        return returns.mean() / std * np.sqrt(252)  # Annualized

    def _calculate_max_drawdown(self, values: pd.Series) -> float:
        """Calculate maximum drawdown"""
        peak = values.expanding(min_periods=1).max()
        drawdown = (values - peak) / peak
        return drawdown.min()