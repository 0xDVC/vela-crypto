from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import Dict, List



## Trading Strategy
# 	Buy/Hold if Bullish.
# 	Sell/Short if Bearish.
# 	Hold cash if Neutral.


@dataclass
class BacktestConfig:
    initial_balance: float = 10000.0  # USDT
    position_size: float = 0.1        # 10% of balance per trade
    stop_loss: float = 0.02          # 2% stop loss
    take_profit: float = 0.03        # 3% take profit
    maker_fee: float = 0.001         # 0.1% maker fee
    taker_fee: float = 0.001         # 0.1% taker fee

class BacktestStrategy:
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        
    def should_trade(self, state: str, confidence: float, volatility: float, rsi: float) -> bool:
        """Determine if conditions are right for trading based on market state"""
        # Base conditions for all states
        meets_volatility = 0.005 < volatility < 0.03
        
        if state == 'bull':
            # More aggressive in bull markets
            return (confidence > 0.70 and  # Lower confidence threshold
                    meets_volatility and
                    rsi > 45 and rsi < 75)  # Wider RSI range
                
        elif state == 'bear':
            # Very conservative in bear markets
            return (confidence > 0.85 and  # Higher confidence needed
                    meets_volatility and
                    rsi < 35)  # Only trade strong oversold conditions
                
        elif state == 'neutral':
            # Moderate approach in neutral markets
            return (confidence > 0.80 and  # Moderate confidence threshold
                    meets_volatility and
                    (rsi < 35 or rsi > 65))  # Trade only at extremes
        
        return False

    def run_backtest(self, price_data: pd.DataFrame, predictions: pd.DataFrame) -> Dict:
        """Run backtest with simplified state-based strategy"""
        print("\nRunning backtest...")
        
        # Initialize portfolio tracking
        portfolio = pd.DataFrame(index=price_data.index)
        portfolio['value'] = self.config.initial_balance
        portfolio['position'] = 0.0
        portfolio['trade'] = ''
        portfolio['buy_hold'] = self.config.initial_balance * (price_data['close'] / price_data['close'].iloc[0])
        
        position_size = self.config.initial_balance * self.config.position_size
        in_position = False
        entry_price = 0
        
        for i in range(1, len(price_data)):
            current_price = price_data['close'].iloc[i]
            state = predictions['label'].iloc[i]
            confidence = predictions['confidence'].iloc[i]
            
            # Copy previous values
            portfolio.loc[portfolio.index[i], 'value'] = portfolio['value'].iloc[i-1]
            portfolio.loc[portfolio.index[i], 'position'] = portfolio['position'].iloc[i-1]
            
            # Simple state-based trading rules
            if not in_position:  # Currently in cash
                if state == 'bull' and confidence > 0.6:  # Enter long in bull market
                    entry_price = current_price
                    in_position = True
                    portfolio.loc[portfolio.index[i], 'trade'] = 'entry'
                    portfolio.loc[portfolio.index[i], 'position'] = float(position_size / current_price)
                    
            elif in_position:  # Currently holding position
                pnl = (current_price - entry_price) / entry_price
                
                # exit conditions:
                # state changes to bear
                # state changes to neutral
                # stop loss hit (-2%)
                # take profit hit (+3%)
                if (state in ['bear', 'neutral'] and confidence > 0.6) or \
                   pnl <= -0.02 or pnl >= 0.03:
                    
                    portfolio.loc[portfolio.index[i], 'value'] *= (1 + pnl * self.config.position_size)
                    in_position = False
                    portfolio.loc[portfolio.index[i], 'trade'] = 'exit'
                    portfolio.loc[portfolio.index[i], 'position'] = 0
        
        # calculate performance metrics
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
        
        # print performance summary
        print("\nPerformance Summary:")
        print(f"Total Return: {results['performance']['total_return']:.2%}")
        print(f"Buy & Hold Return: {results['performance']['buy_hold_return']:.2%}")
        print(f"Sharpe Ratio: {results['performance']['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['performance']['max_drawdown']:.2%}")
        print(f"Win Rate: {results['performance']['win_rate']:.2%}")
        print(f"Total Trades: {results['performance']['total_trades']}")
        
        return results
    
    def _calculate_sharpe(self, returns: pd.Series) -> float:
        """Calculate annualized Sharpe ratio for 15-min returns"""
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        
        # annualize for 15-min returns (4 periods per hour, 24 hours, 365 days)
        return np.sqrt(365*24*4) * returns.mean() / returns.std()
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        return (returns / returns.cummax() - 1).min()