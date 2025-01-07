from typing import Dict
import pandas as pd

class MarketRegimeAnalyzer:
    def analyze_regimes(self, symbols_data: Dict[str, pd.DataFrame], predictions: Dict[str, pd.DataFrame]) -> Dict:
        """analyze market regimes across all symbols"""
        regime_stats = {}
        
        for symbol, data in symbols_data.items():
            pred_df = predictions[symbol]
            
            regime_counts = pred_df['state'].value_counts()
            total_periods = len(pred_df)
            
            regime_stats[symbol] = {
                'bull_percent': (regime_counts.get('bull', 0) / total_periods) * 100,
                'bear_percent': (regime_counts.get('bear', 0) / total_periods) * 100,
                'neutral_percent': (regime_counts.get('neutral', 0) / total_periods) * 100,
                'dominant_regime': pred_df['state'].mode()[0],
                'avg_bull_prob': pred_df['bull_prob'].mean(),
                'volatility': data['close'].pct_change().std() * 100
            }
            
        return regime_stats 