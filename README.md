# vela crypto: hmm-based crypto trading model 

## overview
built a trading model for newly listed crypto assets using hidden markov models (hmm) with Baum-Welch algorithm. the idea is to catch market regimes early and trade based on state transitions.

## approach
- hmm for detecting market regimes (bull/bear/neutral)
- baum-welch algorithm doing the heavy lifting for parameter optimization
- feature engineering for better state classification
- proper backtesting because we're not savages

## getting started
```bash
git clone git@github.com:0xDVC/vela-crypto.git
cd vela-crypto
python -m poetry shell  # need poetry? 'python -m pip install poetry'
python -m poetry install
```

## how to use
1. fire up `vela_crypto_model.ipynb`
2. what you'll get:
   - feature correlation matrix
   - market state classifications
   - trading signals
   - performance metrics with charts

## under the hood

### data collection
- binance api integration (with proper rate limiting)
- creatively gather market cap data from a binance link since the api has no market cap data
- smart caching to avoid api abuse
- historical data on 15 min intervals for 97 days (90 for training, 7 for testing)
- 15m interval because its closest to 10m for granularity and also binance api has no 10m data

### feature engineering
picked these features after way too much testing:
- return
- volatility (high-low spread)
- rsi (because tradfi guys love it)
- bollinger bands (mean reversion stuff)
- momentum (catch some trends)

### hmm implementation
- went with 3 states (bull/bear/neutral) as that typically mimics most market regimes
- baum-welch algo for optimization
- multi-feature approach for state classification
- transition matrix learned from historical data (no curve fitting!)

### parameter tuning
- used aic/bic metrics for state selection
- standardscaler for feature normalization
- backtested different window sizes

## known issues & limitations
1. market regime detection:
   - bit slow to catch regime changes
   - sensitive to parameters (aren't they all?)
   - state classification could be better

2. trading strategy:
   - basic position sizing (needs work)
   - risk management is minimal
   - no stop-loss yet [from current progress, i can see good profits potential, however, i need to implement stop-loss to avoid big losses]

3. data stuff:
   - no access to market cap data via its api

## potential improvements
- [ ] better features:
  - order book integration
  - sentiment analysis ( research shows great strides in this area )

- [ ] actual risk management:
  - smart position sizing
  - stop-loss implementation
  - portfolio rebalancing

- [ ] model improvements:      
  - cross-validation for states   ```# this i could work out some improvements out.```

## project structure
```
.
├── README.md
├── vela_crypto_model.ipynb
├── poetry.lock
├── pyproject.toml
└── requirements.txt
```

## stats on random trade of a symbol
![Random trade](./image.png)

_you can see that the model is able to catch the market regime early and trade accordingly, however, its missing neutral states and acting on just two states. needs a bit of work there_
_stop-loss and take-profit are not implemented yet. that's why we see a sharp decline in the account value after the first few successful trades. of course, wrong signals are also a factor here._

## current performance
- sharpe ratio: -1.22 (working on it...)
- max drawdown: -10.66%
- daily volatility: 0.20
