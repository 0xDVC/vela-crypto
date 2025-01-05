## project
short-term prediction model for recently listed, small-cap cryptocurrencies using the Hidden Markov Model with the Baum-Welch algorithm to detect market regimes.

## project overview
currently, the project is in the exploratory phase and most of the notes are in the ```explore.ipynb``` file.

## track progress
- [x] collect data
    - [x] collect eligible symbols
    - [x] collect historical data of symbols
- [x] feature engineering
    - [x] feature selection
    - [x] calculate features
    - [x] visualize features
- [ ] model training
- [ ] backtesting
- [ ] model evaluation
- [ ] organize cells into relative classes.
- [ ] model deployment on streamlit

## justification of choices:
- collection of data:
    - binance api: in order to interact with the binance api to retrieve the historical data of the symbols, we need to use the ```python-binance``` library.
    - previous attempts to retrieve the symbols by market cap estimation didn't suffice and was totally inaccurate. 
    - i realized that the library was insufficient for my needs, based on the limitations of the api in retrieving the market cap for the symbols.
    - that said, i found a subreddit post that made reference to a binance link. 
    - i integrated a script into the ```collect.py``` file to retrieve the market cap of the symbols from the link.
    - and the binance api was able to retrieve the historical data of the symbols.
    -  the data is retrieved from the binance api and stored in the ```data``` folder.
    -  the data retrieved has been validated to accurately represent the market cap of the symbols. I randomly selected 10 symbols and compared the data on [coingecko.com](https://www.coingecko.com/en/coins).


### get the data
```bash
    python -m core.collect
```


## project structure
_consistently under update_
```
├── README.md
├── explore.ipynb
├── poetry.lock
├── pyproject.toml
└── requirements.txt
```

## setup
### install
```bash
git clone git@github.com:0xDVC/vela-crypto.git
cd vela-crypto
python -m poetry init # 'python -m pip install poetry' if you don't have it
python -m poetry $(cat requirements.txt) 
```
