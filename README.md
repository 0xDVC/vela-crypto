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
