
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