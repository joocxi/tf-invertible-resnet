# Invertible Residual Networks in TensorFlow
A TensorFlow implementation of [Invertible Residual Networks](https://arxiv.org/abs/1811.00995)

## Installation
First, we need to create our Python `3.6` virtual environment using `virtualenv` and install all necessary packages stored in `requirements.txt`
```bash
pip install virtualenv
virtualenv -p python3 .env
source .env/bin/activate
pip install -r requirements.txt
```

## Running
To test loading dataset 
```bash
python main.py --mode data --dataset <dataset-name>
```
To test spectral normalization implementation
```bash
python main.py --mode sn
```

To test trace approximation
```bash
python main.py --mode trace
```

To test block inversion
```bash
python main.py --mode inverse
```
## TODOs
- [x] General architecture
- [x] Spectral norm
- [x] Trace approximation
- [ ] Loss functions
- [ ] Training/testing pipeline
- [ ] Dimension splitting
- [ ] Actnorm (optional)
- [ ] To TensorFlow 2.0

**References**