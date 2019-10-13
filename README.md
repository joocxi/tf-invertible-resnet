# Invertible Residual Networks in TensorFlow
A TensorFlow implementation of [Invertible Residual Networks](https://arxiv.org/pdf/1811.00995.pdf)

## Installation
First, we need to create our Python `3.6` virtual environment using `virtualenv` and install all necessary packages stored in `requirements.txt`
```bash
pip install virtualenv
virtualenv -p python3 .env
source .env/bin/activate
pip install -r requirements.txt
```

## Running
To prepare dataset 
```bash
python main.py --mode prepare --dataset <dataset-name>
```
To test spectral normalization
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

To test invertible residual net forward pass
```bash
python main.py --mode iresnet
```

To test training pipeline
```bash
python main.py --mode debug
```

## TODOs
- [x] General architecture
- [x] Spectral norm
- [x] Trace approximation
- [x] Block inversion
- [x] Loss functions
- [x] Training pipeline
- [ ] Multi-scale
- [ ] Injective padding
- [ ] Dimension splitting
- [ ] Training results
- [ ] Actnorm (optional)
- [ ] To TensorFlow 2.0

**References**