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
python main.py --mode download_data --dataset <dataset-name>
```

**References**