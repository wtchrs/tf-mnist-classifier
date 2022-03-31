# tf-mnist-classifier

This repository is the classifier of hand-write numbers that is trained using mnist dataset.

## Getting Started

```sh
git clone https://github.com/wtchrs/tf-mnist-classifier
cd tf-mnist-classifier
```

The model is built with Tensorflow, so you can run it after installing Tensorflow package using the following command:

```sh
pip install tensorflow sklearn numpy matplotlib

# or, using requirements.txt
pip install -r requirements.txt
```

After installing required packages, run the following command:

```sh
# Just using pre-trained model:
python src/tf_mnist_cnn.py
python src/tf_cifar10_cnn.py

# Retraining model:
python src/tf_mnist_cnn.py 1
python src/tf_cifar10_cnn.py 1
```