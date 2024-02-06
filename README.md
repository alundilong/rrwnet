# RRWNet

This is the official repository for the preprint ["RRWNet: Recursive Refinement Network for Effective Retinal Artery/Vein Segmentation and Classification"](https://doi.org/10.48550/arXiv.2402.03166), by José Morano, Guilherme Aresta, and Hrvoje Bogunović.


**IMPORTANT: Training, testing, and evaluation code to be uploaded upon publication.**



## Predictions and Weights

The predictions for the different datasets as well as the weights for the proposed RRWNet model can be found at the following link:

- <https://drive.google.com/drive/folders/1Pz0z-OxzEft5EWGbZ3MeeqQNWqvv2ese?usp=sharing>


The model trained on the RITE dataset was trained using the original image resolution, while the model trained on HRF was trained using images resized to a width of 1024 pixels. The weights for the RITE dataset are named `rrwnet_RITE_1.pth`, while the weights for the HRF dataset are named `rrwnet_HRF_0.pth`.
Please note that the size of the images used for training is important when using the weights for predictions.


## Setting up the environment

The code was tested using Python 3.10.10. The following instructions are for setting up the environment using `pyenv` and `pip`.
However, the code should work with other Python versions and package managers.
Just make sure to install the required packages listed in `requirements.txt`.

Install `pyenv`.
```sh
curl https://pyenv.run | bash
```

Install `clang`. _E.g._:
```sh
sudo dnf install clang
```

Install Python version 3.10.10.
```sh
CC=clang pyenv install -v 3.10.10
```

Create and activate Python environment.
```sh
~/.pyenv/versions/3.10.10/bin/python3 -m venv venv/
source venv/bin/activate  # bash
. venv/bin/activate.fish  # fish
```

Update `pip`.

```sh
pip install --upgrade pip
```

Install requirements using `requirements.txt`.

```sh
pip3 install -r requirements.txt
```

## Preprocessing (optional)

You can preprocess the images offline using the `preprocessing.py` script. The script will enhance the images and masks and save them in the specified directory.

```bash
python3 preprocessing.py --images-path data/images/ --masks-path data/masks/ --save-path data/enhanced
```


## Get Predictions

To get predictions using the provided weights, run the `get_predictions.py` script. The script will save the predictions in the specified directory.
If the images were not previously preprocessed, you can use the `--preprocess` flag to preprocess the images on the fly.

```bash
python3 get_predictions.py --weights rrwnet_RITE_1.pth --images-path data/images/ --masks-path data/masks/ --save-path predictions/ --preprocess
```
