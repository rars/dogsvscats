# Dogs vs Cats Classifier

A Dogs vs Cats classifier written in Python using Keras.

## Running

    python train_dogscats.py --train --predict --path ../data --submission submission.csv --weightsfile model_weights.h5

The above assumes the following directory structure:

* data/dogscats/train/{cats, dogs}/*.jpg
* data/dogscats/valid/{cats, dogs}/*.jpg
* data/dogscatsredux/test/unknown/*.jpg

In the above case, `--path` should point to the 'data' directory.

The files `utils.py`, `vgg16.py` and `vgg16bn.py` are taken from the fast.ai github repository [here](https://github.com/fastai/courses/tree/master/deeplearning1/nbs).