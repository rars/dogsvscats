import numpy as np
import os
import bcolz

from sklearn.preprocessing import OneHotEncoder

import utils
reload(utils)
from utils import get_batches, get_data

import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import RMSprop
from keras.preprocessing import image

print('Starting train_dogscats.py')

class DataProvider(object):
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size
        
    def get_batches(self, batch_type, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return get_batches(
            os.path.join(self.path, batch_type),
            shuffle=False,
            batch_size=self.batch_size)
    
    def get_data(self, data_type):
        dirpath = os.path.join(self.path, data_type)
        return get_data(dirpath)
    
    def save_array(self, filename, data):
        filepath = os.path.join(self.model_path, filename)
        c = bcolz.carray(data, rootdir=filepath, mode='w')
        c.flush()
        
    def load_array(self, filename):
        filepath = os.path.join(self.model_path, filename)
        try:
            return bcolz.open(filepath)[:]
        except:
            return None

    @property
    def model_path(self):
        dirpath = os.path.join(self.path, 'models')
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
        return dirpath
    
    def get_weight_filepath(self, filename):
        return os.path.join(self.model_path, filename)

batch_size = 64
data_provider = DataProvider('data/dogscats', batch_size)

valid_batches = data_provider.get_batches('valid')
train_batches = data_provider.get_batches('train')

def fetch_data(data_type, filename):
    data = data_provider.load_array(filename)
    if data is None:
        data = data_provider.get_data(data_type)
        data_provider.save_array(filename, data)
    return data

# train_data = fetch_data('train', 'train_data.bc')
# valid_data = fetch_data('valid', 'valid_data.bc')

valid_classes = valid_batches.classes
train_classes = train_batches.classes

def onehot(x):
    return np.array(OneHotEncoder().fit_transform(x.reshape(-1, 1)).todense())

valid_labels = onehot(valid_classes)
train_labels = onehot(train_classes)

from vgg16 import Vgg16
vgg = Vgg16()
model = vgg.model

print('Dropping model layer')

model.pop()
for layer in model.layers:
    layer.trainable = False
model.add(Dense(2, activation='softmax'))

"""
gen = image.ImageDataGenerator()

train_batches = gen.flow(
    train_data,
    train_labels,
    batch_size=batch_size,
    shuffle=True)

valid_batches = gen.flow(
    valid_data,
    valid_labels,
    batch_size=batch_size,
    shuffle=False)
"""

def fit_model(model, train_batches, valid_batches, nb_epoch=1):
    model.fit_generator(
        train_batches,
        samples_per_epoch=train_batches.N,
        nb_epoch=nb_epoch,
        validation_data=valid_batches,
        nb_val_samples=valid_batches.N)

print('Training last layer of model...')    
    
opt = RMSprop(lr=0.1)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

fit_model(model, train_batches, valid_batches, nb_epoch=1)

print('Saving weights...')

model.save_weights(data_provider.get_weight_filepath('finetune1.h5'))

def get_first_dense_layer_index(model):
    layers = model.layers
    for i, layer in enumerate(layers):
        if type(layer) is Dense:
            return i
    return None

def enable_all_dense_layers_trainable(model):
    layers = model.layers
    first_dense_index = get_first_dense_layer_index(model)
    for layer in layers[first_dense_index:]:
        layer.trainable = True
        
enable_all_dense_layers_trainable(model)

print('Training dense layers')

K.set_value(opt.lr, 0.01)
fit_model(model, train_batches, valid_batches, 1)

print('Saving weights...')

model.save_weights(data_provider.get_weight_filepath('finetune2.h5'))

def chunk(values, n):
    for i in xrange(0, len(values), n):
        yield values[i:i+n]

def limit_range(val, low, high):
    if val > high:
        return high
    elif val < low:
        return low
    return val

def predict_dogscats(model, data_provider, batch_size):
    test_batches = data_provider.get_batches('test', batch_size)
    filename_batches = chunk(test_batches.filenames, batch_size)

    with open('submission.csv', 'w') as fout:
        fout.write('id,label\n')
        for i in range(int(12500 / batch_size)):
            imgs, labels = next(test_batches)
            filenames = next(filename_batches)
            ids = [int(f[8:][:-4]) for f in filenames]
            probabilities, categories, labels = vgg.predict(imgs, True)
            for p, c, i, l in zip(probabilities, categories, ids, labels):
                prob_dog = None
                if l == 'dogs':
                    prob_dog = p
                else:
                    prob_dog = 1.0 - p
                prob_dog = limit_range(prob_dog, 0.05, 0.95)
                line = '{0},{1:.4f}\n'.format(i, prob_dog)
                fout.write(line)
                print(line.strip())

print('Predicting dogscats...')
                
batch_size = 100
data_provider = DataProvider('data/dogscatsredux', 100)
predict_dogscats(model, data_provider, batch_size)
