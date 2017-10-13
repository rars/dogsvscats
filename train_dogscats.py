#!/usr/bin/env python

import numpy as np
import os
import bcolz

from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle

import utils
reload(utils)
from utils import get_batches, get_data

import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras.preprocessing import image

print('Starting train_dogscats.py')

class DataProvider(object):
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size
        
    def get_batches(
            self,
            batch_type,
            gen=image.ImageDataGenerator(),
            shuffle=False,
            batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return get_batches(
            os.path.join(self.path, batch_type),
            gen=gen,
            shuffle=shuffle,
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
data_provider = DataProvider('../data/dogscats', batch_size)

valid_batches = data_provider.get_batches('valid', shuffle=False)
train_batches = data_provider.get_batches('train', shuffle=False)

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
# model.add(Dense(2, activation='softmax'))

def fetch_data(model, batches, filename):
    print('Attempting to load data from {0}'.format(filename))
    data = data_provider.load_array(filename)
    if data is None:
        print('Data for {0} not available'.format(filename))
        print('Getting data for {0}'.format(filename))
        data = model.predict_generator(batches, batches.nb_sample)
        print('Saving data to {0}'.format(filename))
        data_provider.save_array(filename, data)
        print('Saved data for {0}'.format(filename))
    return data

valid_features = fetch_data(model, valid_batches, 'valid_features.bc')
train_features = fetch_data(model, train_batches, 'train_features.bc')

train_features, train_labels = shuffle(train_features, train_labels)

print('{0}'.format(train_features.shape))

model = Sequential([Dense(2, activation='softmax', input_shape=(4096,))])

def fit_model(model, train_batches, valid_batches, nb_epoch=3):
    model.fit_generator(
        train_batches,
        samples_per_epoch=train_batches.N,
        nb_epoch=nb_epoch,
        validation_data=valid_batches,
        nb_val_samples=valid_batches.N)

print('Training last layer of model...')    
    
opt = RMSprop(lr=0.1)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(
    train_features,
    train_labels,
    nb_epoch=8,
    batch_size=batch_size,
    validation_data=(valid_features, valid_labels))

dense_layer = model.layers[0]

vgg = Vgg16()
model = vgg.model

model.pop()
model = Sequential([l for l in model.layers] + [dense_layer])

def split_model(model):
    layers = model.layers
    conv_indexes = [index for index, layer in enumerate(layers) if type(layer) is Convolution2D]
    last_conv_index = conv_indexes[-1]
    conv_layers = layers[:last_conv_index + 1]
    conv_model = Sequential(conv_layers)
    dense_layers = layers[last_conv_index + 1:]
    return conv_model, dense_layers

def process_weights(layer):
    return [o / 2.0 for o in layer.get_weights()]

def get_dense_model(conv_model, dense_layers):
    opt = RMSprop(lr = 0.00001)
    
    model = Sequential([
        MaxPooling2D(input_shape=conv_model.layers[-1].output_shape[1:]),
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.),
        Dense(4096, activation='relu'),
        Dropout(0.),
        Dense(2, activation='softmax')                       
        ])

    for l1, l2 in zip(model.layers, dense_layers):
        l1.set_weights(process_weights(l2))

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

conv_model, dense_layers = split_model(model)
dense_model = get_dense_model(conv_model, dense_layers)

valid_features = fetch_data(conv_model, valid_batches, 'valid_conv_features.bc')
train_features = fetch_data(conv_model, train_batches, 'train_conv_features.bc')

train_labels = onehot(train_classes)

train_features, train_labels = shuffle(train_features, train_labels)

for layer in dense_model.layers:
    layer.trainable = True

dense_model.fit(
    train_features,
    train_labels,
    nb_epoch=2,
    batch_size=batch_size,
    validation_data=(valid_features, valid_labels))

# Data augmentation

gen = image.ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True)

train_batches = data_provider.get_batches('train', gen=gen)
valid_batches = data_provider.get_batches('valid', shuffle=False)

for layer in conv_model.layers:
    layer.trainable = False

conv_model.add(dense_model)

print 'Training updated model'

conv_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
conv_model.fit_generator(
    train_batches,
    samples_per_epoch=train_batches.nb_sample,
    nb_epoch=8,
    validation_data=valid_batches,
    nb_val_samples=valid_batches.nb_sample)

exit()

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
fit_model(model, train_batches, valid_batches, nb_epoch=3)

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
    test_batches = data_provider.get_batches('test', batch_size=batch_size)
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
data_provider = DataProvider('../data/dogscatsredux', batch_size)
predict_dogscats(model, data_provider, batch_size)
