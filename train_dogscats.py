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
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop, Adam
from keras.preprocessing import image

print('Starting train_dogscats.py')
is_training = False

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

def train_last_layer_dense_model():
    opt = RMSprop(lr=0.1)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(
        train_features,
        train_labels,
        nb_epoch=8,
        batch_size=batch_size,
        validation_data=(valid_features, valid_labels))

if is_training:
    print('Training last layer of model...')    
    train_last_layer_dense_model()

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

def process_weights2(layer, prev_p, next_p):
    scale = (1.0 - prev_p) / (1.0 - next_p)
    return [o * scale for o in layer.get_weights()]

def copy_weights(from_layers, to_layers):
    for to_layer, from_layer in zip(to_layers, from_layers):
        to_layer.set_weights(from_layer.get_weights())

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

    copy_weights(dense_layers, model.layers)

    for layer in model.layers:
        layer.set_weights(process_weights(layer))

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def load_dense_weights_from_vgg16bn(model):
    from vgg16bn import Vgg16BN
    vgg16_bn = Vgg16BN()
    _, dense_layers = split_model(vgg16_bn.model)
    copy_weights(dense_layers, model.layers)

def get_batchnorm_model(conv_model, p):
    model = Sequential([
        MaxPooling2D(input_shape=conv_model.layers[-1].output_shape[1:]),
        Flatten(),
        Dense(4096, activation='relu'),
        BatchNormalization(),
        Dropout(p),
        Dense(4096, activation='relu'),
        BatchNormalization(),
        Dropout(p),
        Dense(1000, activation='softmax')
        ])

    load_dense_weights_from_vgg16bn(model)

    for layer in model.layers:
        if type(layer) == Dense:
            layer.set_weights(process_weights2(layer, 0.5, 0.6))

    model.pop()
    for layer in model.layers:
        layer.Trainable = False

    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

conv_model, dense_layers = split_model(model)
dense_model = get_batchnorm_model(conv_model, 0.6)

valid_features = fetch_data(conv_model, valid_batches, 'valid_conv_features.bc')
train_features = fetch_data(conv_model, train_batches, 'train_conv_features.bc')

train_labels = onehot(train_classes)

train_features, train_labels = shuffle(train_features, train_labels)

def train_last_layer():
    dense_model.fit(
        train_features,
        train_labels,
        nb_epoch=6,
        batch_size=batch_size,
        validation_data=(valid_features, valid_labels))

if is_training:
    print('Train last layer of dense model with batch normalization.')
    train_last_layer()
    
for layer in dense_model.layers:
    layer.trainable = True

def train_all_dense_layers():
    dense_model.compile(
        optimizer=Adam(lr=0.00001),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    dense_model.fit(
        train_features,
        train_labels,
        nb_epoch=12,
        batch_size=batch_size,
        validation_data=(valid_features, valid_labels))

if is_training:
    print('Train dense layers of model with batch normalization.')
    train_all_dense_layers()

# Join models
for layer in conv_model.layers:
    layer.trainable = False

for layer in dense_model.layers:
    layer.called_with = None
    conv_model.add(layer)
    conv_model.layers[-1].set_weights(layer.get_weights())

def train_final_model(final_model):
    # Data augmentation
    print('Training dense layers with data augmentation')
    gen = image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True)

    train_batches = data_provider.get_batches('train', shuffle=True, gen=gen)
    valid_batches = data_provider.get_batches('valid', shuffle=False)

    opt = Adam(lr = 0.0001)
    final_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    final_model.fit_generator(
        train_batches,
        samples_per_epoch=train_batches.nb_sample,
        nb_epoch=4,
        validation_data=valid_batches,
        nb_val_samples=valid_batches.nb_sample)

    opt = Adam(lr = 0.00001)
    final_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    final_model.fit_generator(
        train_batches,
        samples_per_epoch=train_batches.nb_sample,
        nb_epoch=4,
        validation_data=valid_batches,
        nb_val_samples=valid_batches.nb_sample)

    return final_model

def run(model, train=False):
    if not train:
        model.load_weights(data_provider.get_weight_filepath('final.h5'))
    else:
        final_model = train_final_model(model)
        final_model.save_weights(data_provider.get_weight_filepath('final.h5'))

run(conv_model, train=is_training)
final_model = conv_model

def chunk(values, n):
    for i in xrange(0, len(values), n):
        yield values[i:i+n]

def limit_range(val, low, high):
    if val > high:
        return high
    elif val < low:
        return low
    return val

def model_predict(model, imgs, classes, details=False):
    all_preds = model.predict(imgs)
    idxs = np.argmax(all_preds, axis=1)
    preds = [all_preds[i, idxs[i]] for i in range(len(idxs))]
    labels = [classes[idx] for idx in idxs]
    return np.array(preds), idxs, labels

def determine_classes():
    dp = DataProvider('../data/dogscats', 100)
    train_batches = dp.get_batches('train', shuffle=True)

    classes = list(iter(train_batches.class_indices))
    for c in train_batches.class_indices:
        classes[train_batches.class_indices[c]] = c
    print classes
    return classes

def predict_dogscats(model, data_provider, batch_size):
    test_batches = data_provider.get_batches('test', batch_size=batch_size)
    filename_batches = chunk(test_batches.filenames, batch_size)

    classes = determine_classes()
    with open('submission.csv', 'w') as fout:
        fout.write('id,label\n')
        for i in range(int(12500 / batch_size)):
            imgs, labels = next(test_batches)
            filenames = next(filename_batches)
            ids = [int(f[8:][:-4]) for f in filenames]

            # vgg.predict(imgs, True)
            probabilities, categories, labels = model_predict(model, imgs, classes, True)
            
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

if not is_training:
    print('Predicting dogscats...')
    batch_size = 100
    data_provider = DataProvider('../data/dogscatsredux', batch_size)
    predict_dogscats(final_model, data_provider, batch_size)
