import np

from vgg16 import Vgg16

import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop, Adam
from keras.preprocessing import image

def set_trainable(layer, trainable):
    layer.trainable = trainable

def split_model(model):
    layers = model.layers
    conv_indexes = [index for index, layer in enumerate(layers) if type(layer) is Convolution2D]
    last_conv_index = conv_indexes[-1]
    conv_layers = layers[:last_conv_index + 1]
    conv_model = Sequential(conv_layers)
    dense_layers = layers[last_conv_index + 1:]
    return conv_model, dense_layers

def copy_weights(from_layers, to_layers):
    for to_layer, from_layer in zip(to_layers, from_layers):
        to_layer.set_weights(from_layer.get_weights())

def load_dense_weights_from_vgg16bn(model):
    from vgg16bn import Vgg16BN
    vgg16_bn = Vgg16BN()
    _, dense_layers = split_model(vgg16_bn.model)
    copy_weights(dense_layers, model.layers)

def process_weights(layer, prev_p, next_p):
    scale = (1.0 - prev_p) / (1.0 - next_p)
    return [o * scale for o in layer.get_weights()]

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
            layer.set_weights(process_weights(layer, 0.5, 0.6))

    model.pop()
    for layer in model.layers:
        set_trainable(layer, False)

    model.add(Dense(2, activation='softmax'))
    return model

class DogVsCatsModelBuilder(object):
    def __init__(self, training_data_provider, dropout=0.6, batch_size=64):
        vgg = Vgg16()
        model = _vgg.model

        conv_model, _ = split_model(model)
        dense_model = get_batchnorm_model(conv_model, dropout)

        self._conv_model = conv_model
        self._dense_model = dense_model
        self._batch_size = batch_size
        self._train_feature_set = training_data_provider.get_feature_set(
            'train',
            conv_model,
            'train_conv_features.bc'
        )
        self._train_feature_set.shuffle()
        self._valid_feature_set = training_data_provider.get_feature_set(
            'valid',
            conv_model,
            'valid_conv_features.bc'
        )

    def build(self, data_provider):
        for layer in self._conv_model.layers:
            set_trainable(layer, False)

        for layer in self._dense_model.layers:
            layer.called_with = None
            self._conv_model.add(layer)
            self._conv_model.layers[-1].set_weights(layer.get_weights())

        return DogsVsCatsModel(
            data_provider,
            self._conv_model,
            self._train_feature_set.classes)

    def train_last_layer(self, lr=0.001):
        for layer in self._dense_model.layers:
            set_trainable(layer, False)
        set_trainable(self._dense_model.layers[-1], True)

        self._dense_model.compile(
            optimizer=Adam(lr=lr),
            loss='categorical_crossentropy',
            metrics=['accuracy'])

        self._dense_model.fit(
            self._train_feature_set.features,
            self._train_feature_set.labels,
            nb_epoch=6,
            batch_size=self._batch_size,
            validation_data=(
                self._valid_feature_set.features,
                self._valid_feature_set.labels))

    def train_dense_layers(self, lr=0.00001):
        for layer in self._dense_model.layers:
            set_trainable(layer, True)
        
        self._dense_model.compile(
            optimizer=Adam(lr=lr),
            loss='categorical_crossentropy',
            metrics=['accuracy'])

        self._dense_model.fit(
            self._train_feature_set.features,
            self._train_feature_set.labels,
            nb_epoch=12,
            batch_size=self._batch_size,
            validation_data=(
                self._valid_feature_set.features,
                self._valid_feature_set.labels))

class DogsVsCatsModel(object):
    def __init__(self, data_provider, model, classes):
        self._data_provider = data_provider
        self._model = model
        self._classes = classes

    def train(self):
        print('Training dense layers with data augmentation')
        gen = image.ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True)

        train_batches = self._data_provider.get_batches('train', shuffle=True, gen=gen)
        valid_batches = self._data_provider.get_batches('valid', shuffle=False)

        opt = Adam(lr = 0.0001)
        self._model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['accuracy'])

        self._model.fit_generator(
            train_batches,
            samples_per_epoch=train_batches.nb_sample,
            nb_epoch=4,
            validation_data=valid_batches,
            nb_val_samples=valid_batches.nb_sample)

        opt = Adam(lr = 0.00001)
        self._model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['accuracy'])

        self._model.fit_generator(
            train_batches,
            samples_per_epoch=train_batches.nb_sample,
            nb_epoch=4,
            validation_data=valid_batches,
            nb_val_samples=valid_batches.nb_sample)
    
    def load_weights(self, filename):
        self._model.load_weights(filename)

    def save_weights(self, filename):
        self._model.save_weights(filename)

    def predict(self, imgs):
        all_preds = self._model.predict(imgs)
        idxs = np.argmax(all_preds, axis=1)
        preds = [all_preds[i, idxs[i]] for i in range(len(idxs))]
        labels = [self._classes[idx] for idx in idxs]
        return np.array(preds), idxs, labels
