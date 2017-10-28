import os
import bcolz
import numpy as np

from keras.preprocessing import image
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle

from utils import get_batches, get_data
from precompute import FeatureProvider

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

class FeatureSet(object):
    def __init__(self, features, classes, labels):
        self._features = features
        self._classes = classes
        self._labels = labels

    def shuffle(self):
        self._features, self._labels = shuffle(self._features, self._labels)

    @property
    def features(self):
        return self._features

    @property
    def classes(self):
        return self._classes

    @property
    def labels(self):
        return self._labels

def onehot(x):
    return np.array(OneHotEncoder().fit_transform(x.reshape(-1, 1)).todense())

class TrainingDataProvider(object):
    def __init__(self, data_provider, feature_provider):
        self._data_provider = data_provider
        self._feature_provider = feature_provider

    def get_feature_set(self, datasource, model, filename):
        batches = self._data_provider.get_batches(datasource, shuffle=False)
        features = self._feature_provider.get_features(model, batches, filename)
        classes = list(iter(batches.class_indices))
        for c in batches.class_indices:
            classes[batches.class_indices[c]] = c
        labels = onehot(batches.classes)
        return FeatureSet(features, classes, labels)
