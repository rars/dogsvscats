#!/usr/bin/env python

from dataprovider import DataProvider, TrainingDataProvider
from precompute import FeatureProvider
from model import DogsVsCatsModelBuilder

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

    with open('submission_tmp.csv', 'w') as fout:
        fout.write('id,label\n')
        for i in range(int(12500 / batch_size)):
            imgs, labels = next(test_batches)
            filenames = next(filename_batches)
            ids = [int(f[8:][:-4]) for f in filenames]
            probabilities, categories, labels = model.predict(imgs)            
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

def main():
    print('Starting train_dogscats.py')
    is_training = False

    batch_size = 64
    data_provider = DataProvider('../data/dogscats', batch_size)
    feature_provider = FeatureProvider(data_provider)
    training_data_provider = TrainingDataProvider(data_provider, feature_provider)

    builder = DogsVsCatsModelBuilder(
        training_data_provider,
        dropout=0.6,
        batch_size=batch_size)

    if is_training:
        print('Train last layer of dense model with batch normalization.')
        builder.train_last_layer()

    if is_training:
        print('Train dense layers of model with batch normalization.')
        builder.train_dense_layers()

    model = builder.build(data_provider)

    model_weights_file = 'final_tmp.h5'
    if not is_training:
        model.load_weights(data_provider.get_weight_filepath(model_weights_file))
    else:
        model.train()
        model.save_weights(data_provider.get_weight_filepath(model_weights_file))

    if not is_training:
        print('Predicting dogscats...')
        batch_size = 100
        data_provider = DataProvider('../data/dogscatsredux', batch_size)
        predict_dogscats(model, data_provider, batch_size)

if __name__ == '__main__':
    main()
