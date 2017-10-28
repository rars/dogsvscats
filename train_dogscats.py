#!/usr/bin/env python

import argparse
import os

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

def predict_dogscats(model, data_provider, batch_size, filename):
    test_batches = data_provider.get_batches('test', batch_size=batch_size)
    filename_batches = chunk(test_batches.filenames, batch_size)

    with open(filename, 'w') as fout:
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

def main(path, is_training, is_predicting, model_weights_file, submission_file):
    print('Starting train_dogscats.py')
    print('* using path: {0}'.format(path))
    print('* training: {0}, predicting: {1}'.format(is_training, is_predicting))

    batch_size = 64
    data_provider = DataProvider(os.path.join(path, 'dogscats'), batch_size)
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

    if not is_training:
        print('Loading model weights from {0}'.format(model_weights_file))
        model.load_weights(data_provider.get_weight_filepath(model_weights_file))
    else:
        model.train()
        print('Writing model weights to {0}'.format(model_weights_file))
        model.save_weights(data_provider.get_weight_filepath(model_weights_file))

    if is_predicting:
        print('Writing predictions to {0}'.format(submission_file))
        batch_size = 100
        data_provider = DataProvider(os.path.join(path, 'dogscatsredux'), batch_size)
        predict_dogscats(model, data_provider, batch_size, submission_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train',
        dest='train',
        action='store_true',
        help='If specified, trains the model.')
    parser.add_argument(
        '--predict',
        dest='predict',
        action='store_true',
        help='If specified, predicts classifications.')
    parser.add_argument(
        '--path',
        dest='path',
        required=True,
        help='The directory location containing training/validation/test data.')
    parser.add_argument(
        '--weightfile',
        dest='weightfile',
        required=True,
        help='File to store/read model weights from.')
    parser.add_argument(
        '--submission',
        dest='submission',
        required=True,
        help='Filename of the submission file to write predictions to.')
    args = parser.parse_args()
    main(args.path, args.train, args.predict, args.weightfile, args.submission)
