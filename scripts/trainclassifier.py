#!/usr/bin/env python3
"""
Train classifier on data

Usage:
    trainclassifier.py <datadir>

"""
from __future__ import division

import os
import pickle

import docopt
import numpy as np
import sklearn.ensemble as sklnensemble

def main():
    opts = docopt.docopt(__doc__)
    datadir = opts['<datadir>']

    classifier = sklnensemble.RandomForestClassifier()

    training_fn = os.path.join(datadir, 'training.npz')
    print('Loading training from {}'.format(training_fn))
    training = np.load(training_fn)

    X = np.vstack((training['foreground'], training['background']))
    y = np.hstack((
        np.ones(training['foreground'].shape[0]),
        np.zeros(training['background'].shape[0]),
    ))

    # Shuffle X and y
    print('Shuffling...')
    shuffle_idxs = np.arange(y.shape[0])
    np.random.shuffle(shuffle_idxs)
    X = X[shuffle_idxs, :]
    y = y[shuffle_idxs]

    print('Training sizes: X => {}, y => {}'.format(X.shape, y.shape))

    print('Training classifier...')
    classifier.fit(X, y)

    print('Evaluating...')
    y_pred = classifier.predict(X)

    n_y = y.shape[0]
    n_correct = np.count_nonzero(y_pred == y)
    n_false_pos = np.count_nonzero(np.logical_and(y_pred, np.logical_not(y)))
    n_false_neg = np.count_nonzero(np.logical_and(y, np.logical_not(y_pred)))
    print('Correct {}%'.format(100 * n_correct / n_y))
    print('False +ve: {}%'.format(100 * n_false_pos / n_y))
    print('False -ve: {}%'.format(100 * n_false_neg / n_y))

    classifier_fn = os.path.join(datadir, 'classifier.pickle')
    print('Saving classifier to {}'.format(classifier_fn))
    with open(classifier_fn, 'wb') as fobj:
        pickle.dump(classifier, fobj)

if __name__ == '__main__':
    main()
