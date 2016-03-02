#!/usr/bin/env python3
"""
Run classifier on data

Usage:
    classify.py <datadir>

"""
import glob
import os
import pickle

import docopt
import dtcwt
import imageio
import numpy as np
import sklearn.ensemble as sklnensemblea

import mlmask

def main():
    opts = docopt.docopt(__doc__)
    datadir = opts['<datadir>']
    outputdir = os.path.join(datadir, 'perpixel')

    if not os.path.isdir(outputdir):
        os.makedirs(outputdir)

    classifier_fn = os.path.join(datadir, 'classifier.pickle')
    print('Loading classifier from {}'.format(classifier_fn))
    with open(classifier_fn, 'rb') as fobj:
        classifier = pickle.load(fobj)

    transform = dtcwt.Transform2d()

    for input_fn in glob.glob(os.path.join(datadir, 'input', '*.JPG')):
        mask_base = os.path.join(outputdir,
            os.path.splitext(os.path.basename(input_fn))[0])
        mask_fn = mask_base + '.png'
        if os.path.isfile(mask_fn):
            print('Skipping since {} already exists'.format(mask_fn))
            continue

        print('Input: ' + input_fn)
        input_im = imageio.imread(input_fn)
        feature_vector = mlmask.image_to_features(input_im,
            transform=transform, nlevels=6)

        # Classify
        feature_vector = feature_vector.reshape((-1, feature_vector.shape[-1]))
        mask_pred = classifier.predict(feature_vector)
        mask_pred = mask_pred.reshape(input_im.shape[:2]).astype(np.int)

        print('Writing mask to: {}'.format(mask_fn))

        imageio.imwrite(mask_fn,
            np.where(mask_pred == 0, 0, 255).astype(np.uint8))

        imageio.imwrite(mask_base + '-visualisation.jpg',
            mlmask.visualise_mask(input_im, mask_pred))

if __name__ == '__main__':
    main()
