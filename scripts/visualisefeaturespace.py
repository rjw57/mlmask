#!/usr/bin/env python3
"""
Visualise feature vector.

Usage:
    visualisefeaturespace.py [--outputdir DIR] <file>...

Options:
    -d, --outputdir DIR     Write results to directory DIR

For each input file XXXXX.jpg, a set of output files named
XXXXX_features_NNNN.jpg are generated which provide a visualisation of the
feature space.

"""
from __future__ import division

import os

import docopt
import dtcwt
import imageio
import numpy as np

import mlmask

def main():
    opts = docopt.docopt(__doc__)
    transform = dtcwt.Transform2d()
    out_dir = opts['--outputdir']

    for input_fn in opts['<file>']:
        print('Processing {}'.format(input_fn))
        feature_vector = mlmask.image_to_features(
            imageio.imread(input_fn), transform=transform)

        assert feature_vector.shape[-1] % 3 == 0
        n_images = feature_vector.shape[-1] // 3

        reds = feature_vector[..., :n_images]
        greens = feature_vector[..., n_images:2*n_images]
        blues = feature_vector[..., 2*n_images:]

        assert reds.shape[-1] == greens.shape[-1]
        assert blues.shape[-1] == greens.shape[-1]

        for image_idx in range(reds.shape[-1]):
            out_im = np.dstack((
                reds[..., image_idx], greens[..., image_idx],
                blues[..., image_idx]
            ))
            out_im -= out_im.min()
            out_im *= 255.0 / out_im.max()
            out_fn = '{}_features_{:04d}.jpg'.format(
                os.path.splitext(input_fn)[0], image_idx
            )

            if out_dir is not None:
                if not os.path.isdir(out_dir):
                    print('Creating {}'.format(out_dir))
                    os.makedirs(out_dir)
                out_fn = os.path.join(out_dir, os.path.basename(out_fn))

            print('Writing {}'.format(out_fn))
            imageio.imwrite(out_fn, out_im.astype(np.uint8))

if __name__ == '__main__':
    main()
