#!/usr/bin/env python3
"""
Generate training data

Usage:
    gentraining.py <datadir> <outputnpz>

"""
import os
import glob

import docopt
import dtcwt
import imageio
import numpy as np
import skimage.transform as sktrans

class ReservoirSampler(object):
    """A simple reservoir sampler which will sample k items uniformly from a
    list of unknown length. Implemented to take numpy arrays and sample rows.

    """
    def __init__(self, k):
        self.samples = None # allocated on first call to sample()
        self.k = k
        self._idx = 0 # Index of next datum

    @property
    def sample_count(self):
        return self._idx

    def sample(self, data):
        data = np.atleast_2d(data)

        if self.samples is None:
            self.samples = np.zeros(
                (self.k,) + data.shape[1:], dtype=data.dtype)

        for datum in data:
            if self._idx < self.k:
                # Fill reservoir with first k samples
                self.samples[self._idx, ...] = datum
            else:
                # Randomly replace elements in reservoid
                r = np.random.randint(0, self._idx)
                if r < self.k:
                    self.samples[r, ...] = datum

            self._idx += 1

def resize(image, shape):
    """Resize an input array."""
    sc = np.abs(image).max()
    return sktrans.resize(image / sc, shape, mode='reflect') * sc

def main():
    opts = docopt.docopt(__doc__)
    datadir = opts['<datadir>']
    output_fn = opts['<outputnpz>']

    training_size = 1000000
    foreground_training = ReservoirSampler(training_size)
    background_training = ReservoirSampler(training_size)

    transform = dtcwt.Transform2d()

    for input_fn in glob.glob(os.path.join(datadir, 'input', '*.JPG')):
        trimap_fn = os.path.join(
            datadir, 'trimap',
            os.path.splitext(os.path.basename(input_fn))[0] + '.png'
        )
        if not os.path.isfile(trimap_fn):
            continue

        print('Input: ' + input_fn)
        print('Trimap: ' + trimap_fn)

        input_im = imageio.imread(input_fn)
        trimap_im = imageio.imread(trimap_fn)

        # Only use first channel of trimap_im
        if len(trimap_im.shape) > 2:
            trimap_im = trimap_im[..., 0]

        assert len(trimap_im.shape) == 2
        assert len(input_im.shape) == 3
        assert input_im.shape[2] == 3

        # Extract indices of +ve and -ve training samples
        trimap_im = trimap_im.reshape((-1,))
        foreground_mask = trimap_im > 224
        background_mask = trimap_im < 32

        # Compute DTCWT for each channel
        feature_vector = []
        for c_idx in range(input_im.shape[2]):
            print('Transforming channel {}/{}'.format(
                c_idx+1, input_im.shape[2]))
            input_chan = input_im[..., c_idx].astype(np.float32)
            input_chan /= 255.0
            input_chan = np.asarray(input_chan)
            input_dtcwt = transform.forward(input_chan, nlevels=6)

            # We re-scale each channel of the DTCWT to the input image size. We
            # take the maximum absolute value along each direction.
            feature_vector.append(resize(
                input_dtcwt.lowpass, input_chan.shape))

            # For each highpass...
            for hp in input_dtcwt.highpasses:
                feature_vector.append(resize(
                    np.max(np.abs(hp), axis=-1), input_chan.shape))

        # Stack feature vector up
        feature_vector = np.dstack(feature_vector)

        # Reshape to have one feature vector per row
        feature_vector = feature_vector.reshape((-1, feature_vector.shape[-1]))

        # Update samples
        foreground_training.sample(feature_vector[foreground_mask, :])
        background_training.sample(feature_vector[background_mask, :])

        print('Foreground sample count: {}'.format(
            foreground_training.sample_count))
        print('Background sample count: {}'.format(
            background_training.sample_count))

        # Save as we go
        print('Saving to {}'.format(output_fn))
        np.savez_compressed(output_fn,
                foreground=foreground_training.samples,
                foreground_sample_size=foreground_training.sample_count,
                background=background_training.samples,
                background_sample_size=background_training.sample_count)

if __name__ == '__main__':
    main()
