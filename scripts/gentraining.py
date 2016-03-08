#!/usr/bin/env python3
"""
Generate training data.

Usage:
    gentraining.py <datadir>

"""
import os
import glob

import docopt
import dtcwt
import imageio
import numpy as np
import skimage.measure as skmeas
import skimage.transform as sktrans

import mlmask

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
    output_fn = os.path.join(datadir, 'training.npz')

    training_size = 10000000
    foreground_training = ReservoirSampler(training_size)
    background_training = ReservoirSampler(training_size)

    transform = dtcwt.Transform2d()

    for input_fn in glob.glob(os.path.join(datadir, 'input', '*.JPG')):
        input_base = os.path.splitext(os.path.basename(input_fn))[0]
        trimap_fn = os.path.join(datadir, 'trimap', input_base + '.png')
        if not os.path.isfile(trimap_fn):
            continue
        superpixel_fn = os.path.join(datadir, 'superpixel', input_base + '.npz')
        if not os.path.isfile(superpixel_fn):
            continue

        print('Input: ' + input_fn)
        print('Trimap: ' + trimap_fn)
        print('Superpixel: ' + superpixel_fn)

        input_im = imageio.imread(input_fn)
        trimap_im = imageio.imread(trimap_fn)

        # Only use first channel of trimap_im
        if len(trimap_im.shape) > 2:
            trimap_im = trimap_im[..., 0]

        assert len(trimap_im.shape) == 2
        assert len(input_im.shape) == 3
        assert input_im.shape[2] == 3

        sp_labels = np.load(superpixel_fn)['labels']

        # Modify trimap with superpixels
        modified_trimap = np.copy(trimap_im)
        for props in skmeas.regionprops(sp_labels, trimap_im):
            coords = props.coords
            rows, cols = coords[:, 0], coords[:, 1]
            trimap_pxs = trimap_im[rows, cols]
            n_pxs = trimap_pxs.shape[0]
            n_fg = np.count_nonzero(trimap_pxs > 224)
            n_bg = np.count_nonzero(trimap_pxs < 32)

            # Only extend non-equivocal regions with > 10% labels
            if n_bg == 0 and n_fg * 10 >= n_pxs:
                modified_trimap[rows, cols] = 255
            elif n_fg == 0 and n_bg * 10 >= n_pxs:
                modified_trimap[rows, cols] = 0

        # Write modified trimap for inspection
        modified_trimap_fn = os.path.join(datadir, 'trimap',
            input_base + '_modified.png')
        print('Writing: {}'.format(modified_trimap_fn))
        imageio.imwrite(modified_trimap_fn, modified_trimap)

        modified_trimap_vis_fn = os.path.join(datadir, 'trimap',
            input_base + '_modified_vis.jpg')
        imageio.imwrite(modified_trimap_vis_fn,
            mlmask.visualise_trimap(input_im, modified_trimap))

        trimap_vis_fn = os.path.join(datadir, 'trimap',
            input_base + '_vis.jpg')
        imageio.imwrite(trimap_vis_fn,
            mlmask.visualise_trimap(input_im, trimap_im))

        # Extract indices of +ve and -ve training samples
        modified_trimap = modified_trimap.reshape((-1,))
        foreground_mask = modified_trimap > 224
        background_mask = modified_trimap < 32

        feature_vector = mlmask.image_to_features(input_im,
            transform=transform, nlevels=6)

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
