#!/usr/bin/env python3
"""
Generate superpixel segmentation of input images.

Usage:
    gensuperpixel.py <datadir>

"""
from __future__ import division

import glob
import os

import docopt
import imageio
import numpy as np
import scipy.ndimage as ndi
import skimage.color as skimcolor
import skimage.filters as skimfilt
import skimage.feature as skimfeat
import skimage.morphology as skimmorph
import skimage.measure as skimmeas
import skimage.segmentation as skimseg

def ws_segment(image, sigma=0):
    if sigma > 0:
        image = skimfilt.gaussian_filter(image, sigma=sigma)

    L = (255 * image.astype(np.float32) / image.max()).astype(np.uint8)
    denoised = skimfilt.rank.median(L, skimmorph.disk(5))

    edges = np.clip(255 * skimfilt.sobel(denoised), 0, 255).astype(np.uint8)
    edges_locmin = skimfilt.rank.minimum(edges, skimmorph.disk(5))

    # Start at local minima
    markers = edges == edges_locmin

    # Remove very small markers
    markers = skimfilt.rank.median(markers, skimmorph.disk(5))

    # Label markers
    markers = np.where(markers != 0, skimmeas.label(markers) + 1, 0)

    return skimmorph.watershed(edges, markers)

def main():
    opts = docopt.docopt(__doc__)
    datadir = opts['<datadir>']
    outputdir = os.path.join(datadir, 'superpixel')

    if not os.path.isdir(outputdir):
        os.makedirs(outputdir)

    for input_fn in glob.glob(os.path.join(datadir, 'input', '*.JPG')):
        input_base = os.path.join(outputdir,
            os.path.splitext(os.path.basename(input_fn))[0])
        if os.path.isfile(input_base + '.npz'):
            print('Skipping since {} exists'.format(input_base))
            continue

        print('Input: ' + input_fn)
        input_im = imageio.imread(input_fn)

        print('Converting to LAB colorspace')
        lab_im = skimcolor.rgb2lab(input_im)

        labels = np.zeros(input_im.shape[:2])

        print('Segmenting (watershed)...')
        labels = skimseg.join_segmentations(labels, ws_segment(lab_im[..., 0]))

        print('Segmenting (slic)...')
        # Set number of segments so each segment is roughly seg_size*seg_size in area
        seg_size = 128
        n_segments = 1 + int(input_im.shape[0] * input_im.shape[1] /
            (seg_size*seg_size))
        labels = skimseg.join_segmentations(labels,
            skimseg.slic(lab_im, n_segments=n_segments, sigma=1,
                compactness=0.1, multichannel=True, convert2lab=False,
                slic_zero=True)
        )

        print('Enforcing connectivity')
        # Enforce connectivity. This is important otherwise superpixels may be
        # spread over image.
        labels = skimmeas.label(labels)

        print('Saving output...')

        # Write visualisation
        imageio.imwrite(input_base + '-visualisation.jpg',
            skimseg.mark_boundaries(input_im, labels))

        # Write output
        np.savez_compressed(input_base + '.npz', labels=labels)

if __name__ == '__main__':
    main()
