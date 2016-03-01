#!/usr/bin/env python3
"""
Generate superpixel segmentation of input images.

Usage:
    gensuperpixel.py <datadir>

"""
import glob
import os

import docopt
import imageio
import numpy as np
import scipy.ndimage as ndi
import skimage.color as skimcolor
import skimage.filters as skimfilt
import skimage.morphology as skimmorph
import skimage.segmentation as skimseg

def marker_segment(image):
    luminance = skimcolor.rgb2gray(image.astype(np.float32) / 255)
    denoised = skimfilt.rank.median(luminance, skimmorph.disk(2))
    markers = skimfilt.rank.gradient(denoised, skimmorph.disk(5)) < 10
    markers = ndi.label(markers)[0]
    gradient = skimfilt.rank.gradient(denoised, skimmorph.disk(2))
    return skimseg.random_walker(image, markers, multichannel=True)

def main():
    opts = docopt.docopt(__doc__)
    datadir = opts['<datadir>']
    outputdir = os.path.join(datadir, 'superpixel')

    if not os.path.isdir(outputdir):
        os.makedirs(outputdir)

    for input_fn in glob.glob(os.path.join(datadir, 'input', '*.JPG')):
        print('Input: ' + input_fn)
        input_im = imageio.imread(input_fn)
        input_base = os.path.join(outputdir,
            os.path.splitext(os.path.basename(input_fn))[0])

        # Compute over segmentation
        print('Segmenting...')
        labels = skimseg.slic(input_im, sigma=1,
            multichannel=True, convert2lab=True)

        print('Saving output...')

        # Write visualisation
        imageio.imwrite(input_base + '-visualisation.jpg',
            skimseg.mark_boundaries(input_im, labels))

        # Write output
        np.savez_compressed(input_base + '.npz', labels=labels)

if __name__ == '__main__':
    main()
