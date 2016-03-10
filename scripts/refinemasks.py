#!/usr/bin/env python3
"""
Refine perpixel masks using superpixels.

Usage:
    refinemasks.py <datadir>

"""
from __future__ import division

import glob
import os

import docopt
import imageio
import numpy as np
import skimage.morphology as skmorph
import skimage.measure as skmeas
import skimage.segmentation as skseg

import mlmask

def main():
    opts = docopt.docopt(__doc__)
    datadir = opts['<datadir>']
    vote_threshold = 0.75

    outputdir = os.path.join(datadir, 'masks')
    if not os.path.isdir(outputdir):
        os.makedirs(outputdir)

    for input_fn in glob.glob(os.path.join(datadir, 'input', '*.JPG')):
        input_base = os.path.splitext(os.path.basename(input_fn))[0]
        output_base = os.path.join(outputdir, input_base)
        if os.path.exists(output_base + '.png'):
            print('Skipping {}'.format(output_base))
            continue

        # Input files
        mask_fn = os.path.join(datadir, 'perpixel', input_base + '.png')
        superpixel_fn = os.path.join(datadir, 'superpixel', input_base + '.npz')

        if not os.path.isfile(mask_fn):
            continue
        if not os.path.isfile(superpixel_fn):
            continue

        print('Input: ' + input_fn)
        print('Mask: ' + mask_fn)
        print('Superpixel: ' + superpixel_fn)
        input_im = imageio.imread(input_fn)

        labels = np.load(superpixel_fn)['labels']
        pp_mask = np.where(imageio.imread(mask_fn) > 128, 1, 0)

        # Assign mask based on region voting
        vote_mask = np.copy(pp_mask)
        voting = np.zeros(pp_mask.shape)
        for props in skmeas.regionprops(labels, pp_mask):
            vote = props.mean_intensity
            coords = props.coords

            voting[coords[:, 0], coords[:, 1]] = vote
            if vote > vote_threshold:
                vote_mask[coords[:, 0], coords[:, 1]] = 1
            else:
                vote_mask[coords[:, 0], coords[:, 1]] = 0

            #vote_mask[labels == props.label] = 1 if vote > vote_threshold else 0
            #voting[labels == props.label] = vote

        # Choose largest connected component as mask
        component_labels = skmeas.label(vote_mask)
        largest_label, largest_area = None, 0
        for props in skmeas.regionprops(component_labels, vote_mask):
            if props.area > largest_area and props.mean_intensity > 0.99:
                largest_label = props.label
                largest_area = props.area
        print('Largest component has area of {} pixels'.format(largest_area))

        assert largest_label is not None
        mask = np.where(component_labels == largest_label, 1, 0)

        print('Saving mask')
        imageio.imwrite(output_base + '.png',
            np.where(mask == 0, 0, 255).astype(np.uint8))

        imageio.imwrite(output_base + '-vote-mask.png',
            np.where(vote_mask == 0, 0, 255).astype(np.uint8))

        imageio.imwrite(output_base + '-voting.png',
            (voting * 255).astype(np.uint8))

        imageio.imwrite(output_base + '-visualisation.jpg',
            mlmask.visualise_mask(input_im, mask))


if __name__ == '__main__':
    main()
