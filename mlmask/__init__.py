import docopt
import dtcwt
import imageio
import numpy as np

import skimage.segmentation as skseg
import skimage.transform as sktrans

def _resize(image, shape):
    """Resize an input array."""
    sc = np.abs(image).max()
    return sktrans.resize(image / sc, shape, mode='reflect') * sc

def visualise_mask(image, mask):
    output = skseg.mark_boundaries(image, mask)
    rng = output.max()
    alpha = 0.75
    mask_color = [0.5, 0, 0]
    for c_idx in range(output.shape[2]):
        output[..., c_idx] = np.where(mask != 0,
            output[..., c_idx],
            alpha*mask_color[c_idx] + (1.0-alpha)*output[..., c_idx])
    return output

def image_to_features(image, nlevels=6, transform=None):
    assert len(image.shape) == 3
    assert image.shape[2] == 3

    if transform is None:
        transform = dtcwt.Transform2d()

    # Compute DTCWT for each channel
    feature_vector = []
    for c_idx in range(image.shape[2]):
        input_chan = image[..., c_idx].astype(np.float32)
        input_chan /= 255.0
        input_chan = np.asarray(input_chan)
        input_dtcwt = transform.forward(input_chan, nlevels=nlevels)

        # We re-scale each channel of the DTCWT to the input image size. We
        # take the maximum absolute value along each direction.
        feature_vector.append(_resize(
            input_dtcwt.lowpass, input_chan.shape))

        # For each highpass...
        for hp in input_dtcwt.highpasses:
            feature_vector.append(_resize(
                np.max(np.abs(hp), axis=-1), input_chan.shape))

    # Stack feature vector up
    feature_vector = np.dstack(feature_vector)

    return feature_vector
