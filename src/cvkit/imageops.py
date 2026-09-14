"""Small image measures shared by the mining and export-repair commands."""
import cv2
import numpy as np


def flatness(path, block=8, std_thresh=3.0, size=(256, 144)):
    """Fraction of blocks that are essentially uniform colour.

    Video branding -- title cards, captions, end screens -- is mostly flat
    colour; real footage almost never is. Cheap, and far more reliable here
    than asking a detector whether a caption contains an object.

    A missing or unreadable file scores 1.0, so it fails any "is this real
    footage" test rather than silently passing.
    """
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        return 1.0
    im = cv2.resize(im, size).astype(np.float32)
    h, w = im.shape
    bh, bw = h // block * block, w // block * block
    blocks = (im[:bh, :bw]
              .reshape(bh // block, block, bw // block, block)
              .transpose(0, 2, 1, 3))
    return float((blocks.reshape(-1, block * block).std(1) < std_thresh).mean())


def descriptor(path, size=(32, 32)):
    """A tiny normalised thumbnail, flattened -- enough to spot near-duplicates.

    Mean absolute difference between two of these is a usable perceptual
    distance for frames out of the same footage; it is not a general-purpose
    image hash and will not survive a crop or a big colour shift.
    """
    im = cv2.imread(str(path))
    if im is None:
        return None
    return cv2.resize(im, size).astype(np.float32).ravel() / 255.0
