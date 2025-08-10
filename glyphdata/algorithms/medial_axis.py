import numpy as np

from glyphdata.algorithms.template import SkelAlgorithm
from skimage.morphology import medial_axis
from skimage.util import invert
from skimage.filters import threshold_otsu


class SkelMedialAxis(SkelAlgorithm):
    def __init__(self, invert: False):
        self.invert = invert

    def compute(self, img: np.ndarray) -> np.ndarray:
        if self.invert:
            img = invert(img)
        thresh = threshold_otsu(img)
        binary = img > thresh

        return medial_axis(binary)