import numpy as np

from glyphdata.algorithms.template import SkelAlgorithm
from skimage.morphology import skeletonize
from skimage.util import invert
from skimage.filters import threshold_otsu


class SkelZhang(SkelAlgorithm):
    def __init__(self, invert: False):
        self.invert = invert

    def compute(self, img: np.ndarray) -> np.ndarray:
        if self.invert:
            img = invert(img)
        thresh = threshold_otsu(img)
        binary = img > thresh

        return skeletonize(binary, method="zhang")