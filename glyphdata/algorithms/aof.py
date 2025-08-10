import numpy as np

from glyphdata.algorithms.template import SkelAlgorithm
from skimage.util import invert
from skimage.filters import threshold_otsu
from scipy.ndimage import distance_transform_edt


class SkelAOF(SkelAlgorithm):
    def __init__(self, invert: False, number_of_samples=60, epsilon=1, flux_threshold=18):
        self.invert = invert
        self.number_of_samples = number_of_samples
        self.epsilon = epsilon
        self.flux_threshold = flux_threshold

    def compute(self, img: np.ndarray) -> np.ndarray:
        if self.invert:
            img = invert(img)
        thresh = threshold_otsu(img)
        binary = (img > thresh).astype(int)

        dist_image, IDX = distance_transform_edt(binary, return_indices=True)
        sphere_points = self.sample_sphere_2D()
        flux_image = self.compute_aof(dist_image, IDX, sphere_points)

        skeleton = flux_image.copy()
        skeleton[flux_image < self.flux_threshold] = 0
        skeleton[flux_image > self.flux_threshold] = 1
        return skeleton

    def sample_sphere_2D(self):
        sphere_points = np.zeros((self.number_of_samples, 2))
        alpha = (2 * np.pi) / (self.number_of_samples)
        for i in range(self.number_of_samples):
            sphere_points[i][0] = np.cos(alpha * (i - 1))
            sphere_points[i][1] = np.sin(alpha * (i - 1))
        return sphere_points

    def sub2ind(self, array_shape, rows, cols):
        ind = rows * array_shape[1] + cols
        ind[ind < 0] = -1
        ind[ind >= array_shape[0] * array_shape[1]] = -1
        return ind

    def ind2sub(self, array_shape, ind):
        ind[ind < 0] = -1
        ind[ind >= array_shape[0] * array_shape[1]] = -1
        rows = (ind.astype('int') / array_shape[1])
        cols = ind % array_shape[1]
        return (rows, cols)

    def compute_aof(self, distImage, IDX, sphere_points):
        m = distImage.shape[0]
        n = distImage.shape[1]
        normals = np.zeros(sphere_points.shape)
        fluxImage = np.zeros((m, n))
        for t in range(0, self.number_of_samples):
            normals[t] = sphere_points[t]
        sphere_points = sphere_points * self.epsilon

        XInds = IDX[0]
        YInds = IDX[1]

        for i in range(0, m):
            print(i)
            for j in range(0, n):
                flux_value = 0
                if (distImage[i][j] > -1.5):
                    if (i > self.epsilon and j > self.epsilon and i < m - self.epsilon and j < n - self.epsilon):
                        #                   sum over dot product of normal and the gradient vector field (q-dot)
                        for ind in range(0, self.number_of_samples):

                            # a point on the sphere
                            px = i + sphere_points[ind][0] + 0.5
                            py = j + sphere_points[ind][1] + 0.5

                            # the indices of the grid cell that sphere points fall into
                            cI = np.floor(i + sphere_points[ind][0] + 0.5)
                            cJ = np.floor(j + sphere_points[ind][1] + 0.5)

                            # closest point on the boundary to that sphere point
                            bx = XInds[cI][cJ]
                            by = YInds[cI][cJ]
                            # the vector connect them
                            qq = [bx - px, by - py]

                            d = np.linalg.norm(qq)
                            if d != 0:
                                qq = qq / d
                            else:
                                qq = [0, 0]
                            flux_value = flux_value + np.dot(qq, normals[ind])
                fluxImage[i][j] = flux_value
        return fluxImage
