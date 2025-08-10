import numpy as np
from scipy.interpolate import splprep, splev
from scipy.special import comb as n_over_k


class BezierCurve(object):
    # Define Bezier curves for curve fitting
    def __init__(self, order, num_sample_points=50, fix_start_end=True):
        self.order = order
        self.control_points = []
        self.bezier_coeff = self.get_bezier_coefficient()
        self.num_sample_points = num_sample_points
        self.fix_start_end = fix_start_end
        self.c_matrix = self.get_bernstein_matrix()

    def get_bezier_coefficient(self):
        Mtk = lambda n, t, k: t ** k * (1 - t) ** (n - k) * n_over_k(n, k)
        BezierCoeff = lambda ts: [[Mtk(self.order, t, k) for k in range(self.order + 1)] for t in ts]

        return BezierCoeff

    def interpolate_lane(self, x, y, n=50):
        # Spline interpolation of a lane. Used on the predictions
        assert len(x) == len(y)

        tck, _ = splprep([x, y], s=0, t=n, k=min(3, len(x) - 1))

        u = np.linspace(0., 1., n)
        return np.array(splev(u, tck)).T

    def get_control_points(self, x, y, interpolate=False):
        if interpolate:
            points = self.interpolate_lane(x, y)
            x = np.array([x for x, _ in points])
            y = np.array([y for _, y in points])

        middle_points = self.get_middle_control_points(x, y)
        for idx in range(0, len(middle_points) - 1, 2):
            self.control_points.append([middle_points[idx], middle_points[idx + 1]])

    def get_bernstein_matrix(self):
        tokens = np.linspace(0, 1, self.num_sample_points)
        c_matrix = self.bezier_coeff(tokens)
        return np.array(c_matrix)

    def save_control_points(self):
        return self.control_points

    def assign_control_points(self, control_points):
        self.control_points = control_points

    def quick_sample_point(self, image_size=None):
        control_points_matrix = np.array(self.control_points)
        sample_points = self.c_matrix.dot(control_points_matrix)
        if image_size is not None:
            sample_points[:, 0] = sample_points[:, 0] * image_size[-1]
            sample_points[:, -1] = sample_points[:, -1] * image_size[0]
        return sample_points

    def get_sample_point(self, n=50, image_size=None):
        '''
            :param n: the number of sampled points
            :return: a list of sampled points
        '''
        t = np.linspace(0, 1, n)
        coeff_matrix = np.array(self.bezier_coeff(t))
        control_points_matrix = np.array(self.control_points)
        sample_points = coeff_matrix.dot(control_points_matrix)
        if image_size is not None:
            sample_points[:, 0] = sample_points[:, 0] * image_size[-1]
            sample_points[:, -1] = sample_points[:, -1] * image_size[0]

        return sample_points

    def get_middle_control_points(self, x, y):
        dy = y[1:] - y[:-1]
        dx = x[1:] - x[:-1]
        dt = (dx ** 2 + dy ** 2) ** 0.5
        t = dt / dt.sum()
        t = np.hstack(([0], t))
        t = t.cumsum()
        data = np.column_stack((x, y))
        Pseudoinverse = np.linalg.pinv(self.bezier_coeff(t))  # (9,4) -> (4,9)
        control_points = Pseudoinverse.dot(data)  # (4,9)*(9,2) -> (4,2)
        if self.fix_start_end:
            control_points[0] = [x[0], y[0]]
            control_points[len(control_points) - 1] = [x[len(x) - 1], y[len(y) - 1]]

        medi_ctp = control_points[:, :].flatten().tolist()
        return medi_ctp


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    x = [1, 3, 5, 7, 9, 12, 14, 15, 18, 19, 23, 24, 25, 26, 27, 30]
    y = [5, 7, 8, 9, 9, 9, 10, 11, 11, 11, 12, 12, 13, 13, 12, 11]

    x = np.array(x) - 10
    y = np.array(y) - 10

    fig, ax = plt.subplots()
    ax.scatter(x, y, c='r', marker='x', s=50)

    bc = BezierCurve(order=7, num_sample_points=16, fix_start_end=True)
    bc.get_control_points(x, y)
    print(bc.control_points)
    ax.plot([b[0] for b in bc.control_points], [b[1] for b in bc.control_points], c='g')
    ax.scatter([b[0] for b in bc.get_sample_point(50)], [b[1] for b in bc.get_sample_point(50)], c='g')
    ax.set_ylim([4, 18])
    plt.show()

    fig, ax = plt.subplots()
    ax.scatter(x, y, c='r', marker='x', s=50)

    for o, c in zip([1,3,5,7], ['green', 'blue', 'orange', 'purple']):
        bc = BezierCurve(order=o, num_sample_points=16, fix_start_end=True)
        bc.get_control_points(x, y)
        #ax.plot([b[0] for b in bc.control_points], [b[1] for b in bc.control_points], c=c, linestyle='dotted', alpha=0.5)
        ax.scatter([b[0] for b in bc.get_sample_point(50)], [b[1] for b in bc.get_sample_point(50)], c=c, alpha=0.5)

    ax.set_ylim([y.min()-2, y.max()+2])
    plt.show()
