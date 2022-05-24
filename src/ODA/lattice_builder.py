from analysis_tools import *

class Lattice():
    def __init__(self, p_vector1, p_vector2):
        """A Bravais Lattice defined by two primitive vectors.


        Parameters
        ----------
        p_vector1, p_vector2 : np.ndarray, shape=(1, 2)
            The primitive vectors defining the spacing and orientation of
            axes in the lattice.
        """
        self.p_vector1 = p_vector1
        self.p_vector2 = p_vector2

    def render(self, x_range, y_range, test=False):
        """Produce the elements of the lattice within bounds.


        Parameters
        ----------
        x_range : tuple, (xmin, xmax)
            The range of x values included in the lattice as a pair.
        y_range : tuple, (ymin, ymax)
            The range of y values included in the lattice as a pair.
        
        Returns
        -------
        pts : np.ndarray, shape=(N, 2)
            The N points from the lattice within bounds.
        """
        xmin, xmax = sorted(list(x_range))
        ymin, ymax = sorted(list(y_range))
        # find bounds of the unit array for making the lattice
        bounds = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
        self.basis = np.array([self.p_vector1, self.p_vector2])
        bounds_t = np.dot(bounds, np.linalg.inv(self.basis))
        xmin_t, ymin_t = bounds_t.min(0)
        xmax_t, ymax_t = bounds_t.max(0)
        # get unit lattice within these bounds
        xvals_t = np.arange(xmin_t, xmax_t + 1)
        yvals_t = np.arange(ymin_t, ymax_t + 1)
        xgrid_t, ygrid_t = np.meshgrid(xvals_t, yvals_t)
        # transform to lattice coordinates using dot product
        pts_t = np.array([xgrid_t, ygrid_t]).transpose(1, 2, 0)
        pts = np.dot(pts_t, self.basis)
        xs, ys = pts[..., 0], pts[..., 1]
        # remove points outside the specified ranges
        include = (xs > xmin) * (xs <=xmax) * (ys > ymin) * (ys <= ymax)
        if test:
            # sanity check
            plt.scatter(xs[include], ys[include])
            plt.gca().set_aspect('equal')
            plt.show()
        pts = pts[include]
        return(pts)


class LatticeFit():
    def __init__(self, image, xvals, yvals):
        """Find the lattice that maximizes the mean when sampling a 2D image.


        Parameters
        ---------
        image : np.ndarray, shape=(height, width)
            Image used to fit the lattice. 
        
        Attributes
        ----------
        p_vector1, p_vector2 : np.ndarray, shape=(1, 2)
            The primitive vectors defining the spacing and orientation of
            axes in the lattice. These are the object of the fitness function.
        """
        self.image = image
        self.xvals = xvals
        self.yvals = yvals
        self.x_range = (self.xvals.min(), self.xvals.max())
        self.y_range = (self.yvals.min(), self.yvals.max())
        # to speed up processing, generate a variables for griddata interpolation
        self.xgrid, self.ygrid = np.meshgrid(self.xvals, self.yvals)
        self.xs, self.ys = self.xgrid.flatten(), self.xgrid.flatten()
        self.zs = self.image.flatten()
        self.dc_val = self.image.max()
        # setup non-linear optimization process with bounds on the p_vectors.
        # lower: the p_vectors can't be smaller than twice the smallest distance
        pos_xvals = self.xvals > 0
        pos_yvals = self.yvals > 0
        xval_lower = 2 * self.xvals[pos_xvals].min()
        yval_lower = 2 * self.yvals[pos_yvals].min()
        # upper: the p_vectors can't be larger than half the long distance
        xval_upper = self.xvals.max()
        yval_upper = self.yvals.max()
        # break into separate variables in order to define bounds
        self.p_vector1 = np.array([0, yval_upper/2], dtype=float)
        self.p_vector2 = np.array([xval_upper/2, 0], dtype=float)
        p1_x, p1_y = self.p_vector1
        p2_x, p2_y = self.p_vector2
        # set a boundary constraints on the longest length of the p_vectors
        constraints = [
            optimize.NonlinearConstraint(
                self.min_vector_norm,
                lb=max(xval_lower, yval_lower), ub=None),
            optimize.NonlinearConstraint(
                self.max_vector_norm,
                lb=None, ub=min(xval_upper, yval_upper))]
        bounds = [(-xval_upper, xval_upper), (0, yval_upper),
                  (-xval_upper, xval_upper), (0, yval_upper)]
        # self.results = optimize.minimize(
        #     self.error, x0=np.array([p1_x, p1_y, p2_x, p2_y]),
        #     constraints=constraints, bounds=bounds)
        breakpoint()
        self.results = optimize.basinhopping(
            self.error, x0=np.array([p1_x, p1_y, p2_x, p2_y]))
            # constraints=constraints, bounds=bounds)
        plt.pcolormesh(self.xvals, self.yvals, self.image)
        self.lattice.render(self.x_range, self.y_range, test=True)
        plt.show()


    def max_vector_norm(self, p_vectors):
        # get vector lengths for the constraint
        p_vector1, p_vector2 = p_vectors[:2], p_vectors[2:]
        p_vector1_norm = np.linalg.norm(p_vector1)
        p_vector2_norm = np.linalg.norm(p_vector2)
        max_norm = max(p_vector1_norm, p_vector2_norm)
        print(max_norm)
        return max_norm

    def min_vector_norm(self, p_vectors):
        # get vector lengths for the constraint
        self.p_vector1_norm = np.linalg.norm(self.p_vector1)
        self.p_vector2_norm = np.linalg.norm(self.p_vector2)
        return min(self.p_vector1_norm, self.p_vector2_norm)

    def error(self, p_vectors):
        """The mean sample of values near all lattice elements.


        Parameters
        ----------
        p_vector1, p_vector2 : array-like, len=2
            The primitive 2D vectors defining the spacing and orientation of
            axes in the lattice.
        """
        # generate a lattice with the provided p_vectors
        # p1_x, p1_y, p2_x, p2_y = p_vectors
        # self.p_vector1, self.p_vector2 = np.array([p1_x, p1_y]), np.array([p2_x, p2_y])
        self.p_vector1, self.p_vector2 = p_vectors[:2], p_vectors[2:]
        self.lattice = Lattice(self.p_vector1, self.p_vector2)
        # plt.pcolormesh(self.xvals, self.yvals, self.image)
        self.pts = self.lattice.render(self.x_range, self.y_range, test=False)
        xs, ys = self.pts.T
        zs = self
        # use linear interpolation to approximate the image values
        # for each lattice element
        self.vals = interpolate.griddata(
            np.array([self.xs, self.ys]).T, self.zs, xi=self.pts, method='nearest')
        if len(self.vals) > 0:
            # remove the dc component and return mean value
            self.vals[np.argmax(self.vals)] = 0
            return -self.vals.mean()  # negative so that 'minimize' actually maximizes
        else:
            return 0
        
# # 1. test class by making lattice with two principal vectors
# p1 = np.array([1/2, np.sqrt(3)/2])
# p2 = np.array([1, 0])
# lattice = Lattice(p_vector1=p1, p_vector2=p2)
# # get the points of the lattice
# x_range = (-10, 10)
# y_range = (-10, 10)
# # test the lattice
# pts = lattice.render(x_range, y_range, test=True)

# 2. todo: lattice fit
# use sample image for this
eye = Eye("../../data/002.jpg", mask_fn="../../data/002/mask.jpg")
eye.oda(high_pass=True, plot=True)

