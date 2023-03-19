from analysis_tools import *
import numpy as np
import scipy

class Lattice():
    def __init__(self, p_vector1=np.array([1/2, np.sqrt(3)/2]), p_vector2=np.array([1, 0])):
        """A Bravais Lattice defined by two primitive vectors.


        Parameters
        ----------
        p_vector1, p_vector2 : np.ndarray, shape=(1, 2)
            The primitive vectors defining the spacing and orientation of
            axes in the lattice.
        """
        self.p_vector1 = p_vector1
        self.p_vector2 = p_vector2

    def render(self, x_range, y_range, scale=1, test=False):
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
        self.basis = np.array([scale * self.p_vector1, scale * self.p_vector2])
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
    
    def simulate_eye(self, scale=1, noise_std=0, contrast=1.0, xres=100, yres=100):
        """Generate an artificial hexagonal lattice at depending on scale and position noise.
        
        Parameters
        ----------
        scale : float, default=1
            The scale of the spacing between neighboring points.
        noise : float, default=0
            The standard deviation of gaussian noise applied to the regular hexagonal lattice.
        """
        x_range = (0, xres)
        y_range = (0, yres)
        self.pts = self.render(x_range, y_range, scale=scale)
        self.pts = self.pts.astype('float')
        # add the optional gaussian noise to the pts
        if noise_std > 0:
            # noise = np.random.normal(0, scale=noise_std, size=self.pts.shape)
            noise = np.random.uniform(-noise_std, noise_std, size=self.pts.shape)
            self.pts += noise
        # make a high resolution image with all pts coordinates set to 1 and the others set to 0
        img = np.zeros((yres, xres), dtype='float')
        xs, ys = self.pts.T.astype(int)
        # make a voronoi diagram with these points
        if len(self.pts) < 4:
            xmax, ymax = img.shape
            for x in [0, xmax]: 
                for y in [0, ymax]: 
                    self.pts = np.append(self.pts, [[x, y]], axis=0)
        try:
            voronoi = spatial.Voronoi(self.pts)
        except:
            breakpoint()
        # make the nearest pixels in img to the voronoi verticies white and then blur the image
        lines = []
        dists = []
        height, width = img.shape
        for region in voronoi.regions:
            verts = np.round(voronoi.vertices[region]).astype(int)
            # plot a line between all of the vertices
            for start, stop in zip(verts[:-1], verts[1:]):
                if start.min() > 0 and stop.min() > 0 and start.max() < width and stop.max() < width:
                    line = skimage.draw.line(start[0], start[1], stop[0], stop[1])
                    lines += [np.array([line[0], line[1]]).T]
                    dists += [np.repeat(np.linalg.norm(stop - start), len(line[0]))]
        # todo: fix the long lines! the distance method isn't working at all scales
        breakpoint()
        dists = np.concatenate(dists)
        lines = np.concatenate(lines)
        thresh = 3 * np.percentile(dists, 50)
        include = lines > 0
        include = np.all(include, axis=1)
        include *= lines[:, 0] < height
        include *= lines[:, 1] < width
        include *= dists < thresh            
        lines = lines[include]
        img[lines[:, 0], lines[:, 1]] = 255
        # lines = np.unique(lines)
        # img[ys % yres, xs % xres] = 1
        # # estimate the diameter of each lens based on the distance of each point to their 3 nearest points
        # if len(self.pts) > 1:
        #     tree = spatial.KDTree(self.pts)
        #     dists, inds = tree.query(self.pts, k=4)
        #     diams = dists[:, 1:].min(1)
        #     # diams = dists[:, 1:].mean(1)
        #     # breakpoint()
        #     # var = dists[:, 1:].std(1)
        #     # order = np.argsort(var)[::-1]
        #     # # get all neighbors within 1.5 times the approx diameter
        #     # dists, inds = tree.query(self.pts, k=7, distance_upper_bound=1.5 * diams.mean())
        #     # centers = []
        #     # for ind in inds:  centers += [self.pts[ind[ind < len(self.pts)]].mean(0)]
        #     # centers = np.array(centers)
        #     # neighbors = self.pts[inds[:, 1:]]
        #     # neighbors[inds] = np.nan
        #     # new_pts = np.nanmean(neighbors, axis=1)
        #     # use the smallest diameter
        #     std = diams.min()/5
        #     # todo: place a 2D gaussian at each x and y with the specific diameter
        #     blur = np.zeros(img.shape, dtype=float)
        #     height, width = blur.shape
        #     for x, y, diam in zip(xs, ys, np.round(diams, 0)):
        #         diam *= 1.5
        #         diam = round(diam)
        #         if diam % 2 == 1:
        #             diam += 1
        #         window = signal.windows.gaussian(round(diam), std=diam/5)
        #         window -= .1
        #         window[window < 0] = 0
        #         window = window[:, np.newaxis] * window[np.newaxis]
        #         xmin, xmax = int(x - diam/2), int(x + diam/2)
        #         ymin, ymax = int(y - diam/2), int(y + diam/2)
        #         if xmin < 0:
        #             window = window[:, -xmin:]
        #             xmin = 0
        #         if xmax > width:
        #             window = window[:, :-(xmax - width)]
        #             xmax = width
        #         if ymin < 0:
        #             window = window[-ymin:]
        #             ymin = 0
        #         if ymax > height:
        #             window = window[:-(ymax - width)]
        #             ymax = height
        #         try:
        #             blur[ymin:ymax, xmin:xmax] += window
        #         except:
        #             breakpoint()
        # else:
        #     std = xres/5
        #     # apply a gaussian blur
        #     blur = ndimage.gaussian_filter(img, sigma=std, mode='constant', cval=0)
        blur = ndimage.gaussian_filter(img, 2)
        blur /= blur.max()
        # convert to 8-bit image
        blur = (255 * contrast * blur).astype('uint8')
        return blur


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

def reciprocal(img):
    """Use the 2D FFT to generate the reciprocal image of the supplied array."""
    height, width = img.shape
    window = signal.windows.gaussian(height, height/6)
    window = window[:, np.newaxis] * window[np.newaxis]
    img = window * img
    fft = scipy.fft.fft2(img - img.mean())
    power = np.abs(fft)
    power = np.fft.fftshift(power)
    power = signal.correlate2d(power, power, mode='same')
    return power

scales = np.logspace(0, 2, 10)[3:][::-1]
fig, axes = plt.subplots(ncols=len(scales), nrows=2)
for scale, row in zip(scales, axes.T):
    lattice = Lattice()
    img = lattice.simulate_eye(scale=scale, xres=150, yres=150)
    row[0].imshow(img, cmap='Greys')
    # apply the ODA to this image and get the lens centers
    mask = np.ones(img.shape, dtype=bool)
    # mask = img > 0
    stack = Eye(arr=img, mask_arr=mask)
    stack.oda(bright_peak=False)
    if len(stack.ommatidial_inds) > 0:
        ys, xs = stack.ommatidial_inds.T
        row[0].scatter(xs, ys, color='r', marker='.', s=1)
    # plot the reciprocal image too
    row[1].imshow(reciprocal(img), cmap='Greys')
for ax in axes.flatten():
    ax.set_xticks([])
    ax.set_yticks([])
    sbn.despine(ax=ax, bottom=True, left=True)
plt.tight_layout()
plt.show()

# scale = scales[4]
# noise_stds = np.linspace(0, scale/2, 6)
# fig, axes = plt.subplots(ncols=6, nrows=2)
# for std, row in zip(noise_stds, axes.T):
#     lattice = Lattice()
#     img = lattice.simulate_eye(scale=scale, xres=150, yres=150, noise_std=std)
#     row[0].imshow(img, cmap='Greys')
#     # apply the ODA to this image and get the lens centers
#     mask = np.ones(img.shape, dtype=bool)
#     mask = img > 0
#     stack = Eye(arr=img, mask_arr=mask)
#     stack.oda(regular=True, bright_peak=False)
#     if len(stack.ommatidial_inds) > 0:
#         ys, xs = stack.ommatidial_inds.T
#         row[0].scatter(xs, ys, color='r', marker='.', s=1)
#     # plot the reciprocal image too
#     row[1].imshow(reciprocal(img), cmap='Greys')
# for ax in axes.flatten():
#     ax.set_xticks([])
#     ax.set_yticks([])
#     sbn.despine(ax=ax, bottom=True, left=True)
# plt.tight_layout()
# plt.show()