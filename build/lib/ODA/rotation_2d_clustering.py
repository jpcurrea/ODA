# make a class to use 2d clustering on a 3d dataset using 2d projection and kmeans
class RotateClusterer():
    def __init__(self, pts_3d, centers_3d):
        """Allows for clustering 3D data based on all possible 2D rotation projections.


        Consider the set of all rotations of a 3D coordinate cloud. If this cloud
        contains cylindrical clusters all roughly parallel, then there should be only
        2 rotations that allow collapsing along the longitudinal axis of the cylinders. 
        Thus, the points can be more easily clustered in those 2D projections once their
        known.

        pts_3d : np.ndarray, shape=(N, 3)
            The array of 3D coorinates used in the clustering algorithm.
        centers_3d : np.ndarray, shape=(k, 3)
            The array 3D centroids used for clustering.
        """
        self.pts = np.copy(pts_3d)
        self.centers = np.copy(centers_3d)
        com = pts.mean(0)                # get the center of mass
        self.pts -= com                  # center the points for rotations later on
        self.centers -= com              # the points for rotations later on
        # rotate pts and centers so that pts are projected onto the approximate 
        # plane formed by the centers
        uu, dd, vv = np.linalg.svd(self.centers)
        # rotating self.centers by vv results in 'flattening' centers. These serve as 
        # a good starting point for optimization
        self.centers = np.dot(self.centers, vv)
        self.pts = np.dot(self.pts, vv)
        # store the 3D direction vector to be optimized
        self.orientation_vector = np.array([[1, 0, 0],
                                            [0, 1, 0],
                                            [0, 0, 1]])
        # store empty list of labels associated with each cluster
        self.lbls = np.zeros(len(self.pts.shape), dtype=int)
        self.error(self.orientation_vector)
        # call minimization function
        self.fmin = optimize.fmin(self.error, self.orientation_vector)
        
    def error(self, ori_vector):
        """We want to minimize the KMeans inertia of the 2D projection."""
        # rotate pts and centers using the orientation vector
        pts_rotated = np.dot(self.pts, ori_vector)
        centers_rotated = np.dot(self.centers, ori_vector)
        # project onto the 2D plane by using just the last 2 dimensions
        pts_2d = pts_rotated[:, 1:]
        centers_2d = centers_rotated[:, 1:]
        # cluster the projected pts and centers using KMeans
        new_centers, lbls, inertia = cluster.k_means(
            pts_2d, n_clusters=len(centers_2d), init=centers_2d, n_init=1)
        breakpoint()
        
