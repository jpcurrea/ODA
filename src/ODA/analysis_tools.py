"""
Analysis Tools
==============

Provides
  1. Loading and manipulating images and stacks of images
  2. Measuring number and distribution of ommatidia in compound eye images.
  3. Measuring ommatidia in 3D image stacks of compound eyes.

Classes
-------
ColorSelector
    A GUI for generating a boolean mask based on user input.
LSqEllipse
    From https://doi.org/10.5281/zenodo.3723294, fit ellipse to 2D points.
Layer
    An image loaded from file or 2D array.
Eye
    A Layer child specifically for processing images of compound eyes.
Stack
    A stack of images at different depths for making a focus stack.
EyeStack
    A special stack for handling a focus stack of fly eye images.
CTStack
    A special stack for handling a CT stack of compound eyes.

Functions
---------
rgb_2_gray(rgb) : np.ndarray
    Converts from image with red, green, and blue channels into grayscale.
"""
from functools import partial
import h5py
# when running from pip install:
from .interfaces import *
# when runnning locally:
# from interfaces import *
import math
import matplotlib
from matplotlib import colors, mlab
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import numpy as np
import os
import pandas as pd
import PIL
from PIL import Image
import pickle
import seaborn as sbn
import subprocess
import sys
from tempfile import mkdtemp
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5.QtWidgets import *

import skimage
from skimage.draw import ellipse as Ellipse
from skimage.feature import peak_local_max
from sklearn import cluster, mixture
import scipy
from scipy import interpolate, optimize, ndimage, signal, spatial, stats
from scipy.optimize import minimize
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.measurements import center_of_mass
blue, green, yellow, orange, red, purple = [
    (0.3, 0.45, 0.69), (0.33, 0.66, 0.41), (0.83, 0.74, 0.37),
    (0.78, 0.5, 0.16), (0.77, 0.31, 0.32), (0.44, 0.22, 0.78)]

if 'app' not in globals():
    app = QApplication([])

def print_progress(part, whole):
    import sys
    prop = float(part) / float(whole)
    sys.stdout.write('\r')
    sys.stdout.write('[%-20s] %d%%' % ('=' * int(20 * prop), 100 * prop))
    sys.stdout.flush()

for n in range(100):
    print_progress(n, 100)

def load_image(fn):
    """Import an image as a numpy array using the PIL."""
    return np.asarray(PIL.Image.open(fn))


def save_image(fn, arr):
    """Save an image using the PIL."""
    img = PIL.Image.fromarray(arr)
    if os.path.exists(fn):
        os.remove(fn)
    return img.save(fn)


def rgb_2_gray(rgb):
    """Convert image from RGB to grayscale."""
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def rgb_to_hsv(rgb):
    """Convert image from RGB to HSV."""
    if rgb.ndim == 3:
        ret = matplotlib.colors.rgb_to_hsv(rgb)
    else:
        l, w = rgb.shape
        ret = np.repeat(rgb, 3, axis=(-1))
    return ret


def rectangular_to_spherical(vals, center=[0, 0, 0]):
    """Convert 3D pts from rectangular to spherical coordinates.

    Parameters
    ----------
    vals : np.ndarray, shape (N, 3)
        3D points to be converted.
    center : array-like, shape (3)
        Center point to use for spherical conversion.
    
    Returns
    -------
    polar, shape (N, 3)
        The [inclination, azimuth, radius] per coordinate in vals.
    """
    pts = np.copy(vals)
    center = np.asarray(center)  # center the points
    pts -= center[np.newaxis]
    xs, ys, zs = pts.T
    radius = np.linalg.norm(pts, axis=(-1))  # get polar transformation
    inclination = np.arccos(pts[:, 2] / radius)  # theta [0,   pi]
    azimuth = np.arctan2(pts[:, 1], pts[:, 0])   # phi   [-pi, pi]
    polar = np.array([inclination, azimuth, radius]).T
    return polar


def spherical_to_rectangular(vals):
    """Convert 3D pts from rectangular to spherical coordinates.

    Parameters
    ----------
    vals : np.ndarray, shape (N, 3)
        3D points to be converted.
    
    Returns
    -------
    coords, shape (N, 3)
        The [x, y, z] per polar coordinate in vals.
    """
    pts = np.copy(vals)
    inclination, azimuth, radius = pts.T  # theta, phi, radii
    xs = radius * np.cos(azimuth) * np.sin(inclination)  # inverse polar tranformation
    ys = radius * np.sin(azimuth) * np.sin(inclination)
    zs = radius * np.cos(inclination)
    coords = np.array([xs, ys, zs]).T  # combine into one array
    return coords


def rotate(arr, theta, axis=0):
    """Generate a rotation matrix and rotate input array along a single axis."""
    if axis == 0:
        rot_matrix = np.array(
            [[1, 0, 0],
             [0, np.cos(theta), -np.sin(theta)],
             [0, np.sin(theta), np.cos(theta)]])
    else:
        if axis == 1:
            rot_matrix = np.array(
                [[np.cos(theta), 0, np.sin(theta)],
                 [0, 1, 0],
                 [-np.sin(theta), 0, np.cos(theta)]])
        else:
            if axis == 2:
                rot_matrix = np.array(
                    [[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])
    nx, ny, nz = np.dot(arr, rot_matrix).T
    nx = np.squeeze(nx)
    ny = np.squeeze(ny)
    nz = np.squeeze(nz)
    return np.array([nx, ny, nz])

def project_coords(pos_vectors, dir_vectors, center=np.zeros(3), radius=1e5,
                   convex=False):
    """Project 3D vectors onto an encompassing sphere ('world referenced' coordinates).

    
    Parameters
    ----------
    pos_vectors : array-like, shape=(N, 3)
        The coordinates specifying the origin of each vector.
    dir_vectors : array-like, shape=(N, 3)
        The coordinates specifying the directional components of each vector.
    center : array-like, default=(0, 0, 0)
        The coordinate of the center of the sphere. 
    radius : float, default=1e5
        The radius of the sphere.
    convex : bool, default=False
        Whether to assume the vectors are on a concave or convex surface,
        using the projection further from the center instead of the nearer one.
    """    
    proj_coords = []
    for p_vector, d_vector in zip(pos_vectors, dir_vectors):
        const = np.dot(p_vector, d_vector)
        diff = const ** 2 - np.linalg.norm(p_vector) ** 2 + radius ** 2
        if diff >= 0:
            dists = np.array([-const - np.sqrt(diff),
                              -const + np.sqrt(diff)])
            if convex:
                dist = dists[np.argmax(abs(dists))]
            else:
                dist = dists[np.argmin(abs(dists))]
            proj_pt = p_vector + d_vector * dist
        else:
            proj_pt = np.empty(3)
            proj_pt[:] = np.nan
        proj_coords += [proj_pt]
    proj_coords = np.array(proj_coords)
    return proj_coords

def rotate_compound(arr, yaw=0, pitch=0, roll=0):
    """Rotate the arr of coordinates along all three axes.

    
    Parameters
    ----------
    arr : array-like, shape=(N, 3)
        The array of 3D points to rotate.
    yaw : float
        The angle to rotate about the z-axis.
    pitch : float
        The angle to rotate about the x-axis.
    roll : float
        The angle to rotate about the y-axis.
    """
    yaw_arr = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0],
         [np.sin(yaw), np.cos(yaw), 0],
         [0, 0, 1]])
    pitch_arr = np.array(
        [[np.cos(pitch), 0, np.sin(pitch)],
         [0, 1, 0],
         [-np.sin(pitch), 0, np.cos(pitch)]])
    roll_arr = np.array(
        [[1, 0, 0],
         [0, np.cos(roll), -np.sin(roll)],
         [0, np.sin(roll), np.cos(roll)]])
    rotation_matrix = yaw_arr @ pitch_arr @ roll_arr
    return arr @ rotation_matrix


def fit_line(data):
    """Use singular value decomposition (SVD) to find the best fitting vector to the data.


    Parameters
    ----------
    data : np.ndarray
        Input data points
    """
    m = data.mean(0)
    max_val = np.round(2 * abs(data - m).max()).astype(int)
    uu, dd, vv = np.linalg.svd(data - m)
    return vv


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'."""
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def positive_fit(predictor, outcome):
    """Uses non-linear minimization to fit a polynomial to data.

    
    Parameters
    ----------
    predictor : np.ndarray
        Array of values used to model the outcome.
    outcome : np.ndarray
        Array of values being predicted by the predictor.

    Returns
    -------
    final_func : function
        The final fitted function, producing estimate of outcome.
    residuals_normalized : float
        The squared residuals divided by the number of samples.
    """
    # first, unwrap the predictor and outcome variables, recording the offsets 
    predictor_uw = np.unwrap(predictor)
    predictor_offset = predictor - predictor_uw
    outcome_uw = np.unwrap(outcome)
    outcome_offset = outcome - outcome_uw
    # sort the variables so that the predictor is in ascending order
    order = np.argsort(predictor_uw)
    # test: custom define the knots for the interpolation
    min_resids = np.inf
    final_func = None
    for num_knots in range(2, 6):
        knots = np.linspace(predictor_uw.min(), predictor_uw.max(), num_knots+2)[1:-1]
        spl = interpolate.splrep(predictor_uw[order], outcome_uw[:, 0][order],
                                 k=3, w=np.ones(len(order)), t=knots)
        # make a function using spl and splev
        def func(preds, spl=spl, predictor_offset=predictor_offset,
                       outcome_offset=outcome_offset[:, 0]):
            # unwrap the predictors
            predictor_uw = preds.flatten() - predictor_offset
            # interpolate the unwrapped data
            new_outcome = interpolate.splev(predictor_uw, spl)
            # adjust ('re-wrap') the outcome
            new_outcome += outcome_offset
            return new_outcome.reshape(preds.shape)
        # get the new predicted outcomes
        new_outcome = func(predictor)
        resids = np.sqrt((outcome - new_outcome)**2).mean()
        if resids < .95 * min_resids:
            final_func = func
            min_resids = resids
    # test: 
    # plt.scatter(predictor_uw, outcome[:, 0])
    # plt.scatter(predictor_uw, new_outcome)
    # plt.scatter(predictor_uw, outcome_uw[:, 0])
    # plt.scatter(predictor_uw, new_outcome)
    # plt.show()
    return (final_func, min_resids)

    # # if predictor and outcome have different shapes, make sure they have 
    # # the same dimension
    # if predictor.shape != outcome.shape:
    #     # check dimension
    #     if predictor.ndim != outcome.ndim:
    #         max_dims = max(predictor.ndim, outcome.ndim)
    #         if predictor.ndim < max_dims:
    #             predictor = np.expand_dims(predictor, axis=(-1))
    #         if outcome.ndim < max_dims:
    #             outcome = np.expand_dims(outcome, axis=(-1))
    # # this only works if there are more predictors than outcomes
    # assert outcome.shape[(-1)] == 1, 'There should only be one outcome variable.'
    # # iterate through orders of polynomial degree and find the best fit
    # def func(predictor, pars):
    #     return np.polyval(pars, predictor)
    # def resid(pars):
    #     predicted_vals = func(predictor, pars)
    #     # consider overdetermined case
    #     if predictor.shape[(-1)] > outcome.shape[(-1)]:
    #         predicted_vals = predicted_vals.sum(-1)
    #     return ((outcome - predicted_vals) ** 2).sum()
    # predictor_range = predictor.ptp(0)
    # def constr(pars):
    #     new_predictor = np.linspace(predictor.min(0) - predictor_range / 2, predictor.max(0) + predictor_range / 2, 1000)
    #     pred_vals = func(new_predictor, pars)
    #     deriv = np.diff(pred_vals)
    #     return min(deriv)
    # con1 = {'type':'ineq',  'fun':constr}
    # min_resids = np.inf
    # model = None
    # for deg in np.arange(2, 20, 1):
    #     pars = np.zeros(deg)
    #     pars[0] = 0.1
    #     res = minimize(resid, pars)#, method='cobyla', options={'maxiter': 50000})
    #     new_predictor = np.linspace(min(predictor), max(predictor))
    #     pred_vals = func(new_predictor, res.x)
    #     resids = resid(res.x)
    #     if resids <  min_resids:
    #         model = res
    #         min_resids = resids
    # def final_func(x):
    #     return np.polyval(model.x, x)
    # # normalize the residuals by the number of samples used
    # residuals_normalized = resid(model.x) / len(predictor)
    # return (final_func, residuals_normalized)


def angle_fitter(pts, lbls, angle_deviation_limit=np.pi / 3, display=False):
    """

    (1) Import 3D coordinates and clusterd by lbls. (2) Using the cluster 
    centers, fit a circle in order to do a polar transformation. (3) Use the 
    SVD of each cluster in 3D, projected onto the 2D plane, and then regress 
    the direction vectors on polar angle using robust linear modelling to account
    for noisy SVDs."""
    lbls_set = sorted(set(lbls))
    pts = np.array(pts)
    xs, ys, zs = pts.T
    centers = np.zeros((len(lbls), 3))
    group_centers = np.zeros((len(set(lbls)), 3))
    for num, lbl in enumerate(lbls_set):
        ind = lbls == lbl
        center = pts[ind].mean(0)
        centers[ind] = center
        group_centers[num] = center
    else:
        rays = group_centers[:, :2]
        norms = np.linalg.norm(rays, axis=1)
        rays = rays / np.linalg.norm(rays, axis=1)[:, np.newaxis]
        group_polar_angles = np.arctan2(rays[:, 1], rays[:, 0])
        svds = []
        lengths = []
        for lbl, ray, center in zip(lbls_set, rays, group_centers[:, :2]):
            ind = lbls == lbl
            sub_pts = pts[ind]
            svd = np.linalg.svd(sub_pts - sub_pts.mean(0))
            svds += [svd[(-1)][0]]
            lengths += [svd[1][0]]
            l = 5
        else:
            svds = np.array(svds)
            lengths = np.array(lengths)
            neg_svds = svds[:, 1] > 0
            svds[neg_svds, :2] *= -1
            include = np.ones(len(rays))
            xmean, ymean = svds[:, :2].mean(0)
            angles = np.arctan2(svds[:, 1], svds[:, 0])
            mean_ang = np.arctan2(ymean, xmean)
            rot_ang = mean_ang - np.pi / 2
            svds_rotated = rotate(svds, rot_ang, axis=2).T
            angles = np.arctan2(svds_rotated[:, 1], svds_rotated[:, 0])
            wls_mod = None
            aic = np.inf
            if include.sum() > 2:
                mod, resids = positive_fit(group_polar_angles, angles)
                wls_mod = mod
                new_xs = np.linspace(group_polar_angles.min(), group_polar_angles.max(), 100)
                new_ys = mod(new_xs)
                new_angs = mod(group_polar_angles)
                new_svds_rotated = np.copy(svds_rotated)
                new_svds_rotated[:, 1] = np.tan(new_angs)
                new_svds_rotated[:, 0] = 1
                new_svds = rotate(new_svds_rotated, (-rot_ang), axis=2).T
                norms = np.linalg.norm((new_svds[:, :2]), axis=(-1))
                new_svds /= norms[:, np.newaxis]
                if display:
                    fig = plt.figure()
                    new_lbls = dict()
                    new_vals = np.arange(len(set(lbls))) + 1
                    for lbl, new_val in zip(set(lbls), new_vals):
                        new_lbls[lbl] = new_val
                    else:
                        clbls = []
                        for lbl in lbls:
                            clbls += [new_lbls[lbl]]
                        else:
                            plt.scatter((pts[:, 0]), (pts[:, 1]), c=clbls, cmap='tab20')
                            for lbl, ray, svd, center in zip(lbls_set, rays, new_svds, group_centers[:, :2]):
                                ind = lbls == lbl
                                sub_pts = pts[ind]
                                l = 30
                                p1, p2 = center - l * svd[:2], center + l * svd[:2]
                                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color=red)
                                p1, p2 = center - l * ray, center + l * ray
                                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color=green)
                                plt.scatter((sub_pts[:, 0]), (sub_pts[:, 1]), marker='.', zorder=0)
                            else:
                                plt.gca().set_aspect('equal')
                                plt.show()

            else:
                new_svds = np.zeros((len(lbls_set), 3))
                new_svds.fill(np.nan)
                resids = np.inf
            return (
             lbls_set, new_svds[:, :2], resids)


def fit_circle(spX, spY):
    """Find best fitting sphere to x, y, and z coordinates using OLS."""
    f = np.zeros((len(spX), 1))
    f[:, 0] = spX ** 2 + spY ** 2
    A = np.zeros((len(spX), 3))
    A[:, 0] = spX * 2
    A[:, 1] = spY * 2
    A[:, 2] = 1
    C, residuals, rank, sigval = np.linalg.lstsq(A, f, rcond=None)
    t = C[0] ** 2 + C[1] ** 2 + C[2]
    radius = math.sqrt(t)
    return (radius, np.squeeze(C[:-1]), residuals)


class SphereFit:

    def __init__(self, pts):
        """Fit sphere equation to 3D points using scipy.optimize.minimize.

        Parameters
        ----------
        pts : np.ndarray, shape (N, 3)
            The array of 3D points to be fitted.
        """
        self.pts = np.copy(pts)
        self.original_pts = np.copy(self.pts)
        self.xs, self.ys, self.zs = self.pts.T
        outcome = (self.pts ** 2).sum(1)
        outcome = outcome[:, np.newaxis]
        coefficients = np.ones((len(self.xs), 4))
        coefficients[:, :3] = self.pts * 2
        solution, sum_sq_residuals, rank, singular = np.linalg.lstsq(coefficients,
          outcome, rcond=None)
        self.center = solution[:-1, 0]
        self.radii = np.linalg.norm((self.pts - self.center[np.newaxis]), axis=(-1))
        self.radius = np.mean(self.radii)
        self.pts -= self.center
        self.center_com()
        self.get_polar()

    def center_com(self):
        com = self.pts.mean(0)
        ang1 = np.arctan2(com[2], com[1])
        com1 = rotate(com, ang1, axis=0)
        rot1 = rotate((self.pts), ang1, axis=0).T
        ang2 = np.arctan2(com1[1], com1[0])
        rot2 = rotate(rot1, ang2, axis=2).T
        self.rot_ang1 = ang1
        self.rot_ang2 = ang2
        self.pts = rot2

    def get_polar(self):
        """Transform self.pts to polar coordinates using sphere center.

        Attributes
        ----------
        polar : np.ndarray, shape=(N,3)
            The list of coordinates transformed into spherical coordinates.
        """
        xs, ys, zs = self.pts.T
        radius = np.linalg.norm((self.pts), axis=(-1))
        inclination = np.arccos(zs / radius)
        azimuth = np.arctan2(ys, xs)
        self.polar = np.array([inclination, azimuth, radius]).T


class RotateClusterer:
    def __init__(self, pts_3d, centers_3d, test=False):
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
        test : bool, default=False
            Whether to plot stages of the minimization algorithm.
        """
        # make empty list to store iterations of the program
        self.log = []
        self.pts = np.copy(pts_3d)
        self.centers = np.copy(centers_3d)
        com = self.pts.mean(0)
        self.pts -= com
        self.centers -= com
        uu, dd, vv = np.linalg.svd(self.centers)
        self.centers = np.dot(self.centers, vv)
        self.pts = np.dot(self.pts, vv)
        # use 2 angular variables bound between 0 and pi
        self.theta = np.pi/2
        self.phi = np.pi/2
        # define bounds for theta and phi
        self.lower_bound = np.pi/6
        self.upper_bound = 5 * np.pi/6
        # store an orientation vector
        self.orientation = np.array([self.theta, self.phi])
        self.lbls = np.zeros(self.pts.shape[0], dtype=int)
        # minimize the error function using different orientations
        self.error(self.orientation)
        self.fmin = optimize.differential_evolution(
            # self.error, bounds=[(np.pi/4, 3*np.pi/4), (np.pi/4, 3*np.pi/4)])
            self.error, bounds=[(0, np.pi), (0, np.pi)])
        self.orientation = self.fmin.x
        self.error(self.orientation)
        # test: iterate through the log and plot each stage 
        if test:
            for num, (ori, corr, pts, centers, lbls) in enumerate(self.log[::5]):
                fig, ax = plt.subplots()
                fig.suptitle(f"{50*num}/{len(self.log)}, corr={corr}")
                xs, ys = pts.T
                xs_c, ys_c = centers.T
                ax.scatter(xs, ys, c=lbls+1, alpha=1, marker='.')
                ax.scatter(xs_c, ys_c, color='r')
                ax.set_aspect('equal')
                plt.show()
        
    def my_bounds(self, **kwargs):
        """Define boundaries for the basin hopping algorithm."""
        theta, phi = kwargs['x_new']
        in_bounds = theta > self.lower_bound
        in_bounds *= theta < self.upper_bound
        in_bounds *= phi > self.lower_bound
        in_bounds *= phi < self.upper_bound
        return in_bounds
        
    def error(self, orientation):
        """We want to minimize the KMeans inertia of the 2D projection.


        Parameters
        ----------
        orientation : array-like, len=2
        
        Attributes
        ----------
        self.orientation_vector : array-like, shape=(k, 3)
            The direction vector used in rotating the points
        """
        self.log += [orientation]
        # get the rectangular direction vector from the provided angles
        theta, phi = orientation
        # these represent rotations of the basis vectors
        yaw_matrix = np.array([
            [np.cos(phi), -np.sin(phi), 0],
            [np.sin(phi), np.cos(phi),  0],
            [0,           0,            1]])
        pitch_matrix = np.array([
            [np.cos(theta) , 0, np.sin(theta)],
            [0,              1, 0            ],
            [-np.sin(theta), 0, np.cos(theta)]])
        rot_matrix = np.dot(yaw_matrix, pitch_matrix)
        self.orientation_vector = np.dot(np.identity(3),
                                         rot_matrix)
        # get the 2d projection of pts and their centers
        pts_rotated = np.dot(self.pts, self.orientation_vector)
        centers_rotated = np.dot(self.centers, self.orientation_vector)
        pts_2d = pts_rotated[:, 1:]
        centers_2d = centers_rotated[:, 1:]
        # new_centers, lbls, inertia = cluster.k_means(pts_2d,
        #   n_clusters=(len(centers_2d)), init=centers_2d, n_init=1)
        # use KDTree to measure average distance from nearest centers as our error
        # test: make a 2d histogram rasterization and apply ODA 2D. If len(ommatidia) is
        # around the number of centers provided, use those centers instead of centers_2d. 
        # raster = np.histogram2d(pts_2d[:, 0], pts_2d[:, 1], bins=(50, 50))
        # mask = 255*ndimage.binary_dilation(raster[0] > 0).astype('uint8')
        # self.eye = Eye(arr=raster[0], mask_arr=mask)
        # self.eye.oda()
        # dist_tree = spatial.KDTree(centers_2d)
        # dists, lbls = dist_tree.query(pts_2d)
        # # store associated labels
        # self.lbls[:] = lbls
        # # store values
        # self.log += [(orientation, 1-self.eye.correlation_val, pts_2d, centers_2d, lbls)]
        # return 1-self.eye.correlation_val
        # old option that doesn't really work:
        dist_tree = spatial.KDTree(centers_2d)
        dists, lbls = dist_tree.query(pts_2d)
        # get mean distance per lbl
        lbls_set = sorted(set(lbls))
        dist_maxes = []
        group_aspect_ratio = []
        for lbl in lbls_set:
            inds = lbls == lbl
            dist_maxes += [dists[inds].mean()]
            # measure the average 
            if inds.sum() > 1:
                sub_pts = pts_2d[inds]
                sides = sub_pts.ptp(0)
                ratio = max(sides) / min(sides[sides > 0])
            else:
                ratio = 1
            group_aspect_ratio += [ratio]
        group_aspect_ratio = np.array(group_aspect_ratio)
        dist_maxes = np.array(dist_maxes)
        dist_maxes *= group_aspect_ratio
        # self.log += [(orientation, np.mean(dist_maxes), pts_2d, centers_2d, lbls)]
        self.lbls[:] = lbls
        return sum(dist_maxes)**2


class Points:

    def __init__(self, arr, center=[
 0, 0, 0], polar=None, sphere_fit=True, spherical_conversion=True, rotate_com=True, vals=None):
        """Import array of rectangular coordinates with some options.

        Parameters
        ----------
        arr : np.ndarray, shape (N, 3)
            The input array of 3D points.
        center_points : bool, default=True
            Whether to center the input points.
        polar : np.ndarr, default=None
            Option to input the polar coordinates, to avoid recentering.
        sphere_fit : bool, default=True
            Whether to fit a sphere to the coordinates and center.
        spherical_conversion : bool, default=Trued 
            Whether to calculate polar coordinates.
        rotate_com : bool, default=True
            Whether to rotate input coordinates so that the center of 
            mass is centered in terms of azimuth and inclination.
        vals : np.ndarray, shape (N)
            Values associated with each point in arr.

        Attributes
        ----------
        pts : array_like, shape=(N, 3)
            Array of 3D coordinates.
        original_pts : array_like, shape=(N, 3)
            Array of the input 3D coordinates before any rotations or 
            translations.
        shape : tuple, default=(N, 3)
            Shape of the 3D coordinates.
        center : array_like, default=[0, 0, 0]
            The 3D coordinate of the center point.
        raster : array_like, default=None
            The 2D raster image of the 3D coordinates.
        xvals, yvals : array_like, default=None
            The boundaries of the pixels in self.raster.
        polar : array_like, default=None
            Custom input polar coordinates (optional).
        sphere_model : SphereFit
            Model fitting a sphere to 3D points using OLS.
        radius : float
            Radius of the fitted sphere.
        center : array_like
            3D enter of the fitted sphere.
        """
        self.pts = np.array(arr)
        if vals is None:
            self.vals = np.ones(len(self.pts))
        else:
            self.vals = vals
        if self.pts.ndim > 1:
            assert self.pts.shape[1] == 3, f'Input array should have shape N x 3. Instead it has shape {self.pts.shape[0]} x {self.pts.shape[1]}.'
        else:
            assert self.pts.shape[0] == 3, f'Input array should have shape 3 or N x 3. Instead it has shape {self.pts.shape}.'
            self.pts = self.pts.reshape((1, -1))
        self.original_pts = self.pts
        self.shape = self.pts.shape
        self.center = np.asarray(center)
        self.raster = None
        self.xvals, self.yvals = (None, None)
        self.polar = None
        if polar is not None:
            self.polar = polar
            self.theta, self.phi, self.radii = self.polar.T
        if sphere_fit:
            self.sphere_model = SphereFit(self.pts)
            self.radius = self.sphere_model.radius
            self.center = self.sphere_model.center
            self.pts -= self.center
            self.center = self.center - self.center
        if spherical_conversion:
            if rotate_com:
                sample_inds = np.arange(len(self.pts))
                num_points = 1000
                sample_inds = np.random.choice(sample_inds, num_points)
                com = self.pts.mean(0)
                ang1 = np.arctan2(com[2], com[1])
                com1 = rotate(com, ang1, axis=0)
                rot1 = rotate((self.pts), ang1, axis=0).T
                ang2 = np.arctan2(com1[1], com1[0])
                rot2 = rotate(rot1, ang2, axis=2).T
                self.pts = rot2
            self.spherical()
        self.x, self.y, self.z = self.pts.T

    def __len__(self):
        return len(self.x)

    def __getitem__(self, key):
        out = Points((self.pts[key]), polar=(self.polar[key]), rotate_com=False,
          spherical_conversion=False,
          vals=(self.vals[key]))
        return out

    def spherical(self, center=None):
        """Perform the spherical transformation.

        Parameters
        ----------
        center : bool, default=None
            Option to input custom center point.

        Attributes
        ----------
        polar : array_like, shape=(N, 3)
            The polar coordinates of self.pts with respect to the input center.
        theta, phi, radii : array_like, shape=(N, 1)
            The azimuth, elevation, and radial distance from self.polar.
        residuals : array_like, shape=(N, 1)
            The differences between the radii and the fitted radius.
        """
        if center is None:
            center = self.center
        self.polar = rectangular_to_spherical((self.pts), center=center)
        self.theta, self.phi, self.radii = self.polar.T
        if 'radius' in dir(self):
            self.residuals = self.radii - self.radius

    def rasterize(self, polar=True, axes=[0, 1], image_size=10000, weights=None,
                  pixel_length=None, project=False):
        """Rasterize coordinates onto a grid defined by min and max vals.

        Parameters
        ----------
        polar : bool, default=True
            Whether to rasterize polar (vs. rectangular) coordinates.
        image_size : int, default=1e4
            The number of pixels in the image.
        weights : list, shape=(N, 1), default=None
            Optional weights associated with each point.
        pixel_length : float, default=None
            Alternative method of specifying the grid based on the pixel length, as 
            opposed to image size. This overrides the image size.
        project : bool, default=False
            Whether to project the points onto the fitted surface in a non-linear way.

        Returns
        -------
        raster : np.ndarray
            The 2D histogram of the points, optionally weighted by self.vals.
        (xs, ys) : tuple
            The x and y coordinates marking the boundaries of each pixel. 
            Useful for rendering as a pyplot.pcolormesh.
        """
        if polar:
            arr = self.polar
        else:
            arr = self.pts
        x, y = arr.T[axes]
        x_range = x.max() - x.min()
        y_range = y.max() - y.min()
        ratio = y_range / x_range
        if pixel_length is None:
            x_len = int(np.round(np.sqrt(image_size / ratio)))
            xs = np.linspace(x.min(), x.max(), x_len)
            self.raster_pixel_length = xs[1] - xs[0]
        else:
            self.raster_pixel_length = pixel_length
            xs = np.arange(x.min(), x.max(), self.raster_pixel_length)
        ys = np.arange(y.min(), y.max(), self.raster_pixel_length)
        avg = np.histogram2d(x, y, bins=(xs, ys))[0]
        if weights is not None:
            total = np.histogram2d(x, y, bins=(xs, ys), weights=weights)[0]
            avg = total/avg
        self.raster = avg
        xs = xs[:-1] + self.raster_pixel_length / 2.0
        ys = ys[:-1] + self.raster_pixel_length / 2.0
        self.xvals, self.yvals = xs, ys
        return (self.raster, (xs, ys))

    def surface_projection(self, image_size=10000):
        """Rasterize coordinates onto a grid defined by min and max vals.

        Parameters
        ----------
        image_size : int, default=1e4
            The number of pixels in the image.

        Returns
        -------
        raster : np.ndarray
            The 2D histogram of the points, optionally weighted by self.vals.
        (xs, ys) : tuple
            The x and y coordinates marking the boundaries of each pixel. 
            Useful for rendering as a pyplot.pcolormesh.
        """
        # 1. fit a surface to the points in polar coordinates, approximating radius
        # as the average radius of points within a cell of a grid
        self.fit_surface(polar=True, image_size=1e6)
        # 2. convert surface to recangular coordinates
        surface = spherical_to_rectangular(self.avg)
        theta, phi, radii = self.avg.T
        # 3. for each point, find the nearest point on the surface
        # each projection point should have the 
        dist_tree = spatial.KDTree(surface)
        dists, inds = dist_tree.query(self.pts) # inds point to the nearest points in surface
        proj_radii = radii[inds]
        dists[self.radii < proj_radii] *= -1
        # todo: try using linear interpolation 
        # breakpoint()
        interp = interpolate.LinearNDInterpolator(
            surface[:, :2], surface[:, 2])
        surface_model = interp(self.pts[:, 0], self.pts[:, 1])

        self.surface
        # no_nans = np.isnan(cubic_surface) == False
        # self.surface[sort_inds][no_nans] = cubic_surface[no_nans]
        

        # make projected coordinates using the polar angles of the surface points
        # nearest each element
        proj_theta, proj_phi = theta[inds], phi[inds]
        proj_coords = np.array([proj_theta, proj_phi, dists]).T
        fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)
        axes[0].scatter(proj_theta, proj_phi, c=dists, marker='.', edgecolor='none',
                        alpha=.1)
        axes[1].scatter(self.theta, self.phi, c=self.radii - self.radii.mean(),
                        marker='.', edgecolor='none', alpha=.1)
        [ax.set_aspect('equal') for ax in axes]
        plt.show()


    def fit_surface(self, polar=True, outcome_axis=0, image_size=10000.0):
        """Cubic interpolate surface of one axis using the other two.

        Parameters
        ----------
        polar : bool, default=True
            Whether to fit a surface using polar coordinates.
        outcome_axis : int, default=0
            The axis to use as the outcome of the other axes.
        image_size : int, default=1e4
            The number of pixels in the image.

        Attributes
        ----------
        avg : array_like
            The rolling average
        """
        if polar:
            arr = self.polar
        else:
            arr = self.pts
        x, y, z = arr.T
        x_range = x.max() - x.min()
        y_range = y.max() - y.min()
        ratio = y_range / x_range
        x_len = int(np.round(np.sqrt(image_size / ratio)))
        y_len = int(np.round(ratio * x_len))
        xs = np.linspace(x.min(), x.max(), x_len)
        ys = np.linspace(y.min(), y.max(), y_len)
        avg = []
        print("\nBin averaging the 2D image:")
        for col_num, (x1, x2) in enumerate(zip(xs[:-1], xs[1:])):
            col = []
            in_column = np.logical_and(x >= x1, x < x2)
            in_column = arr[in_column]
            for row_num, (y1, y2) in enumerate(zip(ys[:-1], ys[1:])):
                in_row = np.logical_and(in_column[:, 1] >= y1, in_column[:, 1] < y2)
                if any(in_row):
                    avg += [np.mean((in_column[in_row]), axis=0)]
                print_progress(col_num, len(xs) - 1)
        print()
        avg = np.array(avg)
        self.avg = avg
        self.avg_x, self.avg_y, self.avg_z = self.avg.T

    def surface_predict(self, xvals=None, yvals=None, polar=True, image_size=10000.0):
        """Find the approximate zvalue given arbitrary x and y values."""
        if 'avg_x' not in dir(self):
            self.fit_surface(polar=polar, image_size=image_size)
        if xvals is None or yvals is None:
            if polar:
                arr = self.polar
            else:
                arr = self.pts
            xvals, yvals, zvals = arr.T
        points = np.array([xvals, yvals]).T
        sort_inds = np.argsort(points[:, 0])
        self.surface = np.zeros((len(points)), dtype=float)
        self.surface[sort_inds] = interpolate.griddata(
            (self.avg[:, :2]), (self.avg_z), (points[sort_inds]), method='nearest')
        no_nans = np.any(np.isnan(self.avg[:, :2]), axis=-1) == False
        no_nans *= np.isnan(self.avg_z) == False
        cubic_surface = interpolate.griddata(
            self.avg[:, :2][no_nans], self.avg_z[no_nans],
            points[sort_inds], method='linear')
        no_nans = np.isnan(cubic_surface) == False
        self.surface[sort_inds][no_nans] = cubic_surface[no_nans]
        # try:
        #     cubic_surface = interpolate.griddata(
        #         (self.avg[:, :2]), (self.avg_z), (points[sort_inds]), method='cubic')
        #     no_nans = np.isnan(cubic_surface) == False
        #     self.surface[sort_inds][no_nans] = cubic_surface[no_nans]
        # except:
        #     pass

    def get_polar_cross_section(self, thickness=0.1, pixel_length=0.01):
        """Find best fitting surface of radii using phis and thetas."""
        self.surface_predict(polar=True)
        self.residuals = self.radii - self.surface
        no_nans = np.isnan(self.residuals) == False
        self.cross_section_thickness = np.percentile(abs(self.residuals[no_nans]), thickness * 100)
        self.surface_lower_bound = self.surface - self.cross_section_thickness
        self.surface_upper_bound = self.surface + self.cross_section_thickness
        cross_section_inds = np.logical_and(self.radii <= self.surface_upper_bound, self.radii > self.surface_lower_bound)
        self.cross_section = self[cross_section_inds]

    def save(self, fn):
        """Save using pickle."""
        with open(fn, 'wb') as (pickle_file):
            pickle.dump(self, pickle_file)


class LSqEllipse:
    def __init__(self):
        return

    def fit(self, data):
        """Lest Squares fitting algorithm

        Theory taken from (*)
        Solving equation Sa=lCa. with a = |a b c d f g> and a1 = |a b c>
            a2 = |d f g>

        Args
        ----
        data (list:list:float): list of two lists containing the x and y data of the
            ellipse. of the form [[x1, x2, ..., xi],[y1, y2, ..., yi]]

        Returns
        ------
        coef (list): list of the coefficients describing an ellipse
           [a,b,c,d,f,g] corresponding to ax**2+2bxy+cy**2+2dx+2fy+g
        """
        x, y = np.asarray(data, dtype=float)
        D1 = np.mat(np.vstack([x ** 2, x * y, y ** 2])).T
        D2 = np.mat(np.vstack([x, y, np.ones(len(x))])).T
        S1 = D1.T * D1
        S2 = D1.T * D2
        S3 = D2.T * D2
        C1 = np.mat('0. 0. 2.; 0. -1. 0.; 2. 0. 0.')
        M = C1.I * (S1 - S2 * S3.I * S2.T)
        eval, evec = np.linalg.eig(M)
        cond = 4 * np.multiply(evec[0, :], evec[2, :]) - np.power(evec[1, :], 2)
        a1 = evec[:, np.nonzero(cond.A > 0)[1]]
        a2 = -S3.I * S2.T * a1
        self.coef = np.vstack([a1, a2])
        self._save_parameters()

    def _save_parameters(self):
        """finds the important parameters of the fitted ellipse

        Theory taken form http://mathworld.wolfram

        Args
        -----
        coef (list): list of the coefficients describing an ellipse
           [a,b,c,d,f,g] corresponding to ax**2+2bxy+cy**2+2dx+2fy+g

        Returns
        _______
        center (List): of the form [x0, y0]
        width (float): major axis
        height (float): minor axis
        phi (float): rotation of major axis form the x-axis in radians
        """
        a = self.coef[(0, 0)]
        b = self.coef[(1, 0)] / 2.0
        c = self.coef[(2, 0)]
        d = self.coef[(3, 0)] / 2.0
        f = self.coef[(4, 0)] / 2.0
        g = self.coef[(5, 0)]
        x0 = (c * d - b * f) / (b ** 2.0 - a * c)
        y0 = (a * f - b * d) / (b ** 2.0 - a * c)
        numerator = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
        denominator1 = (b * b - a * c) * ((c - a) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
        denominator2 = (b * b - a * c) * ((a - c) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
        width = np.sqrt(numerator / denominator1)
        height = np.sqrt(numerator / denominator2)
        phi = 0.5 * np.arctan(2.0 * b / (a - c))
        self._center = [
         x0, y0]
        self._width = width
        self._height = height
        self._phi = phi

    @property
    def center(self):
        return self._center

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def phi(self):
        """angle of counterclockwise rotation of major-axis of ellipse to x-axis
        [eqn. 23] from (**)
        """
        return self._phi

    def parameters(self):
        return (
         self.center, self.width, self.height, self.phi)


class Layer:

    def __init__(self, filename=None, arr=None, bw=False):
        """An image loaded from file or numpy array.

        Parameters
        ----------
        filename : str, default=None
            Path to the image file.
        arr : array_like, default=None
            Input image as a 2D array.
        bw : bool, default=False
            Whether the image is greyscale.

        Returns
        -------
        out : Layer
              An general image object with various methods for image processing.
        """
        self.filename = filename
        self.image = arr
        self.bw = bw
        self.gradient = None
        self.color_selector = None
        self.mask = None

    def load(self):
        """Load image using PIL.

        Returns
        -------
        self.image : np.ndarray
            The loaded or directly specified image.
        """
        if self.image is not None:
            if not isinstance(self.image, np.ndarray):
                self.image = np.asarray(self.image)
            else:
                if self.image.ndim <= 1:
                    raise AssertionError('Input array should be at least 2D')
        else:
            assert isinstance(self.filename, str), 'Input filename should be a string.'
            self.image = load_image(self.filename)
        if self.image.ndim == 2:
            self.bw = True
        elif self.image.ndim == 3:
            if self.image.shape[(-1)] == 1:
                self.image = np.squeeze(self.image)
            else:
                if self.image.shape[(-1)] > 3:
                    self.image = self.image[..., :-1]
            if (self.image[..., 0] == self.image.mean(-1)).mean() == 1:
                self.image = self.image[(..., 0)]
                self.bw = True
        if self.bw:
            self.image_bw = self.image.astype('uint8')
        else:
            self.image_bw = rgb_2_gray(self.image.astype('uint8'))
        return self.image

    def load_memmap(self, filename=None):
        """Load image and store as a numpy memmap, deleting the local copy.

        Returns
        -------
        self.image : np.memmap
            The loaded or directly specified image stored to memory.
        """
        self.load()
        if self.filename is None and filename is None:
            memmap_fn = os.path.join(mkdtemp(), 'temp_img.memmap')
        else:
            if filename is not None:
                file_ext = '.' + filename.split('.')[(-1)]
                memmap_fn = filename.replace(file_ext, '.memmap')
            else:
                if self.filename is not None:
                    file_ext = '.' + self.filename.split('.')[(-1)]
                    memmap_fn = self.filename.replace(file_ext, '.memmap')
                elif os.path.exists(memmap_fn):
                    memmap = np.memmap(memmap_fn, mode='r+', shape=(self.image.shape))
                memmap = np.memmap(memmap_fn,
                  dtype='uint8', mode='w+', shape=(self.image.shape))
                memmap[:] = self.image[:]
                self.image = memmap

    def save(self, pickle_fn):
        """Save using pickle.

        Parameters
        ----------
        pickle_fn : str
            Filename of the pickle file to save.
        """
        self.pickle_fn = pickle_fn
        with open(pickle_fn, 'wb') as (pickle_file):
            pickle.dump(self, pickle_file)

    def get_gradient(self, smooth=0):
        """Measure the relative focus of each pixel using numpy.gradient.

        Parameters
        ----------
        smooth : float, default=0
            standard devation of 2D gaussian filter applied to the gradient.

        Returns
        -------
        self.gradient : np.ndarray
            2D array of the magnitude of the gradient image.
        """
        if not self.image is not None:
            raise AssertionError(f"No image loaded. Try running {self.load} or {self.load_memmap}")
        elif not self.bw:
            gray = rgb_2_gray(self.image)
        else:
            gray = self.image
        grad_0 = np.gradient(gray, axis=0)
        grad_1 = np.gradient(gray, axis=1)
        self.gradient = np.linalg.norm((np.array([grad_0, grad_1])), axis=0)
        if smooth > 0:
            self.gradient = ndimage.filters.gaussian_filter((self.gradient), sigma=smooth)
        return self.gradient

    def color_key(self, hue_only=False):
        """Use ColorSelector to apply a mask based on color statistics.
        

        Parameters
        ----------
        hue_only : bool
            Whehter to yse only the hue channel of the images.

        Returns
        -------
        self.mask : np.ndarray
            2D array of the sillouetting mask.
        """
        if self.image is None:
            self.load()
        self.color_selector = ColorSelector(
            self.image, bw=(self.bw), hue_only=hue_only)
        self.color_selector.start_up()
        self.mask = self.color_selector.mask
        return self.mask

    def load_mask(self, mask_fn=None, mask_arr=None):
        """Load a 2D sillhouetting mask from an image file or array.

        
        If the image isn't boolean, we assume pixels > mean == True. 
        You can either load from an image file or directly as an array.

        Parameters
        ----------
        mask_fn : str, default=None
            Path to the masking image file. 
        mask_arr : array_like bool, default=None
            2D boolean masking array. 
        """
        if mask_fn is not None:
            assert isinstance(mask_fn, str), 'Input mask_fn should be a string.'
            if os.path.exists(mask_fn):
                layer = Layer(mask_fn, bw=True)
                self.mask = layer.load()
        elif mask_arr is not None:
            self.mask = np.asarray(mask_arr)
            if not self.mask.ndim > 1:
                raise AssertionError('Input mask_arr should be at least 2D')
        if self.mask is not None:
            if self.mask.dtype is not np.dtype('bool'):
                self.mask = self.mask > self.mask.max() / 2
            assert self.mask.shape == self.image.shape[:2], f"input mask should have the same shape as input image. input shape = {self.mask.shape}, image shape = {self.image.shape[:2]}"
            assert self.mask.mean() > 0, 'input mask is empty'


class Eye(Layer):

    def __init__(self, filename=None, arr=None, bw=False, pixel_size=1, mask_fn=None, mask_arr=None):
        """A class specifically for processing images of compound eyes. 
        

        Parameters
        ----------
        filename : str
            The file path to the eye image.
        bw : bool
            Whether the image in greyscale.
        pixel_size : float, default = 1
            The actual length of the side of one pixel.
        mask_fn : str, default = "mask.jpg"
            The path to the sillhouetting mask image file.
        mask : array_like, default = None
            Boolean masking image with the same shape as the input image array.

        Methods
        -------
        get_eye_outline
            Get the outline of the eye based on an eye mask.
        get_eye_dimensions
            Assuming an elliptical eye, get its length, width, and area.
        crop_eye
            Crop the image so that the frame is filled by the eye with padding.
        get_ommatidia
            Detect ommatidia coordinates assuming hex or square lattice.
        measure_ommatidia
            Measure ommatidial diameter using the ommatidia coordinates.
        ommatidia_detecting_algorithm
            The complete algorithm for measuring ommatidia in images.
        """
        Layer.__init__(self, filename=filename, arr=arr, bw=bw)
        self.eye_contour = None
        self.ellipse = None
        self.ommatidia = None
        self.pixel_size = pixel_size
        self.mask_fn = mask_fn
        self.mask_arr = mask_arr
        self.load()
        self.pickle_fn = None
        self.load_mask(mask_fn=(self.mask_fn), mask_arr=(self.mask_arr))
        self.oda = self.ommatidia_detecting_algorithm

    def get_eye_outline(self, hue_only=False, smooth_factor=11):
        """Get the outline of the eye based on an eye mask.

        Parameters
        ----------
        hue_only : bool, default=False
            Whether to filter using only the hue values.
        smooth_factor : int, default=11
            Size of 2D median filter to smooth outline. smooth_factor=0 -> 
            no smoothing.

        Attributes
        ----------
        eye_outline : np.ndarray
            2D coordinates of N points on the eye contour with shape N x 2.
        eye_mask : np.ndarray
            2D masking image of the eye smoothed and filled.
        """
        assert self.mask is not None, f"No boolean mask loaded. First try running {self.load_mask}"
        mask = np.zeros((np.array(self.mask.shape) + 2), dtype=bool)
        mask[1:-1, 1:-1] = self.mask
        contour = skimage.measure.find_contours(255 / mask.max() * mask.astype(int), 128.0)
        assert len(contour) > 0, 'could not find enough points in the contour'
        contour = max(contour, key=len).astype(float)
        contour -= 1
        contour = np.round(contour).astype(int)
        self.eye_outline = contour
        new_mask = np.zeros((self.mask.shape), dtype=int)
        new_mask[(contour[:, 0], contour[:, 1])] = 1
        ndimage.binary_fill_holes(new_mask, output=new_mask)
        if smooth_factor > 0:
            new_mask = signal.medfilt2d(new_mask.astype('uint8'), smooth_factor).astype(bool)
        self.eye_mask = new_mask

    def get_eye_dimensions(self, display=False):
        """Assuming an elliptical eye, get its length, width, and area.

        Parameters
        ----------
        display : bool, default=False
            Whether to plot the eye with the ellipse superimposed.

        Attributes
        ----------
        ellipse : LSqEllipse
            Ellipse class that uses OLS to fit an ellipse to contour data.
        eye_length : float
            Major diameter of the fitted ellipse.
        eye_width : float
            Minor diameter of the fitted ellipse.
        eye_area : float
            Area of the fitted ellipse
        """
        assert self.eye_outline is not None, f"first run {self.get_eye_outline}"
        least_sqr_ellipse = LSqEllipse()
        least_sqr_ellipse.fit(self.eye_outline.T)
        self.ellipse = least_sqr_ellipse
        center, width, height, phi = self.ellipse.parameters()
        self.eye_length = 2 * self.pixel_size * max(width, height)
        self.eye_width = 2 * self.pixel_size * min(width, height)
        self.eye_area = np.pi * self.eye_length / 2 * self.eye_width / 2
        if display:
            plt.imshow(self.image)
            plt.plot(self.eye_outline[:, 1], self.eye_outline[:, 0])
            plt.show()

    def crop_eye(self, padding=1.05, use_ellipse_fit=False):
        """Crop the image so that the frame is filled by the eye with padding.

        Parameters
        ----------
        padding : float, default=1.05
            Proportion of the length of the eye to include in width and height.
        use_ellipse_fit : bool, default=False
            Whether to use the fitted ellipse to mask the eye.

        Returns
        -------
        self.eye : Eye
            A cropped Eye using the boolean mask.
        """
        out = np.copy(self.image)
        if use_ellipse_fit:
            least_sqr_ellipse = LSqEllipse()
            least_sqr_ellipse.fit(self.eye_outline.T)
            self.ellipse = least_sqr_ellipse
            (x, y), width, height, ang = self.ellipse.parameters()
            self.angle = ang
            w = padding * width
            h = padding * height
            ys, xs = Ellipse(x,
              y, w, h, shape=(self.image.shape[:2]), rotation=ang)
            new_mask = self.mask[min(ys):max(ys), min(xs):max(xs)]
            self.eye = Eye(arr=(out[min(ys):max(ys), min(xs):max(xs)]), mask_arr=new_mask,
              pixel_size=(self.pixel_size))
        else:
            xs, ys = np.where(self.mask)
            minx, maxx, miny, maxy = (min(xs), max(xs), min(ys), max(ys))
            minx -= padding / 2
            miny -= padding / 2
            maxx += padding / 2
            maxy += padding / 2
            minx, maxx, miny, maxy = (
                int(round(minx)),
                int(round(maxx)),
                int(round(miny)),
                int(round(maxy)))
            minx, miny = max(minx, 0), max(miny, 0)
            new_mask = self.mask[minx:maxx, miny:maxy]
            self.eye = Eye(arr=(out[minx:maxx, miny:maxy]), mask_arr=new_mask,
              pixel_size=(self.pixel_size))
        return self.eye

    def get_ommatidia(self, bright_peak=True, fft_smoothing=5, square_lattice=False,
                      high_pass=False, regular=True):
        """Detect ommatidia coordinates assuming hex or square lattice.

        Use the ommatidia detecting algorithm (ODA) to find the center of
        ommatidia assuming they are arranged in a hexagonal lattice. Note: 
        This can be computationally intensive on larger images so we suggest 
        cropping out irrelevant regions via self.crop_eye().

        Parameters
        ----------
        bright_peak : bool, default=True
            Whether the ommatidia are defined by brighter (vs. darker) peaks.
        fft_smoothing : int, default=5
            The standard deviation of a 2D gaussian filter applied to the 
            reciprocal image before finding peaks.
        square_lattice : bool, default=False
            Whether this a square (rather than a hexagonal) lattice.
        high_pass : bool, default=False
            Whether to also filter frequencies below the fundamental one.
        regular : bool, default=False
            Whether to assume the ommatidial lattice is approximately regular.
        
        Atributes
        ---------
        __freqs :  np.ndarray
            2D image of spatial frequencies corresponding to the reciprocal 
            space of the 2D FFT.
        __orientations : np.ndarray
            2D image of spatial orientations corresponding to the reciprocal 
            space of the 2D FFT.
        __fundamental_frequencies : float
            The set of spatial frequencies determined by the peak frequencies 
            in the reciprocal image.
        __upper_bound : float
            The threshold frequency used in the low-pass filter = 1.25 * 
            max(self.fundamental_frequencies)
        __low_pass_filter : np.ndarray, dtype=bool
            2D boolean mask used as a low-pass filter on the reciprocal image.
        __fft_shifted : np.ndarray, dtype=complex
            The filtered 2D FFT of the image with low frequencies shifted to 
            the center.
        __fft : np.ndarray, dtype=complex
            The filtered 2D FFT of the image.
        filtered_image : np.ndarray
            The filtered image made by inverse transforming the filtered 2D fft.
        ommatidial_diameter_fft : float
            The average wavelength of the fundamental frequencies, 
            corresponding to the ommatidial diameters.
        ommatidial_inds : np.ndarray
            2D indices of the N ommatidia with shape N x 2.
        ommatidia : np.ndarray
            2D coordinates of N ommatidia with shape N x 2.
        reciprocal : np.ndarray
            2D reciprocal image of self.image, correcting for the natural 
            1/(f^2) distribution of spatial frequencies and the low horizontal
            and vertical spatial frequencies corresponding to the vertical
            and horizontal boundaries.
        """
        if not self.eye_outline is not None:
            raise AssertionError(f"first run {self.get_eye_dimensions}")
        else:
            image_bw_centered = self.image_bw - self.image_bw.mean()
            height, width = image_bw_centered.shape
            window_h = signal.windows.gaussian(height, height / 3)
            window_w = signal.windows.gaussian(width, height / 3)
            window = window_w[np.newaxis, :] * window_h[:, np.newaxis]
            image_windowed = image_bw_centered * window
            fft = np.fft.fft2(image_windowed)
            fft_shifted = np.fft.fftshift(fft)
            xfreqs = np.fft.fftfreq(self.image_bw.shape[1], self.pixel_size)
            yfreqs = np.fft.fftfreq(self.image_bw.shape[0], self.pixel_size)
            xgrid, ygrid = np.meshgrid(xfreqs, yfreqs)
            self._Eye__freqs = np.array([xgrid, ygrid])
            self._Eye__freqs = np.array((self._Eye__freqs), dtype=float)
            self._Eye__freqs = np.fft.fftshift(self._Eye__freqs)
            self._Eye__xfreqs = np.fft.fftshift(xfreqs)
            self._Eye__yfreqs = np.fft.fftshift(yfreqs)
            self._Eye__orientations = np.arctan2(self._Eye__freqs[1], self._Eye__freqs[0])
            self._Eye__freqs = np.linalg.norm((self._Eye__freqs), axis=0)
            i = self._Eye__orientations < 0
            self._Eye__orientations[i] = self._Eye__orientations[i] + np.pi
            self.reciprocal = abs(fft_shifted)
            height, width = self.reciprocal.shape
            if regular:
                # use 2d cross correlation 
                self.reciprocal = signal.correlate(
                    (self.reciprocal), (self.reciprocal), mode='same', method='fft')
                # normalize to the maximum correlation for comparisons
                correlations = self.reciprocal/self.reciprocal.max()
                # find the peak of the upper half of the reciprocal image
                pos_freqs = self._Eye__freqs > 0
                thresh = 2 * self._Eye__freqs[pos_freqs].min()
                # find peak frequencies
                peaks = peak_local_max((self.reciprocal),
                                       num_peaks=10, min_distance=5)
                ys, xs = peaks.T
                # get the key frequencies and correlation values
                key_vals = correlations[ys, xs]
                key_freqs = self._Eye__freqs[(ys, xs)]
                # the center point is unique, has maximum correlation,and has
                # approximately 0 frequency
                thresh = 3 * np.unique(self._Eye__freqs)[1]  # 3 x smallest distance
                include = key_freqs > thresh
                include *= key_vals < 1.0
                key_freqs = key_freqs[include]
                key_vals = key_vals[include]
                self.correlation_val = key_vals[:6].mean()
                # remove any without a duplicate
                key_freq_set, counts = np.unique(key_freqs, return_counts=True)
                include = counts > 1
                key_freqs = key_freq_set[include]
                i = np.argsort(key_freqs)
                if square_lattice:
                    self._Eye__fundamental_frequencies = key_freqs[i][:2]
                else:
                    self._Eye__fundamental_frequencies = key_freqs[i][:3]
            else:
                product = np.log(self.reciprocal)
                freqs = self._Eye__freqs.flatten()
                power_normalized = product.flatten()
                hist, xvals, yvals = np.histogram2d(freqs, power_normalized, bins=50)
                xs = (xvals[:-1] + xvals[:-1]) / 2
                ys = (yvals[:-1] + yvals[:-1]) / 2
                weighted_means = []
                for col in hist:
                    weighted_means += [sum(col * ys / col.sum())]
                weighted_means = np.array(weighted_means)
                xmax = peak_local_max(weighted_means, num_peaks=1)
                if len(xmax) > 0:
                    xmax = xmax[0][0]
                    peak_frequency = xs[xmax]
                    self._Eye__fundamental_frequencies = np.array([peak_frequency])
                else:
                    self._Eye__fundamental_frequencies = np.array([])
        self.ommatidial_diameter_fft = 1 / self._Eye__fundamental_frequencies.mean()
        dist = self.ommatidial_diameter_fft / self.pixel_size
        if len(self._Eye__fundamental_frequencies) > 0 and dist > 2:
            self._Eye__upper_bound = 1.25 * self._Eye__fundamental_frequencies.max()
            in_range = self._Eye__freqs < self._Eye__upper_bound
            self._Eye__low_pass_filter = np.ones(self._Eye__freqs.shape)
            self._Eye__low_pass_filter[in_range == False] = 0
            if high_pass:
                in_range = self._Eye__freqs < 0.75 * self._Eye__fundamental_frequencies.min()
                self._Eye__low_pass_filter[in_range] = 0
            self._Eye__fft_shifted = np.zeros((fft.shape), dtype=complex)
            self._Eye__fft_shifted[:] = fft_shifted * self._Eye__low_pass_filter
            self._Eye__fft = np.fft.ifftshift(self._Eye__fft_shifted)
            self.filtered_image = np.fft.ifft2(self._Eye__fft).real
            smooth_surface = self.filtered_image
            if not bright_peak:
                smooth_surface = smooth_surface.max() - smooth_surface
            self.ommatidial_inds = peak_local_max(smooth_surface,
              min_distance=(int(round(dist / 4))), exclude_border=False,
              threshold_abs=1)
            ys, xs = self.ommatidial_inds.T
            self.ommatidial_inds = self.ommatidial_inds[self.mask[(ys, xs)]]
            self.ommatidia = self.ommatidial_inds * self.pixel_size
        else:
            print('Failed to find fundamental frequencies.')
            empty_img = np.copy(self.image)
            empty_img[:] = empty_img.mean().astype('uint8')
            self.filtered_image = np.copy(empty_img)
            self.ommatidial_diameter_fft = np.nan
            self.ommatidial_inds = np.array([])
            self.ommatidia = np.array([])

    def measure_ommatidia(self, num_neighbors=3, sample_size=100):
        """Measure ommatidial diameter using the ommatidia coordinates.

        Once the ommatidia coordinates are measured, we can measure ommatidial
        diameter given the expected number of neighbors

        Parameters
        ----------
        num_neighbors : int, default=6
            The number of neighbors to check for measuring the ommatidial 
            diameter. Defaults to 6, assuming a hexagonal lattice.
        sample_size : int, default=100
            The number of ommatidia near the center to include in diameter
            estimation.

        Atributes
        ---------
        __ommatidial_dists_tree : scipy.spatial.kdtree.KDTree
            K-dimensional tree for efficiently taking distance measurements.
        __ommatidial_dists : np.ndarray
            N X num_neighbors array of the distance to neighboring ommatidia.
        ommatidial_diameters : np.ndarray
            1-D array of average diameter per ommatidium.
        ommatidial_diameter : float
            Average ommatidial diameter of sample near the mask center of mass.
        ommatidial_diameter_SD : float
            Standard deviation of ommatidial diameters in sample.
        """
        if not self.ommatidia is not None:
            raise AssertionError(f"first run {self.get_ommatidia}")
        elif len(self.ommatidia) > 0:
            self._Eye__ommatidial_dists_tree = spatial.KDTree(self.ommatidia)
            self._Eye__ommatidial_dists, inds = self._Eye__ommatidial_dists_tree.query((self.ommatidia),
              k=(num_neighbors + 1))
            self.ommatidial_diameters = self._Eye__ommatidial_dists[:, 1:].mean(1)
            com = self.ommatidia.mean(0)
            near_dists, near_center = self._Eye__ommatidial_dists_tree.query(com,
              k=sample_size)
            near_dists, near_center = self._Eye__ommatidial_dists_tree.query(com,
              k=sample_size)
            near_center = near_center[(near_dists < np.inf)]
            self.ommatidial_diameter = self.ommatidial_diameters[near_center].mean()
            self.ommatidial_diameter_SD = self.ommatidial_diameters[near_center].std()
        else:
            self.ommatidial_diameter = np.nan
            self.ommatidial_diameter_SD = np.nan

    def ommatidia_detecting_algorithm(self, bright_peak=True, fft_smoothing=5,
                                      square_lattice=False, high_pass=False,
                                      num_neighbors=3, sample_size=100, plot=False,
                                      plot_fn=None, regular=True, manual_edit=False):
        """The complete algorithm for measuring ommatidia in images.

        
        Parameters
        ----------
        bright_peak : bool, default=True
            Whether the ommatidia are defined by brighter (vs. darker) peaks.
        fft_smoothing : int, default=5
            The standard deviation of a 2D gaussian filter applied to the 
            reciprocal image before finding peaks.
        square_lattice : bool, default=False
            Whether this a square (rather than a hexagonal) lattice.
        high_pass : bool, default=False
            Whether to also filter frequencies below the fundamental one.
        num_neighbors : int, default=6
            The number of neighbors to check for measuring the ommatidial 
            diameter. Defaults to 6, assuming a hexagonal lattice.
        sample_size : int, default=100
            The number of ommatidia near the center to include in diameter
            estimation.
        plot : bool, default=False
            Whether to plot the eye with ommatidia and diameters superimposed.
        plot_fn : str, default=None
            Filename to save the plotted eye with superimposed ommatidia and 
            their diameters.
        regular : bool, default=False
            Whether to assume the ommatidial lattice is approximately regular.
        manual_edit : bool
            Whether to allow the user to manually edit the ommatidia coordinates.

        Attributes
        ----------
        eye_length : float
            Major diameter of the fitted ellipse.
        eye_width : float
            Minor diameter of the fitted ellipse.
        eye_area : float
            Area of the fitted ellipse
        ommatidia : np.ndarray
            2D coordinates of N ommatidia with shape N x 2.
        ommatidial_diameter_fft : float
            The average wavelength of the fundamental frequencies, 
            corresponding to the ommatidial diameters.
        ommatidial_diameter : float
            Average ommatidial diameter of sample near the mask center of mass.
        ommatidial_diameter_SD : float
            Standard deviation of ommatidial diameters in sample.
        """
        self.get_eye_outline()
        self.get_eye_dimensions()
        area, length, width = np.round([
            self.eye_area, self.eye_length, self.eye_width], 3)
        print(f"Eye: \tArea = {area}\tLength = {length}\tWidth = {width}")
        self.get_ommatidia(bright_peak=bright_peak,
                           fft_smoothing=fft_smoothing,
                           square_lattice=square_lattice,
                           high_pass=high_pass,
                           regular=regular)
        # todo: allow for editing the ommatidia coordinates
        if manual_edit:
            fix_ommatidia = OmmatidiaGUI(img_arr=self.image, coords_arr=self.ommatidial_inds)
            self.ommatidial_inds = fix_ommatidia.coords
            self.ommatidia = self.pixel_size * self.ommatidial_inds
        self.measure_ommatidia(num_neighbors=num_neighbors, sample_size=sample_size)
        count = len(self.ommatidia)
        sample_diameter = np.round(self.ommatidial_diameter, 4)
        sample_std = np.round(self.ommatidial_diameter_SD, 4)
        fft_diameter = np.round(self.ommatidial_diameter_fft, 4)
        print(f"Ommatidia: \tN={count}\tmean={sample_diameter}\tstd={sample_std}\tfft={fft_diameter}")
        print()
        if plot or plot_fn is not None:
            fig = plt.figure()
            gridspec = fig.add_gridspec(ncols=2, nrows=1, width_ratios=[9, 1])
            img_ax = fig.add_subplot(gridspec[(0, 0)])
            colorbar_ax = fig.add_subplot(gridspec[(0, 1)])
            img_ax.imshow((self.image_bw), cmap='gray', vmin=0,
              vmax=(np.iinfo(self.image.dtype).max))
            img_ax.set_xticks([])
            img_ax.set_yticks([])
            if len(self.ommatidia) > 0:
                dot_radii = self.ommatidial_diameters / (2 * self.pixel_size)
                dot_areas = np.pi * dot_radii ** 2
                colorvals = self.ommatidial_diameters
                vmin, vmax = np.percentile(colorvals, [0.5, 99.5])
                ys, xs = self.ommatidial_inds.T
                img_ax.scatter(xs, ys, marker='.', c=colorvals, vmin=vmin,
                  vmax=vmax,
                  cmap='plasma')
                ys, xs = np.where(self.mask)
                width = xs.max() - xs.min()
                height = ys.max() - ys.min()
                xpad, ypad = 0.05 * width, 0.05 * height
                img_ax.set_xlim(xs.min() - xpad, xs.max() + xpad)
                img_ax.set_ylim(ys.max() + ypad, ys.min() - ypad)
                if not any([np.isnan(vmin), np.isnan(vmax), np.isinf(vmin), np.isinf(vmax)]):
                    colorbar_histogram(colorvals, vmin, vmax, ax=colorbar_ax,
                      bin_number=25,
                      colormap='plasma')
            colorbar_ax.set_ylabel(f"Ommatidial Diameter (N={len(self.ommatidia)})", rotation=270)
            colorbar_ax.get_yaxis().labelpad = 15
            fig.tight_layout()
            if plot_fn is not None:
                plt.savefig(plot_fn)
            if plot:
                plt.show()
            del fig


class Stack:

    def __init__(self, dirname='./', img_extension='.jpg', bw=False, layer_class=Layer, pixel_size=1, depth_size=1):
        """A stack of images for making a focus stack.

        Parameters
        ----------
        dirname : str
            Path to the directory containing the images to load.
        img_extension : str
            The file extension of the images to load.
        bw : bool
            Whether the images are greyscale.
        pixel_size : float, default=1
            Actual length of the side of a pixel.
        depth_size : float, default=1
            Actual depth interval between individual layers.

        Attributes
        ----------
        layers : list
            The list of image layers.
        layer_class : Layer, Eye
            The class to use for each layer.
        img_extension : str
            The file extension to use for importing images.
        fns : list
            The list of filenames included in the Stack.
        """
        self.dirname = dirname
        self.pixel_size = pixel_size
        self.depth_size = depth_size
        self.fns = os.listdir(self.dirname)
        self.fns = sorted([os.path.join(self.dirname, f) for f in self.fns])
        self.fns = [f for f in self.fns if f.endswith(img_extension)]
        self.layers = []
        self.bw = bw
        self.layer_class = layer_class
        self.img_extension = img_extension

    def load(self):
        """Load the individual layers."""
        print('\nloading images: ')
        self.layers = []
        for num, f in enumerate(self.fns):
            layer = self.layer_class(f, bw=(self.bw))
            layer.load()
            self.layers += [layer]
            print_progress(num + 1, len(self.fns))
        else:
            print()

    def iter_layers(self):
        """Generator yielding Layers in order."""
        for fn in self.fns:
            layer = self.layer_class(fn, bw=(self.bw))
            (yield layer)

    def load_memmaps(self):
        """Load the individual layers as memmaps to free up RAM."""
        print('\nloading images: ')
        self.layers = []
        for num, f in enumerate(self.fns):
            layer = self.layer_class(f, bw=(self.bw))
            layer.load_memmap()
            self.layers += [layer]
            print_progress(num + 1, len(self.fns))
        else:
            print()

    def load_masks(self, mask_fn=None, mask_arr=None):
        """Load the masks using either their mask file or array.

        Parameters
        ----------
        mask_fn : str, default=None
            Filename of the mask image.
        arr : np.ndarray, default=None
            2D boolean masking array. 
        """
        print('\nloading masks: ')
        for num, layer in enumerate(self.layers):
            layer.load_mask(mask_fn=mask_fn, mask_arr=mask_arr)
            print_progress(num + 1, len(self.fns))
        else:
            print()

    def get_average(self, num_samples=5):
        """Grab unmoving 'background' by averaging over some layers.

        Parameters
        ----------
        num_samples : int, default=5
            Maximum number of samples to average over.
        """
        first = self.layers[0].load_image()
        avg = np.zeros((first.shape), dtype=float)
        intervals = len(self.layers) / num_samples
        for layer in self.layers[::int(intervals)]:
            img = layer.load_image().astype(float)
            avg += img
            layer.image = None
        else:
            return (avg / num_samples).astype('uint8')

    def get_focus_stack(self, smooth_factor=0):
        """Generate a focus stack accross the image layers.
        

        Parameters
        ----------
        smooth_factor : float, default=0
            The standard deviation of the gaussian 2D filter applied to the 
            approximate heights.

        Attributes
        ----------
        images : np.ndarray
            The 4D (image number, height, width, rgb) array of images.
        gradients : np.ndarray
            The 3D (image number, height, width) array of image gradients.
        heights : np.ndarray
            The 2D (height, width) array of image indices maximizing 
            gradient values.
        stack : np.ndarray
            The 2D (height, width, rgb) array of pixels maximizing gradient
            values.
        """
        assert len(self.layers) > 0, f"Images have not been imported. Try running {self.load} or {self.load_memmaps}."
        first_image = self.layers[0].image
        self.stack = np.copy(first_image)
        self.max_gradients = np.zeros((
         first_image.shape[0], first_image.shape[1]),
          dtype=float)
        self.height_indices = np.zeros((
         first_image.shape[0], first_image.shape[1]),
          dtype=int)
        print('generating focus stack:')
        for num, layer in enumerate(self.layers):
            img = layer.image
            layer.get_gradient(smooth=smooth_factor)
            increases = np.greater_equal(layer.gradient, self.max_gradients)
            self.max_gradients[increases] = layer.gradient[increases]
            del layer.gradient
            self.height_indices[increases] = num
            print_progress(num + 1, len(self.layers))
        print()
        if smooth_factor > 0:
            self.height_indices = np.round(ndimage.filters.gaussian_filter((self.height_indices),
              sigma=smooth_factor)).astype(int)
        for num, layer in enumerate(self.layers):
            include = self.height_indices == num
            self.stack[include] = layer.image[include]
        else:
            self.heights = self.depth_size * np.copy(self.height_indices)

    def get_smooth_heights(self, sigma):
        """A 2d smoothing filter for the heights array.

        Parameters
        ----------
        sigma : int
            The standard deviation of the gaussian 2D filter applied to the 
            approximate heights.

        Returns
        -------
        new_heights : np.ndarray, shape=(height, width)
            The heights array smoothed using a fourier gaussian filter.
        """
        new_heights = self.heights.astype('float32')
        new_heights = np.fft.ifft2(ndimage.fourier_gaussian((np.fft.fft2(new_heights)),
          sigma=sigma)).real
        return new_heights


class EyeStack(Stack):

    def __init__(self, dirname, img_extension='.jpg', bw=False, pixel_size=1, depth_size=1, mask_fn='mask.jpg', mask_arr=None):
        """A special stack for handling a focus stack of fly eye images.

        Parameters
        ----------
        img_extension : str, default=".jpg"
            The image file extension used to avoid unwanted images.
        bw : bool, default=False
            Whether to treat the image as grayscale.
        pixel_size : float, default=1
            The real length of the side of one pixel in the image. Used for
            converting from pixel into real distances.
        depth_size : float, default=1
            The real distance between stack layers. 
        mask_fn : str, default="mask.jpg"
            The filename of the boolean masking image.
        mask_arr : array-like, default=None
            2D boolean masking array.         
        
        Attributes
        ----------
        eye : Eye
            The Eye object of the focus stack of cropped image layers.
        pixel_size : float, default=1
            The real length of the side of one pixel.
        depth_size : float, default=1
            The real distance between stack layers.
        eye_mask : array-like, default="mask.jpg"
            2D boolean masking array.
        ommatidia_polar : np.ndarray, default=None
            The ommatidia locations in spherical coordinates relative to 
            the best fitting sphere.
        fns : list
            The list of included filenames.
        """
        Stack.__init__(self, dirname, img_extension, bw, layer_class=Eye)
        self.eye = None
        self.pixel_size = pixel_size
        self.depth_size = depth_size
        self.eye_mask_fn = mask_fn
        self.eye_mask = mask_arr
        self.ommatidia_polar = None
        if mask_fn is not None:
            if os.path.exists(mask_fn):
                new_fns = [fn for fn in self.fns if fn != mask_fn]
                self.fns = new_fns

    def crop_eyes(self):
        """Crop each layer."""
        assert len(self.layers) > 0, f"No layers loaded yet. Try running {self.load}."
        try:
            self.load_masks(mask_fn=(self.eye_mask_fn), mask_arr=(self.eye_mask))
        except:
            breakpoint()
        new_layers = []
        for layer in self.layers:
            new_layers += [layer.crop_eye()]
        self.layers = new_layers
        self.mask = Eye(filename=(self.eye_mask_fn), arr=(self.eye_mask))
        self.mask.load_mask(mask_fn=(self.eye_mask_fn), mask_arr=(self.eye_mask))
        self.mask = self.mask.crop_eye()
        self.mask_arr = self.mask.image.astype('uint8')
        self.mask_fn = None

    def get_eye_stack(self, smooth_factor=0):
        """Generate focus stack of images and then crop out the eye.

        Parameters
        ----------
        smooth_factor : float, default=0
            The standard deviation of the gaussian 2D filter applied to the 
            approximate heights.

        Attributes
        ----------
        eye : Eye
            The Eye object of the focus stack of cropped image layers.
        """
        assert len(self.layers) > 0, f"No layers loaded. Try running {self.load} and {self.crop_eyes}."
        self.get_focus_stack(smooth_factor)
        self.eye = Eye(arr=(self.stack.astype('uint8')), mask_arr=(self.mask_arr),
          pixel_size=(self.pixel_size))

    def get_ommatidia(self, bright_peak=True, fft_smoothing=5, square_lattice=False,
                      high_pass=False, num_neighbors=3, sample_size=100, plot=False,
                      plot_fn=None, manual_edit=False):
        """Use Eye object of the eye stack image to detect ommatidia.

        Parameters
        ----------
        (see Eye.ommatidia_detecting_algorithm and self.oda_3d)
        """
        assert isinstance(self.eye, Eye), "The focus stack hasn't been processed yet. Try running " + str(self.get_eye_stack)
        self.eye.ommatidia_detecting_algorithm(
            bright_peak=bright_peak,
            fft_smoothing=fft_smoothing,
            square_lattice=square_lattice,
            high_pass=high_pass,
            num_neighbors=num_neighbors,
            sample_size=sample_size,
            plot=plot,
            plot_fn=plot_fn,
            manual_edit=manual_edit)

    def oda_3d(self, eye_stack_smoothing=0, bright_peak=True, fft_smoothing=5,
               square_lattice=False, high_pass=False, num_neighbors=3, sample_size=100,
               plot=False, plot_fn=None, use_memmaps=False, manual_edit=False):
        """Detect ommatidia using the 3D surface data.

        Parameters
        ----------
        eye_stack_smoothing : float, default=0
            Std deviation of gaussian kernal used to smooth the eye surface.
        bright_peak : bool, default=True
            Whether the ommatidia are defined by brighter (vs. darker) peaks.
        fft_smoothing : int, default=5
            The standard deviation of a 2D gaussian filter applied to the 
            reciprocal image before finding peaks.
        square_lattice : bool, default=False
            Whether this a square---rather than a hexagonal---lattice.
        high_pass : bool, default=False
            Whether to also filter frequencies below the fundamental one.
        num_neighbors : int, default=6
            The number of neighbors to check for measuring the ommatidial 
            diameter. Defaults to 6, assuming a hexagonal lattice.
        sample_size : int, default=100
            The number of ommatidia near the center to include in diameter
            estimation.
        plot : bool, default=False
            Whether to plot the eye with ommatidia and diameters superimposed.
        plot_fn : str, default=None
            Filename to save the plotted eye with superimposed ommatidia and 
            their diameters.
        use_memmaps : bool, default=False
            Whether to use memory maps instead of loading the images to RAM.
        manual_edit : bool, default=False
            Whether to use a GUI to edit the results of the ODA.


        Attributes
        ----------
        sphere : SphereFit
            An OLS-fitted sphere to the 3D ommatidia coordinates. 
            Transforms the points into polar coordinates relative to 
            the fitted sphere.
        fov_hull : float
            The field of view of the convex hull of the ommatidia in
            polar coordinates.
        fov_long : float
            The longest angle of view using the long diameter of the
            ellipse fitted to ommatidia in polar coordinates.
        fov_short : float, steradians
            The shortest angle of view using the short diameter of 
            the ellipse fitted to ommatidia in polar coordinates.
        surface_area : float, steradians
            The surface area of the sphere region given fov_hull and
            sphere.radius.
        io_angles : np.ndarray, rads
            The approximate inter-ommatidial angles per ommatidium 
            using eye.ommatidial_diameters and eye radius in rads.
        io_angle : float, rad
            The average approximate inter-ommatidial angle using 
            eye.ommatidial_diameter / self.sphere.radius
        io_angle_fft : float, rad
            The average approximate inter-ommatidial angle using
            eye.ommatidial_diameter_fft / self.sphere.radius
        """
        if use_memmaps:
            self.load_memmaps()
        else:
            self.load()
        if 'mask' not in dir(self):
            # make an eye object from the focus stack
            self.get_focus_stack()
            eye = Eye(arr=self.stack)
            # color key the stack and use this as the mask
            eye.color_key()
            self.eye_mask = eye.mask
            self.eye_mask_fn = None
        self.crop_eyes()
        self.get_eye_stack(smooth_factor=eye_stack_smoothing)
        self.get_ommatidia(
            bright_peak=bright_peak, fft_smoothing=fft_smoothing,
            square_lattice=square_lattice,
            high_pass=high_pass,
            num_neighbors=num_neighbors,
            sample_size=sample_size,
            plot=plot,
            plot_fn=plot_fn,
            manual_edit=manual_edit)
        ys, xs = self.eye.ommatidial_inds.T.astype(int)
        zs = self.heights[ys, xs]
        ys, xs = self.eye.ommatidia.T
        new_ommatidia = np.array([ys, xs, zs]).T
        self.eye.ommatidia = new_ommatidia
        self.eye.measure_ommatidia()
        self.sphere = SphereFit(self.eye.ommatidia)
        hull = spatial.ConvexHull(self.sphere.polar[:, :2])
        hull_polar = self.sphere.polar[hull.vertices, :2]
        self.fov_hull = hull.area
        plt.scatter(hull_polar[:, 0], hull_polar[:, 1])
        polar_ellipse = LSqEllipse()
        polar_ellipse.fit(hull_polar.T)
        (theta_center, phi_center), width, height, ang = polar_ellipse.parameters()
        self.fov_short = 2 * width
        self.fov_long = 2 * height
        self.surface_area = self.fov_hull * self.sphere.radius ** 2
        self.io_angles = self.eye.ommatidial_diameters / self.sphere.radius
        self.io_angle = self.eye.ommatidial_diameter / self.sphere.radius
        self.io_angle_fft = self.eye.ommatidial_diameter_fft / self.sphere.radius
        area = np.round(self.surface_area, 4)
        fov = np.round(self.fov_hull, 4)
        fov_long, fov_short = np.round(self.fov_long, 4), np.round(self.fov_short, 4)
        count = len(self.eye.ommatidia)
        sample_diameter = np.round(self.eye.ommatidial_diameter, 4)
        sample_std = np.round(self.eye.ommatidial_diameter_SD, 4)
        diameter_fft = np.round(self.eye.ommatidial_diameter_fft, 4)
        io_angle = np.round(self.io_angle * 180 / np.pi, 4)
        io_angle_std = np.round(self.io_angles.std() * 180 / np.pi, 4)
        io_angle_fft = np.round(self.io_angle_fft * 180 / np.pi, 4)
        print(f"3D results:\nEye:\tSurface Area={area}\tFOV={fov}\tFOV_l={fov_long}\tFOV_s={fov_short}\nOmmatidia:\tmean={sample_diameter}\tstd={sample_std}\tfft={diameter_fft}\nIO angles(deg):\tmean={io_angle}\tstd={io_angle_std}\tfft={io_angle_fft}\n")
        print()


class CTStack(Stack):

    def __init__(self, database_fn='_compound_eye_data.h5', **kwargs):
        """A special stack for handling a CT stack of a compound eye.

        Parameters
        ----------
        database_fn : str, default="_compoint_eye_data.h5"
            The filename of the H5 database with loaded values.
        dirname : str
            Path to the directory containing the images to load.
        img_extension : str
            The file extension of the images to load.
        bw : bool
            Whether the images are greyscale.
        pixel_size : float, default=1
            Actual length of the side of a pixel.
        depth_size : float, default=1
            Actual depth interval between individual layers.

        Attributes
        ----------
        layers : list
            The list of image layers.
        layer_class : Layer, Eye
            The class to use for each layer.
        img_extension : str
            The file extension to use for importing images.
        fns : list
            The list of filenames included in the Stack.

        Methods
        -------
        __del__
        load_database
            Initialize and load the H5 database.
        close_database
            Delete the H5PY database.
        save_database
            Save the H5PY database by closing and reopening it.
        prefilter
            Filter the layers and then save in a new folder.
        import_stack
            Filter the images including values between low and high.
        get_cross_sections
            Approximate surface splitting the inner and outer sections.
        find_ommatidial_clusters
            2D running window applying ODA to spherical projections.
        measure_ommatidia
            Take measurements of ommatidia using the ommatidial clusters.
        measure_interommatidia
            Use the anatomical axes to quickly measure interommatidial angles.
        plot_raw_data, plot_cross_section, plot_ommatidial_clusters, 
        plot_ommatidial_data, plot_ommatidial_data_3d, plot_interommatidial_data,
        plot_interommatidial_data_3d, plot_data_3d 
            Convenient functions for plotting the corresponding data.
        ommatidia_detecting_algorithm
            Apply the 3D ommatidia detecting algorithm (ODA-3D).
        stats_summary
            Calculate important statistics for whatever data is available.
        """
        (Stack.__init__)(self, **kwargs)
        self.oda = self.ommatidia_detecting_algorithm
        self.database_fn = os.path.join(self.dirname, database_fn)
        self.load_database()

    def __del__(self):
        # when deleting the object, make sure to close the database
        self.close_database()

    def load_database(self, mode='r+'):
        """Initialize and load the H5 database.

        Parameters
        ----------
        mode : str, default='r+'
            The access privileges of the database.
        """
        if not os.path.exists(self.database_fn):
            new_database = h5py.File(self.database_fn, 'w')
            new_database.close()
        try:
            self.database = h5py.File(self.database_fn, mode)
        except:
            os.remove(self.database_fn)
            new_database = h5py.File(self.database_fn, 'w')
            new_database.close()
            self.database = h5py.File(self.database_fn, mode)
        if len(self.fns) > 0:
            first_layer = Layer(self.fns[0])
            first_layer.load()
            self.dtype = first_layer.image.dtype
        for key in self.database.keys():
            setattr(self, key, self.database[key])
        files_to_load = [
            os.path.join(self.dirname, 'ommatidial_data.pkl'),
            os.path.join(self.dirname, 'interommatidial_data.pkl')]
        for var, fn in zip(['ommatidial_data', 'interommatidial_data'],
                           files_to_load):
            csv_fn = fn.replace(".pkl", ".csv")
            if var in dir(self):
                delattr(self, var)
            if os.path.exists(fn):
                try:
                    setattr(self, var, pd.read_pickle(fn))
                except:
                    if os.path.exists(csv_fn):
                        setattr(self, var, pd.read_csv(csv_fn))
                    else:
                        print(f"failed to load {var}")

    def close_database(self):
        """Delete the H5PY database."""
        if 'database' in dir(self):
            try:
                self.database.close()
            except:
                # make a temporary copy of this database
                temp_fn = "temp_database.h5"
                with h5py.File(temp_fn, 'w') as database_copy:
                    for key in self.database.keys():
                        dataset = self.database[key][:]
                        database_copy.create_dataset(
                            key, data=dataset, dtype=dataset.dtype)
                # now delete the old database
                del self.database
                os.remove(self.database_fn)
                os.rename(temp_fn, self.database_fn)

    def save_database(self):
        """Save the H5PY database by closing and reopening it."""
        self.close_database()
        self.load_database()

    def prefilter(self, low=0, high=None, folder='_prefiltered_stack', gui=False):
        """Filter the layers and then save in a new folder.

        Parameters
        ----------
        low : int, default=0
            The minimum value for an inclusive filter.
        high : int, default=None
            The maximum value for an inclusing filter, defaulting to 
            the maximum.
        folder : str, default="_prefiltered_stack"
            The directory to store the prefiltered image.
        gui : bool, default=False
            Whether to use the GUI for selecting the low and high thresholds.
        """
        first_layer = Layer(self.fns[0])
        dirname, basename = os.path.split(first_layer.filename)
        # use GUI if selected
        if gui:
            self.gui = StackFilter(self.fns)
            low, high = self.gui.get_limits()
        if high is None:
            first_layer.load()
            dtype = first_layer.image.dtype
            high = np.iinfo(dtype).max
        if not os.path.isdir(os.path.join(dirname, folder)):
            os.mkdir(os.path.join(dirname, folder))
        print("Pre-filtering layer by layer:")
        for num, layer in enumerate(self.iter_layers()):
            layer.load()
            include = (layer.image >= low) * (layer.image <= high)
            new_img = np.zeros(layer.image.shape, layer.image.dtype)
            new_img[include] = layer.image[include]
            basename = os.path.basename(layer.filename)
            new_fn = os.path.join(dirname, folder, basename)
            save_image(new_fn, new_img)
            print_progress(num + 1, len(self.fns))
        print()

    def import_stack(self, low=0, high=None):
        """Filter the images including values between low and high.

        Parameters
        ----------
        low : int, default=0
            The minimum value for an inclusive filter.
        high : int, default=None
            The maximum value for an inclusing filter, defaulting to 
            the maximum.
        """
        if 'points' in dir(self):
            del self.points
            if 'points' in self.database:
                del self.database['points']
        self.points = self.database.create_dataset(
            'points', data=(np.zeros((1000, 3))), dtype=float,
            chunks=(1000, 3), maxshape=(None, 3))
        self.points_subset = self.database.create_dataset(
            'points_subset', data=(np.zeros((0, 3))), dtype=float,
            chunks=(1000, 3), maxshape=(None, 3))
        first_layer = Layer(self.fns[0])
        dirname, basename = os.path.split(first_layer.filename)
        if high is None:
            first_layer.load()
            dtype = first_layer.image.dtype
            high = np.iinfo(dtype).max
        if self.points.shape[0] > 0:
            self.points.resize(0, axis=0)
        print("Importing stack layer by layer:")
        for num, layer in enumerate(self.iter_layers()):
            layer.load()
            include = (layer.image >= low) * (layer.image <= high)
            if np.any(include):
                x, y = np.where(include)
                pts = np.array([
                 np.repeat(
                     float(num) * self.depth_size, len(x)),
                    self.pixel_size * x.astype(float),
                    self.pixel_size * y.astype(float)]).T
                self.points.resize((
                    self.points.shape[0] + len(x), 3))
                self.points[-len(x):] = pts
                num_points = len(x)
                sample = math.ceil(0.01 * num_points)
                inds = np.random.choice((np.arange(num_points)),
                  size=sample, replace=False)
                self.points_subset.resize((
                    self.points_subset.shape[0] + sample, 3))
                self.points_subset[-sample:] = pts[inds]
            print_progress(num, len(self.fns))
        print()

    def get_cross_sections(self, thickness=1.0, chunk_process=True):
        """Approximate surface splitting the inner and outer sections.

        Uses 2D spline interpolation, modelling point radial distance 
        as a function of its polar coordinate.

        Parameters
        ----------
        thickness : float, default=.3
            Proportion of the residuals to include in the cross section 
            used for the ODA.
        chunk_process : bool, default=False
            Whether to process  polar coordinates in chunks or all at 
            once, relying on RAM.

        Attributes
        ----------
        theta, phi, radii : array-like, dtype=float, shape=(N, 1)
            The azimuth, elevation, and radial distance of the loaded 
            coordinates centered around the center of a fitted sphere.
        residual : array-like, dtype=float, shape=(N, 1)
            Residual distance of points from a fitted interpolated surface.
        """
        assert self.points.shape[0] > 0, ("No points have been loaded. Try running"+
                                          f" {self.import_stack} first.")
        ind_range = range(len(self.points))
        num_samples = min(len(self.points), 1000000)
        inds = np.random.choice(ind_range, size=num_samples, replace=False)
        inds.sort()
        chunksize = 100
        num_chunks = int(np.ceil(len(inds) / chunksize))
        subset = []
        new_step = 100
        for chunk in self.points.iter_chunks():
            chunk_a, chunk_b = chunk
            new_chunk_a = slice(chunk_a.start, chunk_a.stop, new_step)
            new_chunk = (new_chunk_a, chunk_b)
            subset += [self.points[new_chunk]]
        subset = np.concatenate(subset)
        sphere = SphereFit(subset)
        center = sphere.center
        self.center = center
        pts = self.points[:]
        pts = pts - self.center[np.newaxis]
        pts = rotate(pts, (sphere.rot_ang1), axis=0).T
        pts = rotate(pts, (sphere.rot_ang2), axis=2).T
        self.points[:] = pts
        subset -= center[np.newaxis, :]
        # pts.surface_predict(image_size=10000)
        self.shell = pts
        if chunk_process:
            vars_to_check = [
             'theta', 'phi', 'radii', 'residual']
            for var in vars_to_check:
                if var in dir(self):
                    delattr(self, var)
                if var in self.database.keys():
                    if len(self.database[var]) == len(self.points):
                        self.database[var][:] = 0
                    else:
                        del self.database[var]
                if var not in self.database.keys():
                    self.database.create_dataset(var, (len(self.points)),
                                                 dtype=float, chunks=1000)
            self.save_database()
            # self.load_database()
            print("Finding the approximate cross-section in chunks:")
            for chunk_num, (subset_inds, theta_inds, phi_inds,
                            radii_inds) in enumerate(zip(
                                self.points.iter_chunks(),
                                self.theta.iter_chunks(),
                                self.phi.iter_chunks(),
                                self.radii.iter_chunks())):
                subset = self.points[subset_inds]
                polar = rectangular_to_spherical(subset)
                for storage, vals, inds in zip([
                        self.theta, self.phi, self.radii], polar.T, [
                            theta_inds, phi_inds, radii_inds]):
                    storage[inds] = vals
                print_progress(
                    chunk_num, self.points.shape[0] / self.points.chunks[0])
            polar = np.array([self.theta[:], self.phi[:], self.radii[:]]).T
            arr = spherical_to_rectangular(polar)
            pts = Points(arr=arr, polar=polar, sphere_fit=False, rotate_com=False,
                         spherical_conversion=False)
            pts.surface_predict(xvals=(self.theta[:]), yvals=(self.phi[:]))
            predicted_radii = pts.surface
            # plt.scatter(self.theta[:], self.phi[:], c=predicted_radii)
            # plt.gca().set_aspect('equal')
            # plt.show()
            self.residual[:] = self.radii[:] - predicted_radii
        else:
            polar = rectangular_to_spherical(self.points[:])
            theta, phi, radii = polar.T
            pts = Points(arr=pts, polar=polar, sphere_fit=False, rotate_com=False,
                         spherical_conversion=False)
            pts.surface_predict(xvals=(theta), yvals=(phi))
            predicted_radii = pts.surface
            residuals = radii - predicted_radii
            vars_to_check = [('theta', theta), ('phi', phi),
                             ('radii', radii), ('residual', residuals)]
            for (name, var) in vars_to_check:
                if name in dir(self):
                    delattr(self, name)
                if name in self.database.keys():
                    del self.database[name]
                dataset = self.database.create_dataset(name, data=var, dtype=float)
                setattr(self, name, dataset)

    def find_ommatidial_clusters(self, polar_clustering=True, window_length=np.pi / 6,
                                 window_pad=np.pi / 20, image_size=10000, mask_blur_std=2,
                                 square_lattice=False, test=False, regular=True,
                                 manual_edit=False, thickness=.5):
        """2D running window applying ODA to spherical projections.

        
        Parameters
        ----------
        polar_clustering : bool, default=True
            Whether to use polar coordinates for clustering (as 
            opposed to the 3D rectangular coordinates).
        window_length : float, default=pi/4
            The angle of view of the rolling square window.
        window_pad : float, default=pi/20
            The padding of overlap used to avoide border issues.
        image_size : float, default=1e6
            The number of pixels to use in rasterizing the rolling
            window.
        mask_blur_std : float, default=2
            The standard deviation of the gaussian blur used for
            smoothing the mask for the ODA.
        square_lattice : bool, default=False
            Wether the ommatidial lattice is square vs. hexagonal.
        test : bool, default=False
            Whether to run parts of the script for troubleshooting.
        regular : bool, default=True
            Whether to assume the ommatidial lattice is regular.
        manual_edit : bool, default=False
            Whether to allow the user to manually edit the ommatidial 
            coordinates per segment.
        thickness : float, default=.5
            The proportion of the residuals used for generating the 
            eye raster for running the ODA.

        Attributes
        ----------
        include : np.ndarray, dtype=bool
        

        """
        # assume the the shell was loaded
        assert 'theta' in self.database.keys(), (
            "No polar coordinates found. "
            f"Try running {self.get_cross_sections} "
            f"first or running {self.ommatidia_detecting_algorithm}")
        # get the elevation and azimuth ranges
        theta_min, theta_max = np.percentile(self.theta[:], [0, 100])
        phi_min, phi_max = np.percentile(self.phi[:], [0, 100])
        # determine number of segments based on window length, and then adjust
        # window_length to evenly devide the area
        theta_range = theta_max - theta_min
        theta_segments = math.ceil(theta_range / window_length)
        window_length_theta = theta_range / theta_segments
        phi_range = phi_max - phi_min
        phi_segments = math.ceil(phi_range / window_length)
        window_length_phi = phi_range / phi_segments
        # make datasets for storing the included coordinates and their labels
        # vars_to_check = ['include', 'labels']
        # dtypes = [bool, int]
        # for var, dtype in zip(vars_to_check, dtypes):
        #     if var in dir(self):
        #         delattr(self, var)
        #     if var in self.database.keys():
        #         if len(self.database[var]) == len(self.points):
        #             try:
        #                 self.database[var][:] = 0
        #             except:
        #                 del self.database[var]
        #         else:
        #             del self.database[var]
        #     if var not in self.database.keys():
        #         self.database.create_dataset(var, (len(self.points)),
        #                                      dtype=dtype, chunks=1000)
        # self.save_database()
        # instead of datasets, use a simple numpy array for the labels and inclusion array
        self.include = np.zeros(self.points.shape[0], dtype=bool)
        self.labels = np.zeros(self.points.shape[0], dtype=int)
        # for var, dtype in zip(vars_to_check, dtypes):
        #     if var in self.database.keys():
        #         del self.database[var]
        #         if var in dir(self):
        #             delattr(self, var)
        #     if var not in self.database.keys():
        #         self.database.create_dataset(var, (len(self.points)),
        #                                      dtype=dtype, chunks=1000)
        # self.save_database()
        theta_low = theta_min
        phi_low = phi_min
        # use center points to reorient polar coordinates and avoid distortion
        theta_center = np.mean([theta_min, theta_max])
        phi_center = np.mean([phi_min, phi_max])
        # iterate through the windows +/- padding
        max_val = 0
        segment_number = 1
        # low, high = np.percentile(self.residual[:], [25, 75])
        thickness_padding = 100 * ((1 - thickness) / 2)
        lower_b, upper_b = thickness_padding, 100 - thickness_padding
        low, high = np.percentile(self.residual[:], [lower_b, upper_b])
        in_cross_section = (self.residual[:] > low) * (self.residual[:] < high)
        print('Processing the polar coordinates in segments:')
        while theta_low <= theta_max:
            # get relevant theta values
            theta_high = theta_low + window_length_theta
            theta_center = np.mean([theta_low, theta_high])
            while phi_low <= phi_max:
                print(f"Segment #{segment_number}:")
                # and phi values
                phi_high = phi_low + window_length_phi
                phi_center = np.mean([phi_low, phi_high])
                # store inclusion criteria
                self.include[:] = True
                # elevation filter
                self.include[:] *= self.theta > theta_low - window_pad
                self.include[:] *= self.theta <= theta_high + window_pad
                # azimuth filter
                self.include[:] *= self.phi > phi_low - window_pad
                self.include[:] *= self.phi <= phi_high + window_pad
                # how many of these are actually included in the window?
                in_window = self.theta[:] > theta_low
                in_window *= self.theta[:] <= theta_high
                in_window *= self.phi[:] > phi_low
                in_window *= self.phi[:] <= phi_high
                # try clustering only if more than 30 points were included
                if np.sum(in_window) > 30:
                    # get subset of points and their corresponding indices
                    subset = []
                    in_shell = []
                    included_inds = []
                    for num, chunk in enumerate(self.points.iter_chunks()):
                        chunk = chunk[0]
                        sub_include = self.include[chunk]
                        if np.any(sub_include):
                            sub_points = self.points[chunk]
                            subset += [sub_points[sub_include]]
                            sub_cross_section = in_cross_section[chunk]
                            in_shell += [sub_cross_section[sub_include]]
                            inds = np.arange(chunk.start, chunk.stop, chunk.step)
                    #         included_inds += [inds[sub_include]]
                    # included_inds = np.concatenate(included_inds)
                    included_inds = np.arange(len(self.include))[self.include]
                    subset_original = np.concatenate(subset)
                    subset = np.copy(subset_original)
                    in_shell = np.concatenate(in_shell)
                    if sum(in_shell) > 10:
                        # rotate points so that center of mass is ideal for spherical coordinates
                        com = subset.mean(0)
                        com_polar = rectangular_to_spherical(com[np.newaxis])
                        theta_displacement, phi_displacement, _ = com_polar[0]
                        subset = rotate(subset, phi_center, axis=2).T
                        subset = rotate(subset, (theta_center - np.pi / 2), axis=1).T
                        polar = rectangular_to_spherical(subset)
                        # get the now centered polar coordinates
                        segment = Points(subset, sphere_fit=False, rotate_com=False,
                                         spherical_conversion=False, polar=polar)
                        # segment.surface_projection()
                        sub_segment = Points((subset[in_shell]), sphere_fit=False, rotate_com=False,
                                             spherical_conversion=False, polar=polar[in_shell])
                        # sub_segment.surface_projection(image_size=1e6)
                        # find the shortest distances to figure out the necessary pixel size
                        # for our raster image
                        dists_tree = spatial.KDTree(sub_segment.polar[:, :2])
                        dists, inds = dists_tree.query((sub_segment.polar[:, :2]), k=2)
                        min_dist = 2 * np.mean(dists[:, 1])
                        raster, (theta_vals, phi_vals) = sub_segment.rasterize(
                            image_size=image_size, pixel_length=min_dist)
                        pixel_size = phi_vals[1] - phi_vals[0]
                        # generate a boolean mask by smoothing the thresholded raster image
                        mask = raster > 0
                        mask = ndimage.gaussian_filter(mask.astype(float), 2)
                        mask /= mask.max()
                        thresh = 0.1
                        mask = mask > thresh
                        mask = 255 * mask.astype(int)
                        raster = 255 * (raster / raster.max())
                        raster = raster.astype('uint8')
                        # use the ODA 2D to find ommatidial centers
                        eye = Eye(arr=raster, pixel_size=pixel_size, mask_arr=mask,
                                  mask_fn=None)
                        eye.oda(plot=False, square_lattice=False, bright_peak=True,
                                regular=regular, manual_edit=manual_edit)
                        centers = eye.ommatidia
                        # TODO: fix bug resulting in one HUGE segment of points. this is compressing
                        # the raster image or warping it due to the curvature and affecting the ODA.
                        # Basically, make sure that the segments are evenly partitioned
                        # if we found a reasonable number of ommatidia,
                        if len(centers) < len(subset) and len(centers) > 0:
                            # uncenter their center coordinates
                            centers += [theta_vals.min(), phi_vals.min()]
                            segment.surface_predict(xvals=(centers[:, 0]),
                                                    yvals=(centers[:, 1]))
                            model_radii = segment.surface
                            # use model radii
                            centers = np.array([
                                centers[:, 0], centers[:, 1], model_radii]).T
                            if polar_clustering:
                                # use the 2d projection
                                no_nans = np.any((np.isnan(centers)), axis=1) == False
                                centers = centers[no_nans]
                                # use KDTree to to find points nearest the centers
                                dist_tree = spatial.KDTree(centers[:, :2])
                                dists, lbls = dist_tree.query(polar[:, :2])
                                lbls += 1
                            else:
                                # use 3d coordinates
                                no_nans = np.isnan(centers[:, -1]) == False
                                centers = centers[no_nans]
                                centers_rect = spherical_to_rectangular(centers)
                                dist_tree_centers = spatial.KDTree(centers_rect)
                                dist_tree_pts = spatial.KDTree(subset)
                                dists, lbls = dist_tree_centers.query(subset)
                                lbls += 1  # avoid 0, which is the default group
                                # use 2 times the ommatidial diameter to include
                                # neighboring clusters
                                ommatidial_diam_angle = eye.ommatidial_diameter.mean()
                                ommatidial_diam = ommatidial_diam_angle * self.radii[:].mean()
                                radius = 2 * ommatidial_diam
                                # measure how many pts have nearest neighbors in a different cluster
                                # find nearest neighbors
                                nearest_dists, nearest_neighbors = dist_tree_pts.query(subset, k=2)
                                nearest_neighbors = nearest_neighbors[:, 1]
                                # get labels of those neighbors
                                neighbor_lbls = lbls[nearest_neighbors]
                                # compare the two lists of labels and aggregate based on label
                                different_neighbors = lbls != neighbor_lbls
                                broken_pts = []
                                for lbl in range(len(centers_rect)):
                                    subset_inds = lbls == lbl
                                    subset_broken = different_neighbors[subset_inds]
                                    broken_pts += [subset_broken.mean()]
                                broken_pts = np.array(broken_pts)
                                for lbl, (center, broken_prop) in enumerate(zip(
                                        centers_rect, broken_pts)):
                                    if broken_prop > .1:
                                        # 1. find nearby pts and cluster centers, recentering to the
                                        # current center
                                        nearby_centers = np.array(
                                            dist_tree_centers.query_ball_point(
                                                center, r=radius))
                                        # wrong! I need to find the 
                                        nearby_inds = np.where(np.in1d(lbls-1, nearby_centers))[0]
                                        pts_nearby = subset[nearby_inds]
                                        lbls_nearby = lbls[nearby_inds]
                                        # center pts
                                        pts_center = pts_nearby.mean(0)
                                        pts_nearby -= pts_center
                                        # center centers
                                        centers_nearby = centers_rect[nearby_centers]
                                        centers_nearby -= pts_center
                                        # 2. apply Gaussian Mixture clustering to find individual clusters
                                        # test: can we orient the clusters using SVD, since the
                                        # SVD doesn't really work on more skewed ommaitidia, but
                                        # can work as a starting point for the rotation fitting
                                        # variability should be lowest along the longitudinal axis
                                        uu, ss, vv = np.linalg.svd(pts_nearby)
                                        uu_c, ss_c, vv_c = np.linalg.svd(centers_nearby)
                                        # rotate using the new basis vectors
                                        pts_nearby_rotated = np.dot(pts_nearby, vv.T)
                                        centers_nearby_rotated = np.dot(centers_nearby, vv.T)
                                        # test: plot the projected points to see if this worked
                                            # x, y, z = pts_nearby_rotated.T
                                            # fig = plt.figure()
                                            # # ax = fig.add_subplot(111, projection='3d')
                                            # # ax.scatter(x, y, z)
                                            # ax = fig.add_subplot(121)
                                            # ax.scatter(x, y, alpha=.25)
                                            # ax.set_aspect('equal')
                                            # # plot the old orientation
                                            # ax = fig.add_subplot(122)
                                            # ax.scatter(pts_nearby[:, 0], pts_nearby[:, 1], alpha=.25)
                                            # # plot in 3d
                                            # fig = plt.figure()
                                            # ax = fig.add_subplot(111, projection='3d',
                                            #                      proj_type='ortho')
                                            # ax.scatter(x, y, z)
                                            # plt.show()
                                        # test: use RotateClusterer with +/- 60 degrees from the
                                        # rotated points. SUCCESS!!!
                                        clusterer = RotateClusterer(pts_nearby_rotated,
                                                                    centers_nearby_rotated)
                                        new_lbls = clusterer.lbls
                                        new_lbls_set = np.unique(new_lbls)
                                        # 3. get all points with the lbl closest to 0
                                        # centers_new = clusterer.means_
                                        centers_new = []
                                        for new_lbl in set(new_lbls):
                                            sub_pts = pts_nearby[new_lbls == new_lbl]
                                            centers_new += [sub_pts.mean(0)]
                                        centers_new = np.array(centers_new)
                                        center_dists = np.linalg.norm(centers_new, axis=-1)
                                        new_lbl_nearest = np.argmin(center_dists)  # closest new center
                                        # use linear sum sorting to find new label closest to old label
                                        diffs = centers_new[np.newaxis] - centers_nearby[:, np.newaxis]
                                        dists = np.linalg.norm(diffs, axis=-1)
                                        row_inds, col_inds = optimize.linear_sum_assignment(
                                            dists)
                                        center_ind = np.where(nearby_centers == lbl)[0][0]
                                        # test: an index error came up. i may have switched rows with
                                        # columns. if I reach this breakpoint, there's a bigger
                                        # problem
                                        if center_ind in row_inds:
                                            new_lbl_nearest = new_lbls_set[row_inds == center_ind][0]
                                            # new_lbl_nearest = new_lbls[center_nearest]
                                            pts_nearest = pts_nearby[new_lbls == new_lbl_nearest]
                                            pts_nearest_uncentered = pts_nearest + pts_center
                                            # 4. get indices of those points in self.points
                                            # measure number of broken elements
                                            dist_tree_pts_new = spatial.KDTree(pts_nearby)
                                            dists, inds = dist_tree_pts_new.query(
                                                pts_nearest, k=2)
                                            lbl_pairs = new_lbls[inds].T
                                            broken_pts = lbl_pairs[0] != lbl_pairs[1]
                                            broken_prop_new = broken_pts.mean()
                                            # only accept new labels if it produces less broken
                                            # elements
                                            if broken_prop_new < .05:
                                                inds_original = lbls == lbl
                                                inds_new = nearby_inds[new_lbls == new_lbl_nearest]
                                                # 5. replace the lbl value associated with those
                                                # points
                                                # with the current label
                                                lbls[inds_original] = -1
                                                lbls[inds_new] = lbl
                                        # test: plot with the original
                                        if test:
                                            fig = plt.figure()
                                            # plot the subset colorcoded by old labels, highlight main
                                            # cluster
                                            ax = fig.add_subplot(121, projection='3d')
                                            ax.scatter((pts_nearby[:, 0]), (pts_nearby[:, 1]),
                                                       (pts_nearby[:, 2]), c=lbls_nearby,
                                                       marker='.', cmap='tab20')
                                            old_cluster = pts_nearby[lbls_nearby == lbl]
                                            ax.scatter(old_cluster[:, 0], old_cluster[:, 1],
                                                       old_cluster[:, 2], marker='o', color='k')
                                            ax.scatter(centers_nearby[:, 0], centers_nearby[:, 1],
                                                       centers_nearby[:, 2], marker='o', color='r')
                                            ax.set_title(f"{np.round(100 * broken_prop, 2)}% outliers")
                                            # plot the whole subset highlighting the new main cluster
                                            ax = fig.add_subplot(122, projection='3d')
                                            ax.scatter((pts_nearby[:, 0]), (pts_nearby[:, 1]),
                                                       (pts_nearby[:, 2]), c=new_lbls,
                                                       marker='.', cmap='tab20')
                                            ax.scatter(pts_nearest[:, 0], pts_nearest[:, 1],
                                                       pts_nearest[:, 2], marker='o', color='k')
                                            ax.scatter(centers_new[:, 0], centers_new[:, 1],
                                                       centers_new[:, 2], marker='o', color='r')
                                            ax.set_title(f"{np.round(100 * broken_prop_new, 2)}% outliers")
                                            plt.show()
                                    print_progress(lbl, len(centers_rect))
                        # store the labels + previous maximum to avoid duplicates
                        # todo: use arbitrary labels and then replace with unique ones later
                        # get the centers for each lbl in lbl_set
                        centers_original = []
                        lbls_set = np.unique(lbls)
                        for lbl in lbls_set:
                            inds = lbls == lbl
                            cluster = subset_original[inds]
                            centers_original += [cluster.mean(0)]
                        centers_original = np.array(centers_original)
                        centers_original_polar = rectangular_to_spherical(
                            centers_original).T
                        theta, phi, radii = centers_original_polar
                        # check for points within the window (and not in padding)
                        in_window = theta > theta_low
                        in_window *= theta <= theta_high
                        in_window *= phi > phi_low
                        in_window *= phi <= phi_high
                        if np.any(in_window):
                            # store all cluster labels with centers within the window
                            lbls_set_in_window = lbls_set[in_window]
                            lbls_in_window = np.in1d(lbls, lbls_set_in_window)
                            # find the indices inside window
                            included_inds = included_inds[lbls_in_window]
                            # find those labels
                            inds_in_window = np.in1d(lbls, lbls_set_in_window)
                            lbls_in_window = lbls[inds_in_window]
                            # convert to unique, ordered range of values
                            lbls_old_set, lbls_new = np.unique(
                                lbls_in_window, return_inverse=True)
                            lbls_new += max_val + 1
                            max_val = lbls_new.max()
                            # replace labels within window 
                            num_blocks = math.ceil(included_inds.shape[0] / 1000)
                            # store labels in chunks, ignoring lbls set to 0
                            order = np.argsort(included_inds)
                            self.labels[included_inds[order]] = lbls_new[order]
                            # for num, (ind_vals, sub_lbls) in enumerate(zip(
                            #         np.array_split(included_inds[order], num_blocks),
                            #         np.array_split(lbls_new[order], num_blocks))):
                            #     self.labels[ind_vals] = sub_lbls
                        segment_number += 1
                        print()
                # update the phi lower bound
                phi_low += window_length_phi
            # update the theta lower bound and reset phi lower bound
            theta_low += window_length_theta
            phi_low = phi_min
        # store the labels in a dataset
        if 'labels' in self.database.keys():
            try:
                self.database['labels'][:] = self.labels # try just updating the dataset
            except:
                del self.database['labels'] # instead, deleta and make a new one
                self.database.create_dataset('labels', data=self.labels, dtype=int, chunks=1000)
        else:
            self.database.create_dataset('labels', data=self.labels, dtype=int, chunks=1000)
        self.save_database()
        # group the coordinate data by their cluster labels
        centers = []
        xs, ys, zs, thetas, phis, radii = ([], [], [], [], [], [])
        size = []
        labels = self.labels[:]
        # label_set = np.array(list(set(self.labels[:])))
        label_set = np.unique(self.labels[:])
        # iterate through the set of labels
        print('Grouping the data by cluster labels:')
        for num, label in enumerate(label_set):
            inds = np.where(labels == label)[0]
            num_blocks = math.ceil(inds.shape[0] / 1000)
            sub_pts, sub_thetas, sub_phis, sub_radii = [], [], [], []
            for ind_chunk in np.array_split(inds, num_blocks):
                # store the centroid of the cluster
                sub_pts += [self.points[ind_chunk]]
                sub_thetas += [self.theta[ind_chunk]]
                sub_phis += [self.phi[ind_chunk]]
                sub_radii += [self.radii[ind_chunk]]
            sub_pts, sub_thetas = np.concatenate(sub_pts), np.concatenate(sub_thetas)
            sub_phis, sub_radii = np.concatenate(sub_phis), np.concatenate(sub_radii)
            center = sub_pts.mean(0)
            # store mean x, y, and z and the polar coordinates
            x, y, z = center
            for storage, vals in zip(
                    [thetas, phis, radii, centers, xs, ys, zs, size],
                    [sub_thetas.mean(), sub_phis.mean(), sub_radii.mean(),
                     center, x, y, z, len(inds)]):
                storage += [vals]
            print_progress(num, len(label_set))
        # convert to numpy arrays to save with pandas
        centers = np.array(centers)
        xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)
        thetas, phis, radii = np.array(thetas), np.array(phis), np.array(radii)
        size = np.array(size)
        # store measurements in a dictionary
        data_to_save = dict()
        for var, arr in zip(
                ['label', 'x', 'y', 'z', 'theta', 'phi', 'radius', 'size'],
                [label_set, xs, ys, zs, thetas, phis, radii, size]):
            data_to_save[var] = arr
        # make into a pandas dataframe
        ommatidial_data = pd.DataFrame(data_to_save)
        ommatidial_data = ommatidial_data.loc[1:]
        # lbls = ommatidial_data.label.values
        # ommatidial_data = ommatidial_data[lbls > 0]
        # save as csv
        csv_filename = os.path.join(self.dirname, 'ommatidial_data.csv')
        ommatidial_data.to_csv(csv_filename, index=False)
        ommatidial_data.to_pickle(csv_filename.replace('.csv', '.pkl'))


    def measure_ommatidia(self, square_lattice=False, test=False):
        """Take measurements of ommatidia using the ommatidial clusters.

        
        Parameters
        ----------
        square_lattice : bool
            Whether the ommatidial lattice is square vs. hexagonal.

        Attributes
        ----------
        ommatidial_data : pd.DataFrame
            The dataframe containing data on the ommatidial clusters including
            each position in rectangular (x, y, and z) and polar (theta, phi, 
            radius) coordinates. 
        """
        assert 'ommatidial_data' in dir(self), (
            f"No clustered data found. Try running {selfind_ommatidial_clusters}")
        labels = self.labels[:]
        label_set = self.ommatidial_data.label.values
        centers = self.ommatidial_data[['x', 'y', 'z']].values
        # use distance tree of cluster centers to find nearest neighbors
        centers_tree = spatial.KDTree(centers)
        # find the 12 nearest neighbors in order to determine the
        # appropriate distance threshold
        dists, inds = centers_tree.query(centers, k=13)
        dists = dists[:, 1:]
        upper_limit = np.percentile(dists.flatten(), 99)
        dists = dists[(dists < upper_limit)].flatten()
        clusterer = cluster.KMeans(
            2, init=(np.array([0, 100]).reshape(-1, 1)),
            n_init=1).fit(dists[:, np.newaxis])
        distance_groups = clusterer.fit_predict(dists[:, np.newaxis])
        # use the maxmum distance in the neighbors group as a criterion
        # for IO diameters
        upper_limit = dists[(distance_groups == 0)].max()
        if square_lattice:
            num_neighbors = 4
        else:
            num_neighbors = 6
        neighbor_dists, neighbor_lbls = centers_tree.query(
            centers, k=(num_neighbors + 1), distance_upper_bound=upper_limit)
        neighbor_dists = neighbor_dists[:, 1:]
        neighbor_lbls = neighbor_lbls[:, 1:]
        # replace infs with nans
        to_replace = np.isinf(neighbor_dists)
        neighbor_dists[to_replace] = np.nan
        # get a larger neighborhood of points for calculating the normal vector
        big_neighborhood_dists, big_neighborhood_lbls = centers_tree.query(
            centers, k=51)
        to_replace = np.isinf(big_neighborhood_dists)
        big_neighborhood_dists[to_replace] = np.nan
        # start storing ommatidial measurements
        cross_section_area = []
        cross_section_height = []
        lens_area = []
        anatomical_vectors = []
        approx_vectors = []
        skewness = []
        neighborhood = []
        print('\nProcessing ommatidial data:')
        for num, (center, lbl, neighbor_group, neighbor_dist, big_group,
                  big_dist) in enumerate(zip(
                      centers, label_set, neighbor_lbls, neighbor_dists,
                      big_neighborhood_lbls, big_neighborhood_dists)):
            # find points within the cluster
            no_nans = np.isnan(neighbor_dist) == False
            neighbor_group = neighbor_group[no_nans]
            pts_nearby = centers[neighbor_group]
            # test: check that the neighbor distances match the actual neighbor distances
            # test_dists, test_inds = centers_tree.query(
            #     center, k=7, distance_upper_bound=upper_limit)
            # test_inds = test_inds[1:]
            # test_inds = test_inds[no_nans]
            # if no_nans.sum() > 0:
            #     if np.any(test_inds != neighbor_group):
            #         breakpoint()
            new_dists, new_inds = centers_tree.query(
                pts_nearby, k=len(neighbor_group) + 1)
            # fig = plt.figure()
            # ax = fig.add_subplot(projection='3d')
            # ax.scatter(center[0], center[1], center[2])
            # ax.scatter(pts_nearby[:, 0], pts_nearby[:, 1], pts_nearby[:, 2])
            # plt.show()
            # ignore neighbors with NaN distances from the center
            no_nans = np.isnan(big_dist) == False
            big_group = big_group[no_nans]
            # get centers of neighboring clusters, avoiding those with NaN distances
            big_neighborhood = centers[big_group]
            # 1. lens area ~ mean distance to nearest neighbors
            if np.all(np.isnan(neighbor_dist)):
                diam = np.nan
            else:
                diam = np.nanmean(neighbor_dist)
            area = np.pi * (0.5 * diam) ** 2
            lens_area += [area]
            # 2. ommatidial axis ~ mean normal vector of the plane formed by
            # neighboring ommatidia
            # center neighborhood of points
            pts_centered = big_neighborhood - center
            pts_centered = pts_centered[1:]
            # find cross product of each pair of neighoring ommatidia
            cross = np.cross(pts_centered[np.newaxis], pts_centered[:, np.newaxis])
            magn = np.linalg.norm(cross, axis=(-1))
            non_zero = magn != 0
            cross[non_zero] /= magn[(non_zero, np.newaxis)]
            # find angle between cross product and unit vector pointed at center
            sphere_vector = -center
            sphere_vector /= np.linalg.norm(sphere_vector)
            angles_between = np.arccos(np.dot(cross, sphere_vector))
            avg_vector = cross[(angles_between < np.pi / 2)]
            # center
            approx_vector = avg_vector.mean(0)
            approx_vectors += [approx_vector]
            # get the indices for points in this cluster
            inds = np.where(labels == lbl)[0]
            if len(inds) > 3 and lbl > 0:
                # 3. calculate the ommatidial axis by regressing the cluster data
                # get the ommatidial cluster coordinates
                pts_centered = np.copy(self.points[inds])
                pts_centered = pts_centered - center
                anatomical_basis = fit_line(pts_centered)
                # the anatomical vector is the direction vector that minimizes the angle
                # with the vector pointing to the eye center
                anatomical_vector = np.array([
                    anatomical_basis, -anatomical_basis])
                angs = np.arccos(np.dot(anatomical_vector, sphere_vector))
                ind = angs == angs.min()
                anatomical_vector = anatomical_vector[ind].min(0)
                # get the cross-sectional area of the cluster using the anatomical
                # vector. Use vector as basis, rotate pts, find area of convex hull
                pts_rotated = np.dot(pts_centered, anatomical_basis.T)
                # test: plot the rotated vs the original points in 3d
                # fig = plt.figure()
                # # plot the original centered points
                # ax = fig.add_subplot(121, projection='3d')
                # ax.scatter(pts_centered[:, 0], pts_centered[:, 1], pts_centered[:, 2])
                # # plot the newly rotated points
                # ax = fig.add_subplot(122, projection='3d')
                # ax.scatter(pts_rotated[:, 0], pts_rotated[:, 1], pts_rotated[:, 2])
                # plt.show()
                # find the 50% 2D kernal containing the rotated cross-section
                xs, ys = pts_rotated[:, 1], pts_rotated[:, 2]
                # get x and y ranges
                xvals, yvals = np.linspace(xs.min(), xs.max()), np.linspace(ys.min(), ys.max())
                xgrid, ygrid = np.meshgrid(xvals, yvals)
                position = np.array([xgrid.flatten(), ygrid.flatten()])
                xlen, ylen = xvals[1] - xvals[0], yvals[1] - yvals[0]
                pixel_area = xlen * ylen
                try:
                    # find 2D kernal density estimation 
                    kernal = stats.gaussian_kde(pts_rotated[:, 1:].T)
                    density = kernal(position).reshape((50, 50))
                    density /= density.max() # normalize to maximum
                    # get the area of the 50% density kernal
                    cross_sectional_area = (density > .5).sum() * pixel_area
                    # get the height using the first principle component
                    heights = pts_rotated[:, 0]
                    cross_sectional_height = heights.ptp()
                except:
                    cross_sectional_area = np.nan
                    cross_sectional_height = np.nan
            else:
                anatomical_vector = np.array([np.nan, np.nan, np.nan])
                cross_sectional_area = np.nan
                cross_sectional_height = np.nan
            anatomical_vectors += [anatomical_vector]
            cross_section_area += [cross_sectional_area]
            cross_section_height += [cross_sectional_height]
            # 4. calculate the ommatidial skewness as the inside angle between
            # the approximate and anatomical vectors
            if len(pts_centered) > 0:
                inside_ang = angle_between(approx_vector, anatomical_vector)
            else:
                inside_ang = np.nan
            skewness += [inside_ang]
            # 5. store nearby groups
            neighborhood += [label_set[neighbor_group]]
            print_progress(num + 1, len(centers))
        # covert to numpy arrays
        lens_area = np.array(lens_area)
        anatomical_vectors = np.array(anatomical_vectors)
        approx_vectors = np.array(approx_vectors)
        skewness = np.array(skewness) * 180 / np.pi # in degrees
        # calculate the spherical IO angle using eye radius and ommatidial diameter
        diameter = 2 * np.sqrt(lens_area / np.pi)
        # calculate the reduction in diameter due to skewness
        diameter_adjusted = diameter * np.cos(skewness * np.pi / 180)
        radius = np.nanmean(self.ommatidial_data.radius)
        spherical_IO_angle = diameter / radius * 180 / np.pi
        # add to the spreadsheet
        for var, arr in zip(
                ['lens_area', 'anatomical_axis', 'approx_axis',
                 'skewness', 'spherical_IOA', 'neighbors', 'cross_section_area',
                 'cross_section_height', 'lens_diameter', 'lens_diameter_adj'],
                [lens_area, anatomical_vectors.tolist(), approx_vectors.tolist(),
                 skewness, spherical_IO_angle, neighborhood, cross_section_area,
                 cross_section_height, diameter, diameter_adjusted]):
            self.ommatidial_data[var] = arr
        # save as csv and pickle object for pandas
        csv_filename = os.path.join(self.dirname, 'ommatidial_data.csv')
        self.ommatidial_data.to_csv(csv_filename, index=False)
        self.ommatidial_data.to_pickle(csv_filename.replace('.csv', '.pkl'))
        labels = self.ommatidial_data.neighbors.values
        cluster_lbls = self.labels[:]
        # test: do the neighborhoods result as expected? plot all the pairs in polar coords
        # process interommatidial pair data
        pairs = []
        orientations = []
        pair_centers = []
        print('\nPre-processing interommatidial pairs:')
        # iterate through clusters and get unique pairs based on the neighborhood
        for num, cone in self.ommatidial_data.iterrows():
            lbl = cone.label
            neighbor_labels = cone.neighbors
            for neighbor_ind in neighbor_labels:
                pair = tuple(sorted([lbl, neighbor_ind]))
                # only go through each pair once
                if all([pair not in pairs,
                        np.any(self.ommatidial_data.label.values == neighbor_ind)]):
                    try:
                        inds = np.where(self.ommatidial_data.label.values == neighbor_ind)[0][0]
                        neighbor_cone = self.ommatidial_data.iloc[inds]
                    except:
                        breakpoint()
                    inds = np.where(cluster_lbls == neighbor_ind)[0]
                    theta1, phi1 = cone[['theta', 'phi']].values.T
                    x, y, z = cone[['x', 'y', 'z']]
                    # use the 2 polar coordinates to get the interommatidial orientation
                    theta2, phi2 = neighbor_cone[['theta', 'phi']].values.T
                    neighbor_x, neighbor_y, neighbor_z = neighbor_cone[['x', 'y', 'z']]
                    # test: check that the distance between neighbors is less than the upper limit
                    pt_1 = np.array([x, y, z]).T
                    pt_2 = np.array([neighbor_x, neighbor_y, neighbor_z]).T
                    dist = np.linalg.norm(pt_2 - pt_1)
                    # if dist > upper_limit:
                    #     breakpoint()
                    # get the angle
                    if theta1 < theta2:
                        orientation = np.array([phi2 - phi1, theta2 - theta1])
                    else:
                        orientation = np.array([phi1 - phi2, theta1 - theta2])
                    orientations += [orientation]
                    # store the pair to avoid reprocessing
                    pairs += [pair]
                    pair_centers += [
                        ((x + neighbor_x) / 2,
                         (y + neighbor_y) / 2,
                         (z + neighbor_z) / 2)]
                    print_progress(num + 1, len(labels))
        # make arrays
        pair_centers = np.array(list(pair_centers))
        pairs_tested = np.array(list(pairs))
        orientations = np.array(list(orientations))
        # store for later
        var = 'pairs_tested'
        if var in dir(self):
            delattr(self, var)
        if var in self.database.keys():
            del self.database[var]
        dataset = self.database.create_dataset(
            var, data=pairs_tested)

    def measure_interommatidia(self, test=False, display=False, num_slices=21,
                                    neighborhood_smoothing=1):
        """Use the anatomical axes to quickly measure interommatidial angles.

        
        Parameters
        ----------
        test : bool, default=False
            Whether to run troublshooting options.
        display : bool, default=False
            Whether to display the processed information.
        neighborhood_smoothing : int, default=1
            The order of neighbors to include in averaging. 1 means use the first-order 
            neighbors (max of 6 in a hexagonal lattice).
        
        Attributes
        ----------
        
        """
        # model the 3D vectors as a function of azimuth and elevation
        thetas, phis = self.ommatidial_data[['theta', 'phi']].values.T
        pts = self.ommatidial_data[['x', 'y', 'z']].values
        axes = self.ommatidial_data.anatomical_axis.values
        axes = np.array([ax for ax in axes])
        if test:
            # transform cluster elements as well
            children = self.points[:]
            children_lbls = self.labels[:]
        # smooth the 3 directional components first, and then slice and dice
        new_axes = np.zeros(axes.shape)
        for vals, new_vals in zip(axes.T, new_axes.T):
            distance_tree = spatial.KDTree(pts)
            # figure out the maximum distance for nearest neighbors
            dists, inds = distance_tree.query(pts, k=12)
            # use K-means to find the first mode after removing 0's
            dists = dists[:, 1:]
            clusterer = cluster.KMeans(n_clusters = 2, init=np.array([[0], [100]]))
            groups = clusterer.fit_predict(dists.flatten()[:, np.newaxis])
            dist_upper_limit = dists.flatten()[groups == 0].max()
            # query the distance tree for nearest neighbors within the dist_upper_limit
            num = neighborhood_smoothing
            num_neighbors = 1 + 3 * (num + 1) * (num)
            dists, inds = distance_tree.query(pts, k=num_neighbors,
                                              distance_upper_bound=num*dist_upper_limit)
            vals_in_neighborhood = []
            for ind in inds:
                include = ind < len(vals)
                vals_in_neighborhood += [np.nanmean(vals[ind[include]])]
            new_vals[:] = np.array(vals_in_neighborhood)
            if test:
                fig, fig_axes = plt.subplots(ncols=2)
                fig_axes[0].scatter(thetas, phis, c=vals,
                                    vmin=np.nanmin(vals), vmax=np.nanmax(vals),
                                    marker='.')
                fig_axes[1].scatter(thetas, phis, c=new_vals,
                                    vmin=np.nanmin(vals), vmax=np.nanmax(vals),
                                    marker='.')
                [ax.set_aspect('equal') for ax in fig_axes]
                plt.show()
        # normalize the new directional vectors to have norm = 1
        new_axes /= np.linalg.norm(new_axes, axis=-1)[:, np.newaxis]
        # use SVD to reorient pts and axes such that the dimensions are in decreasing order
        com = pts.mean(0)
        pts -= com
        # do the same to the cluster elements
        if test:
            children -= com
        uu, dd, vv = np.linalg.svd(pts)
        # cross product of centered coordinates and vv should reorient the points
        pts = np.dot(pts, vv.T)[:, [1, 2, 0]]
        axes = np.dot(axes, vv.T)[:, [1, 2, 0]]
        new_axes = np.dot(new_axes, vv.T)[:, [1, 2, 0]]
        if test:
            # same to the cluster elements
            children = np.dot(children, vv.T)[:, [1, 2, 0]]
        # get angle components
        h_angs = np.arctan2(axes[:, 1], axes[:, 0])
        v_angs = np.arctan2(axes[:, 2], axes[:, 1])
        # center the points using the center of the best fitted sphere
        no_nans = np.isnan(pts) == False
        no_nans = np.any(no_nans, axis=1)
        pt_model = SphereFit(np.copy(pts[no_nans]))
        center = pt_model.center
        pts -= center
        if test:
            children -= center
        # get new thetas and phis
        phis, thetas, radii = rectangular_to_spherical(pts).T
        # get interommatidial pairs of labels
        pair_labels = self.pairs_tested[:]
        # if test:
        #     # plot the ommatidial axes in polar coordinates
        #     dth, dph, dr = rectangular_to_spherical(axes).T
        #     th, ph, r = rectangular_to_spherical(pts).T
        #     fig = plt.figure()
        #     plt.scatter(th, ph, c=r)
        #     l = 0.005
        #     lows = [th - l * dth, ph - l * dph]
        #     highs = [th + l * dth, ph + l * dph]
        #     plt.plot([lows[0], highs[0]], [lows[1], highs[1]], color='k', alpha=0.1)
        #     plt.gca().set_aspect('equal')
        #     plt.show()
        #     # plot the resulting angular measurements
        #     fig, axs = plt.subplots(ncols=2)
        #     h_ax, v_ax = axs
        #     no_nans = np.isnan(v_angs) == False
        #     no_nans *= np.isnan(h_angs) == False
        #     random_order = np.random.permutation(np.arange(sum(no_nans)))
        #     for ax, vals, title in zip(
        #             [h_ax, v_ax], [np.unwrap(h_angs[no_nans]), np.unwrap(v_angs[no_nans])],
        #             ['Horizontal', 'Vertical']):
        #         ax.scatter((thetas[no_nans][random_order]),
        #                    (phis[no_nans][random_order]), c=(vals[random_order]),
        #                    alpha=0.4)
        #         ax.set_aspect('equal')
        #         ax.set_title(title)
        #     plt.show()
        #     # scatter plots for each vector component
        #     fig, axs = plt.subplots(ncols=3)
        #     for ax, vals, title in zip(axs, axes.T, [
        #      'x', 'y', 'z']):
        #         scatter = ax.scatter((thetas[random_order]), (phis[random_order]),
        #                              c=(vals[random_order]), alpha=0.8)
        #         ax.set_aspect('equal')
        #         ax.set_title(title + ' component')
        #         plt.colorbar(scatter, ax=ax)
        #     plt.show()
        #     # plot each component within a narrow band of elevations around the center
        #     included = np.abs(phis - phis.mean()) < np.pi / 8
        #     fig, axs = plt.subplots(ncols=3)
        #     for ax, vals, title in zip(axs, new_axes.T, [
        #      'x', 'y', 'z']):
        #         ax.scatter((thetas[included]), (vals[included]), color='k')
        #         ax.set_aspect('equal')
        #         ax.set_title(title + ' component')
        #     plt.show()
        #     # plot all the interommatidial pairs in polar coordinates to
        #     # be sure that they are nearest neighbors
        #     thetas1, thetas2 = thetas[pair_labels].T
        #     phis1, phis2 = phis[pair_labels].T
        #     fig, ax = plt.subplots()
        #     ax.plot([thetas1, thetas2], [phis1, phis2])
        #     ax.set_aspect('equal')
        #     plt.show()
        # give vertical and horizontal slice numbers to each interommatidial pair
        labels = np.copy(self.ommatidial_data.label.values)
        # remove any labels if either within a pair are not included in dataset already
        in_dataset = np.all((np.isin(pair_labels, labels)), axis=1)
        pair_labels = pair_labels[in_dataset]
        # grab the relevant cluster centers and ommatidial axes
        pair_inds = np.searchsorted(labels, pair_labels)
        # pair_inds -= 1
        pair_pts = pts[pair_inds]
        pair_axes = axes[pair_inds]
        pair_axes_smooth = new_axes[pair_inds]
        # calculate centers of mass and pair orientations
        pair_centers = pair_pts.mean(1)
        # get pair orientation in polar coordinates
        pair_polar = rectangular_to_spherical(np.vstack(pair_pts)).reshape(pair_pts.shape)
        pair_orientations = pair_polar[:, 0, :2] - pair_polar[:, 1, :2]
        pair_diams = np.linalg.norm((pair_pts[:, 0] - pair_pts[:, 1]), axis=(-1))
        pair_orientations /= np.linalg.norm(pair_orientations, axis=1)[:, np.newaxis]
        # IO pairs are aligned with the x-axis
        # todo: use the x and y components of the orientations because of points about the boundary
        init_angs = (np.arange(6) * 2 * np.pi / 6)
        init_coords = np.array([np.cos(init_angs), np.sin(init_angs)]).T
        clusterer = cluster.KMeans(6, init=init_coords, n_init=1).fit(pair_orientations)
        cluster_centers = clusterer.cluster_centers_
        # i. consider the orientations and their compliments, using only those in the upper half
        # pos_orientations = np.copy(pair_orientations)
        # neg_oris = pos_orientations[:, 1] < 0
        # pos_orientations[neg_oris] = -pos_orientations[neg_oris]
        # # ii. partition the orientations into the 3 clusters
        # init_angles = np.arange(3) * np.pi/6
        # init_coords = np.array([np.cos(init_angles), np.sin(init_angles)]).T
        # clusterer = cluster.KMeans(n_clusters=len(init_coords),
        #                            init=init_coords).fit(pos_orientations)
        # cluster_centers = clusterer.cluster_centers_
        if test:
            plt.scatter(pair_orientations[:, 0], pair_orientations[:, 1], alpha=.005)
            plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c=init_angs)
            plt.gca().set_aspect('equal')
            plt.show()
        # ii. find the cluster center with an angle closest to np.pi/2
        cluster_angles = np.arctan2(cluster_centers[:, 0], cluster_centers[:, 1])
        horizontal_angles = cluster_angles % (2 * np.pi)
        horizontal_axis = np.argmin(abs(horizontal_angles))
        # horizontal_axis = np.argmin(abs(cluster_angles))
        # iii. rotate everything by the difference between the horizontal axis and np.pi/2
        ang_diff = -cluster_angles[horizontal_axis]
        # ang_diff = cluster_angles[horizontal_axis]
        # ang_diff = 0 - cluster_angles[horizontal_axis]
        # test: check that the rotation is correct at every level. for each of the following,
        # check that either the main orientation is around np.pi/2 or that a major ommatidial
        # axis is horizontal when plotted:
        # pair_pts
        # test_rot = rotate(pair_pts[:, 0], ang_diff, axis=1).T
        # fig, axes = plt.subplots(ncols=2)
        # for ax, vals in zip(axes, [pair_pts[:, 0], test_rot]):
        #     ax.scatter(vals[:, 0], vals[:, 2])
        #     ax.set_aspect('equal')
        # plt.show()
        # breakpoint()
        pair_pts = rotate(pair_pts, ang_diff, axis=1).T
        pair_axes = rotate(pair_axes, ang_diff, axis=1).T
        pair_axes_smooth = rotate(pair_axes_smooth, ang_diff, axis=1).T
        if test:
            children = rotate(children, ang_diff, axis=1).T
        # for posterity, store the rotated points and axes in self.ommatidial_data
        pts = rotate(pts, ang_diff, axis=1).T
        axes = rotate(axes, ang_diff, axis=1).T
        axes_smooth = rotate(new_axes, ang_diff, axis=1).T
        self.ommatidial_data[['x_', 'y_', 'z_']] = pts
        self.ommatidial_data[['dx_', 'dy_', 'dz_']] = axes_smooth
        self.ommatidial_data[['dx', 'dy', 'dz']] = axes
        csv_filename = os.path.join(self.dirname, 'ommatidial_data.csv')
        self.ommatidial_data.to_csv(csv_filename, index=False)
        self.ommatidial_data.to_pickle(csv_filename.replace('.csv', '.pkl'))
        # convert to polar coordinates
        pair_polar = rectangular_to_spherical(
            np.vstack(pair_pts)).reshape(pair_pts.shape)
        pair_orientations = pair_polar[:, 0, :2] - pair_polar[:, 1, :2]
        pair_diams = np.linalg.norm((pair_pts[:, 0] - pair_pts[:, 1]), axis=(-1))
        pair_orientations /= np.linalg.norm(pair_orientations, axis=1)[:, np.newaxis]

        orientation_components = np.copy(pair_orientations)
        pair_orientations = np.arctan2(pair_orientations[:, 1], pair_orientations[:, 0])
        # for each section, project anatomical vectors onto the parallel plane
        # and ignore the flattened dimension. For instance, for a vertical
        # section, we flatten the dimension of width, ignoring x values and
        # then find the polar angle of all the flattened points (y, z), rotating
        # to avoid the polar boundary problems. Store the resulting angle
        # between projected axis vectors as a measure of one component of the
        # io angle
        # make an empty dataset with one row per interommatidial pair
        cols = [
            'lbl1', 'lbl2',
            'pt1_x', 'pt1_y', 'pt1_z', 'pt1_th', 'pt1_ph', 'pt1_r',
            'pt1_dx', 'pt1_dy', 'pt1_dz',
            'pt1_dx_', 'pt1_dy_', 'pt1_dz_',
            'pt2_x', 'pt2_y', 'pt2_z', 'pt2_th', 'pt2_ph', 'pt2_r',
            'pt2_dx', 'pt2_dy', 'pt2_dz',
            'pt2_dx_', 'pt2_dy_', 'pt2_dz_',
            'orientation', 'diameter',
            'angle_h', 'angle_v', 'angle_total']
        interommatidial_data = pd.DataFrame((np.zeros((len(pair_labels), len(cols)))),
                                            columns=cols)
        # store the reference data
        for col, val in zip(cols[:2], [pair_labels[:, 0], pair_labels[:, 1]]):
            interommatidial_data[col] = val
        for pt, arr, polar, axes_original, axes_new in zip(
                ['1', '2'], [pair_pts[:, 0], pair_pts[:, 1]],
                [pair_polar[:, 0], pair_polar[:, 1]],
                [pair_axes[:, 0], pair_axes[:, 1]],
                [pair_axes_smooth[:, 0], pair_axes_smooth[:, 1]]):
            for col, vals in zip([
             f"pt{pt}_x", f"pt{pt}_y", f"pt{pt}_z"], arr.T):
                interommatidial_data[col] = vals
            for col, vals in zip([
             f"pt{pt}_ph", f"pt{pt}_th", f"pt{pt}_r"], polar.T):
                interommatidial_data[col] = vals
            for col, vals in zip([
             f"pt{pt}_dx", f"pt{pt}_dy", f"pt{pt}_dz"], axes_original.T):
                interommatidial_data[col] = vals
            for col, vals in zip([
             f"pt{pt}_dx_", f"pt{pt}_dy_", f"pt{pt}_dz_"], axes_new.T):
                interommatidial_data[col] = vals
        interommatidial_data.orientation = pair_orientations
        interommatidial_data['ori_dx'] = orientation_components[:, 0]
        interommatidial_data['ori_dy'] = orientation_components[:, 1]
        num = 0
        # for flat_dim, angle_col in zip([0, 2], ['angle_h', 'angle_v']):
        #     # get the un-flattened dimensions
        #     other_dims = np.arange(3)
        #     other_dims = other_dims[other_dims != flat_dim]
        #     # calculate the angle between pairs as the arccosine of the dot products
        #     dots = []
        #     for pair in pair_axes_smooth:
        #         dots += [np.dot(pair[0, other_dims], pair[1, other_dims])]
        #     dots = np.array(dots)
        #     angles = np.arccos(dots)
            

        # process interommatidial pairs in vertical and horizontal sections
        print('\nProcessing interommatidial data:')
        for flat_dim, angle_col in zip(
                [0, 2], ['angle_v', 'angle_h']):
            # find limits of the flattened dimension
            flat_vals = pair_centers[(..., flat_dim)]
            limits = np.linspace(flat_vals.min(), flat_vals.max(), num_slices)
            # find pair centers within the limits of the non-flat dimensions
            for lim_low, lim_high in zip(limits[:-1], limits[1:]):
                include = pair_centers[:, flat_dim] >= lim_low
                include *= pair_centers[:, flat_dim] <= lim_high
                no_nans = np.isnan(pair_axes) == False
                no_nans = np.all(no_nans, axis=(1, 2))
                include *= no_nans
                if include.sum() > 3:
                    labels_in_slice = pair_labels[include]
                    pts_in_slice = pair_pts[include]
                    axes_in_slice = pair_axes[include]
                    axes_new = pair_axes_smooth[include]
                    # get non-flattened dimensions
                    other_dims = np.arange(pts.shape[(-1)])
                    other_dims = other_dims[(other_dims != flat_dim)]
                    other_dim, depth = pts_in_slice[(..., other_dims)].T
                    # get the axes
                    x, y = np.copy(pts_in_slice[(..., other_dims)].transpose(2, 0, 1))
                    dx, dy = np.copy(axes_in_slice[(..., other_dims)].transpose(2, 0, 1))
                    new_dx, new_dy = np.copy(axes_new[(..., other_dims)].transpose(2, 0, 1))
                    # convert 2D position to get 1D polar angle
                    theta = np.arctan2(x, y)
                    theta = np.unwrap(theta, axis=0)
                    # convert the 2D axis to a 1D polar angle
                    phi = np.arctan2(dx, dy)
                    phi = np.unwrap(phi, axis=0)
                    # compare to smoothed measurements
                    phi_new = np.arctan2(new_dx, new_dy)
                    phi_new = np.unwrap(phi_new, axis=0)
                    axes_new = np.array([new_dx, new_dy]).transpose(1, 2, 0)  # has shape (num_samples, 2 cones, 2 dimensions)
                    no_nans = np.isnan(phi_new) == False
                    no_nans *= np.isnan(phi) == False
                    corr_r, corr_p = stats.pearsonr(phi[no_nans], phi_new[no_nans])
                    # # use polynomial fitting to regress phi on theta
                    # model, resids = positive_fit(
                    #     theta.flatten(), phi.flatten()[:, np.newaxis])
                    # new_theta = np.copy(theta)
                    # new_phi = model(theta)
                    # # get dy and dx base on modelled phi
                    # norm = np.sqrt(dx ** 2 + dy ** 2)
                    # new_dy = norm * np.cos(new_phi)
                    # new_dx = norm * np.sin(new_phi)
                    # new_x = np.copy(x)
                    # new_y = np.copy(y)
                    # axes_new = np.array([new_dx, new_dy]).transpose(1, 2, 0)  # has shape (num_samples, 2 cones, 2 dimensions)
                    # # check correlation
                    # corr_r, corr_p = stats.pearsonr(phi.flatten(), new_phi.flatten())
                    # test: can I replace this level of modelling and use the
                    # smoothed ommatidial axes?
                    prop = (num % num_slices) / num_slices
                    key_props = np.array([0, .5, 1.])
                    if test and any(abs(prop - key_props) < .05) and flat_dim == 2:
                        # plot the position and direction vectors of the modeled and real data
                        l = 1000
                        fig, axs = plt.subplots(ncols=2)
                        cart_ax, polar_ax = axs
                        data_line = cart_ax.plot(
                            [x.flatten() - l * dx.flatten(),
                             x.flatten() + l * dx.flatten()],
                            [y.flatten() - l * dy.flatten(),
                             y.flatten() + l * dy.flatten()],
                            color=green, alpha=0.5)
                        cart_ax.scatter(x, y)
                        cart_ax.set_aspect('equal')
                        # plot polar coords and projected ommatidial axes
                        polar_ax.scatter(theta, phi)
                        inds = np.argsort(theta.flatten())
                        polar_ax.scatter(
                            theta.flatten()[inds], phi_new.flatten()[inds],
                            color='k', marker='.')
                        polar_ax.set_aspect('equal')
                        polar_ax.set_title(f"r={corr_r}, p={corr_p}")
                        # plot the coords and new phi
                        model_line = cart_ax.plot(
                            [x.flatten() - l * new_dx.flatten(),
                             x.flatten() + l * new_dx.flatten()],
                            [y.flatten() - l * new_dy.flatten(),
                             y.flatten() + l * new_dy.flatten()],
                            color=red, alpha=0.5)
                        cart_ax.set_aspect('equal')
                        plt.show()
                        # todo: plot the clusters with individual colors
                        # grab the points for each label in the slice
                        labels_set = np.unique(labels_in_slice)
                        l = 2000
                        if flat_dim == 0:
                            fig, axs = plt.subplots(ncols=2, figsize=(12, 8), sharey=True)
                        else:
                            fig, axs = plt.subplots(nrows=2, figsize=(8, 12), sharex=True)
                        summary_ax, cart_ax = axs
                        # show which part of the slice we are plotting
                        summary_ax.scatter(
                            pair_centers[:, 0], pair_centers[:, 2],
                            color='gray')
                        summary_ax.scatter(
                            pair_centers[:, 0][include],
                            pair_centers[:, 2][include],
                            color='k', marker='.')
                        summary_ax.set_aspect('equal')
                        # grab the elements of each label in the slice
                        labels = []
                        slice_inds = []
                        labels_all = self.labels[:]
                        for label in labels_set:
                            inds = labels_all == label
                            slice_inds += [np.where(inds)[0]]
                            labels += [np.repeat(label, inds.sum())]
                        labels = np.concatenate(labels, axis=0)
                        slice_inds = np.concatenate(slice_inds, axis=0) 
                        # grab cluster elements in order by cluster label
                        order = np.argsort(labels)
                        labels = labels[order]
                        slice_inds = slice_inds[order]
                        children_sub = np.zeros((len(slice_inds), 3), dtype='float')
                        children_sub[order] = children[slice_inds[order]]
                        children_x, children_y = children_sub[..., other_dims].T
                        (xmin, ymin), (xmax, ymax) = np.percentile(
                            children[..., other_dims], [0, 100], axis=0)
                        # # get new labels
                        new_lbls_set, new_lbls = np.unique(labels, return_inverse=True)
                        lbls_random = np.random.permutation(np.arange(new_lbls.max()))
                        clbls = np.zeros(len(new_lbls))
                        for lbl, replace_lbl in zip(np.unique(new_lbls), lbls_random):
                            clbls[new_lbls == lbl] = replace_lbl
                        # data_line = cart_ax.plot(
                        #     [x.flatten() - l * dx.flatten(),
                        #      x.flatten() + l * dx.flatten()],
                        #     [y.flatten() - l * dy.flatten(),
                        #      y.flatten() + l * dy.flatten()],
                        #     color=green, alpha=0.5)
                        colors = plt.cm.tab20(clbls/clbls.max())
                        # colors = plt.cm.tab10(new_lbls/new_lbls.max())
                        
                        # todo: plot same colors for both main plot and the zoomed inset
                        # order = np.argsort(new_lbls)
                        # ch_x, ch_y, new_lbls = children_x[order], children_y[order], new_lbls[order]                        
                        ch_x, ch_y = children_x, children_y
                        changes = np.diff(new_lbls)
                        changes = np.where(changes > 0)[0]
                        changes += 1
                        starts = np.append([0], changes)
                        ends = np.append(changes, [len(new_lbls)])
                        for start, end, color in zip(
                                starts, ends, colors[starts]):
                            pts_sub = np.array([ch_x[start:end], ch_y[start:end]]).T
                            # eliminate outliers
                            dist_tree = spatial.KDTree(pts_sub)
                            dists, inds = dist_tree.query(pts_sub, k=3)
                            dists = dists[:, 1:].min(1)
                            # remove dists > 99%
                            thresh = np.percentile(dists, 99)
                            include_sub = dists < thresh
                            if include_sub.sum() > 3:
                                # use 2d histogram to get outline
                                sub = pts_sub[include_sub]
                                # sub_order = np.argsort(sub[:, 0])
                                # kde = stats.gaussian_kde(sub[sub_order].T)
                                # # get kernal density estimate
                                # xmin, xmax = np.percentile(sub[:, 0], [0, 100])
                                # ymin, ymax = np.percentile(sub[:, 1], [0, 100])
                                # xvals = np.linspace(xmin, xmax, 50)
                                # yvals = np.linspace(ymin, ymax, 50)
                                # xgrid, ygrid = np.meshgrid(xvals, yvals)
                                # pos = np.array([xgrid, ygrid]).T
                                # # kernal estimation
                                # probs = kde(pos.reshape(-1, 2).T).T.reshape(pos.shape[:-1])
                                # probs /= probs.max()
                                # # get 50% approximate contour
                                # edge = skimage.feature.canny(probs > .95)
                                # xs, ys = xgrid[edge], ygrid[edge]
                                # contours = skimage.measure.find_contours(probs, level=.5)
                                # contour = max(contours, key=len)
                                # contour = np.round(contour).astype(int)
                                # contour_pts = pos[contour[:, 1], contour[:, 0]]
                                # xs, ys = contour_pts.T
                                # plt.pcolormesh(xgrid, ygrid, probs)
                                # plt.scatter(sub[:, 0], sub[:, 1])
                                # plt.plot(xs, ys, color='r')
                                # plt.gca().set_aspect('equal')
                                # plt.show()
                                # sub_x, sub_y = pts_sub[include_sub].T
                                # hist, xvals, yvals = np.histogram2d(sub_x, sub_y, bins=10)
                                chull = spatial.ConvexHull(pts_sub[include_sub])
                                xs, ys = pts_sub[include_sub][chull.vertices].T
                                cart_ax.fill(xs, ys, color=color, alpha=.5, edgecolor='none')
                        # svds = np.array(svds)
                        # centers = np.array(centers)
                        # cart_ax.scatter(children_x, children_y, c=clbls, cmap='tab20',
                        #                 alpha=.25, marker='o', edgecolors='none')
                        # TODO: add zoomed insets to the 25, 50, and 75th percentile y or x values
                        if flat_dim == 0:
                            key_vals = np.percentile(children_y, [25, 50, 75])
                            other_vals = []
                            for key_val in key_vals:
                                ind = np.argmin(abs(y.mean(1) - key_val))
                                other_vals += [x.mean(1)[ind].mean()]
                            key_vals = np.array([other_vals, key_vals]).T
                        else:
                            key_vals = np.percentile(children_x, [25, 50, 75])
                            other_vals = []
                            for key_val in key_vals:
                                ind = np.argmin(abs(x.mean(1) - key_val))
                                other_vals += [y.mean(1)[ind].mean()]
                            key_vals = np.array([key_vals, other_vals]).T
                        # for each key value, plot the inset and zoomed inset indicator
                        width = 100
                        locs = ['upper left', 'center left', 'lower left']
                        for lbl, roi, loc in zip(['A.', 'B.', 'C.'], key_vals[::-1], locs):
                            # ins_ax = cart_ax.inset_axes([roi[0], roi[1], width, width],
                            #                             transform=cart_ax.transData)
                            ins_ax = zoomed_inset_axes(
                                cart_ax, 4, loc=loc)
                            ins_ax.text(.05, .05, lbl, transform=ins_ax.transAxes)
                            
                            low, high = roi - width/2, roi + width/2
                            # plot the roi box
                            cart_ax.plot([low[0], low[0], high[0], high[0], low[0]],
                                         [low[1], high[1], high[1], low[1], low[1]],
                                         color='k')
                            # plot included children
                            children_x_roi = (children_x >= low[0]) * (children_x <= high[0])
                            children_y_roi = (children_y >= low[1]) * (children_y <= high[1])
                            roi_inds = children_x_roi * children_y_roi
                            ins_ax.scatter(children_x[roi_inds], children_y[roi_inds],
                                           c=colors[roi_inds], alpha=.1,
                                           marker='.', edgecolors='none')
                            # plot the centers
                            roi_inds = (x >= low[0]) * (x <= high[0])
                            roi_inds *= (y >= low[1]) * (y <= high[1])
                            ins_ax.scatter(x[roi_inds], y[roi_inds], color='k', marker='o')
                            # plot the lines
                            lines = ins_ax.plot(
                                [x[roi_inds].flatten(),
                                 x[roi_inds].flatten() + l * new_dx[roi_inds].flatten()],
                                [y[roi_inds].flatten(),
                                 y[roi_inds].flatten() + l * new_dy[roi_inds].flatten()],
                                color='k', alpha=.2)
                            # limit the inset
                            ins_ax.set_xlim(roi[0] - width/2, roi[0] + width/2)
                            ins_ax.set_ylim(roi[1] - width/2, roi[1] + width/2)
                            ins_ax.set_xticks([])
                            ins_ax.set_yticks([])
                            # mark the inset
                            # mark_inset(cart_ax, ins_ax, loc1=1, loc2=3,
                            #            fc='none', ec='k', lw=.5)
                        cart_ax.scatter(x, y, marker='.', color='k')
                        cart_ax.set_aspect('equal')
                        # cart_ax.set_xticks([])
                        # cart_ax.set_yticks([])
                        # polar_ax.scatter(theta, phi)
                        # inds = np.argsort(theta.flatten())
                        # polar_ax.scatter(
                        #     theta.flatten()[inds], phi_new.flatten()[inds],
                        #     color='k', marker='.')
                        # polar_ax.set_aspect('equal')
                        # polar_ax.set_title(f"r={corr_r:.2f}, p={corr_p:.4f}")
                        # plot the coords and new phi
                        center_x, center_y = np.copy(
                            pts_in_slice[(..., other_dims)].transpose(2, 0, 1))
                        model_line = cart_ax.plot(
                            [center_x.flatten() - 0 * new_dx.flatten(),
                             center_x.flatten() + l * new_dx.flatten()],
                            [center_y.flatten() - 0 * new_dy.flatten(),
                             center_y.flatten() + l * new_dy.flatten()],
                            color='k', alpha=0.05)
                        # svd_line = cart_ax.plot(
                        #     [centers[:, 0] - 0 * svds[:, 0],
                        #      centers[:, 0] + l * svds[:, 0]],
                        #     [centers[:, 1] - 0 * svds[:, 1],
                        #      centers[:, 1] + l * svds[:, 1]],
                        #     color='r', alpha=0.05)
                        cart_ax.set_aspect('equal')
                        # figure out helpful x and y limits
                        xs, ys = pts[..., other_dims].T
                        xmin, xmax = np.percentile(xs, [0, 100])
                        ymin, ymax = np.percentile(ys, [0, 100])
                        xmin, xmax = min(1.1 * xmin, 0), max(1.1 * xmax, 0)
                        ymin, ymax = min(1.1 * ymin, 0), max(1.1 * ymax, 0)
                        cart_ax.set_xlim(xmin, xmax)
                        cart_ax.set_ylim(ymin, ymax)
                        if flat_dim == 0:
                            summary_ax.set_ylim(ymin, ymax)
                        else:
                            summary_ax.set_xlim(xmin, xmax)
                        # plt.tight_layout()
                        plt.show()
                    # measure IO angles per pair
                    # measure angle between new_axes[:, 0], and new_axes[:, 1]
                    angles_included = []
                    inds = []
                    norm = np.linalg.norm(axes_new, axis=(-1))
                    axes_new_normed = axes_new / norm[(..., np.newaxis)]
                    dots = np.array([np.dot(pair[0], pair[1]) for pair in axes_new_normed])
                    angles = np.arccos(dots)
                    lbls = interommatidial_data[['lbl1', 'lbl2']].values
                    # store the measured angles
                    for angle, lbl_pair in zip(angles, labels_in_slice):
                        ind = np.all((lbls == lbl_pair), axis=1)
                        if np.any(ind):
                            inds += [np.where(ind)[0][0]]
                            angles_included += [angle]
                        interommatidial_data.loc[(inds, angle_col)] = angles_included
                    # calculate the total IO angle
                    dx, dy = interommatidial_data.angle_h, interommatidial_data.angle_v
                    interommatidial_data.angle_total = np.sqrt(dx ** 2 + dy ** 2)
                    num += 1
                    print_progress(num, 2 * num_slices)
        csv_filename = os.path.join(self.dirname, 'interommatidial_data.csv')
        interommatidial_data.to_csv(csv_filename, index=False)
        interommatidial_data.to_pickle(csv_filename.replace('.csv', '.pkl'))


    def plot_raw_data(self, three_d=False, app=None, window=None, **kwargs):
        """Function for plotting the imported data with a 3D option.

        Parameters
        ----------
        three_d : bool, default=False
            Whether to use pyqtgraph to plot the data in 3D.
        """
        max_points = 10000000
        if self.points.shape[0] > max_points:
            pts = self.points_subset[:]
        else:
            pts = self.points[:]
        if three_d:
            window.setWindowTitle('Raw Data')
            scatter = ScatterPlot3d(pts, title='Imported Stack', app=app, window=window)
            scatter.show()
        else:
            xs, ys, zs = pts.T
            order = np.argsort(zs)
            scatter = plt.scatter((xs[order]), (ys[order]), c=(zs[order]), alpha=0.1,
                                  marker='.')
            plt.gca().set_aspect('equal')
            plt.colorbar()
            plt.show()

    def plot_cross_section(self, three_d=False, residual_proportion=0.5, app=None,
                           window=None, img_size=1e5, margins=True, inset=False,
                           **kwargs):
        """Plot the points near the cross section fit along the eye surface.

        Parameters
        ----------
        three_d : bool, default=False
            Whether to use pyqtgraph to plot the cross section in 3D.
        residual_proportion : float, default
            Proportion of the residuals to include in cross section, 
            affecting its thickness.
        """
        percentage = residual_proportion * 100
        residuals = self.residual[:]
        residuals_sq = residuals ** 2
        # low, high = np.percentile(residuals, [50 - percentage / 2, 50 + percentage / 2])
        # include = (residuals > low) * (residuals < high)
        high = np.percentile(residuals_sq, percentage)
        include = residuals_sq < high
        # include = np.where(include)[0]
        # include.sort()
        cross_section = self.points[:][include]
        theta = self.theta[:][include]
        phi = self.phi[:][include]
        resids = residuals[include]
        if three_d:
            window.setWindowTitle('Cross-Section')
            scatter = ScatterPlot3d(cross_section, title='Cross Section Residuals',
                                    size=2, app=app, window=window)
            scatter.show()
        else:
            # plot the cross section residuals
            summary = VarSummary(180 / np.pi * phi, 180 / np.pi * theta,
                                 resids, suptitle='Cross Section Residuals',
                                 scatter=False, image_size=img_size,
                                 color_label='Residual Difference (um)', marker='.')
            summary.plot(inset=inset, margins=margins)
            plt.show()
            # plot the points within the residual proportion specified


    def plot_ommatidial_clusters(self, three_d=False, app=None, window=None, **kwargs):
        """Plot the ommatidial clusters, color coded by cluster.

        Parameters
        ----------
        three_d : bool, default=False
            Whether to use pyqtgraph to plot the cross section in 3D.
        """
        lbls = self.labels[:]
        lbls_set = np.arange(max(lbls) + 1)
        scrambled_lbls = np.random.permutation(lbls_set)
        new_lbls = scrambled_lbls[lbls]
        if three_d:
            window.setWindowTitle('Clusters')
            scatter = ScatterPlot3d(
                (self.points[:]), colorvals=new_lbls, cmap=(plt.cm.tab20),
                title='Ommatidial Clusters', app=app, window=window, size=1)
            scatter.show()
        else:
            theta_unwrapped = np.unwrap(self.theta[:])
            phi_unwrapped = np.unwrap(self.phi[:])
            colorvals = new_lbls
            order = np.argsort(self.radii[:])
            # summary = VarSummary(phi_unwrapped[order] * 180 / np.pi,
            #                      theta_unwrapped[order] * 180 / np.pi,
            #                      colorvals[order].astype(float),
            #                      suptitle=f"Ommatidial Clusters (N={len(lbls_set)})",
            #                      color_label='Color Label', cmap='tab20',
            #                      scatter=True, marker='.')
            # summary.plot(inset=True, margins=False)
            # plt.show()
            # plot and fill in the convex hull of each cluster and superimpose centers
            new_lbls_set = np.unique(new_lbls)
            order = np.argsort(new_lbls)
            theta, phi, new_lbls = theta_unwrapped[order], phi_unwrapped[order], new_lbls[order]
            changes = np.diff(new_lbls)
            changes = np.where(changes > 0)[0]
            starts = np.append([0], changes)
            ends = np.append(changes, [len(new_lbls)])
            # make figure and axis
            fig, ax = plt.subplots()
            colors = plt.cm.tab20(new_lbls_set/new_lbls_set.max())
            areas = []
            polys = []
            centers = []
            # use just those points within 50% residual distance of the fitted surface
            residuals = self.residual[:][order]
            
            residuals_sq = residuals ** 2
            # low, high = np.percentile(residuals, [50 - percentage / 2, 50 + percentage / 2])
            # include = (residuals > low) * (residuals < high)
            high = np.percentile(residuals_sq, 50)
            in_sheet = residuals_sq < high
            for lbl, start, end, color in zip(
                    new_lbls_set, starts, ends, colors):
                in_sheet_sub = in_sheet[start:end]
                if np.any(in_sheet_sub):
                    # get indices within start and end interval and in the cross section
                    th, ph = theta[start:end][in_sheet_sub], phi[start:end][in_sheet_sub]
                    pts = np.array([th, ph]).T
                    # eliminate outliers
                    dist_tree = spatial.KDTree(pts)
                    dists, inds = dist_tree.query(pts, k=3)
                    dists = dists[:, 1:].min(1)
                    # remove dists > 99%
                    thresh = np.percentile(dists, 99)
                    include = dists < thresh
                    if include.sum() > 2:
                        chull = spatial.ConvexHull(pts[include])
                        # Order = np.argsort(pts[include][:, 0])
                        areas += [chull.area]
                        # plot the chull, fill in with color, and superimpose center point
                        xs, ys = pts[include][chull.vertices].T
                        poly = ax.fill(ys, xs, color=color)[0]
                        polys += [poly]
                        centers += [[xs.mean(), ys.mean()]]
                        # dots += [ax.scatter(ys.mean(), xs.mean(), color='k', marker='.')]
            # remove polgons with an outlier area
            thresh = np.percentile(areas, 99)
            for area, poly in zip(areas, polys):
                if area > thresh:
                    # dot.set_visible(False)
                    poly.set_visible(False)
            # plot the centers 
            areas = np.array(areas)
            centers = np.array(centers)[areas <= thresh]
            ax.scatter(centers[:, 1], centers[:, 0], color='k', marker='.')
            # plot
            ax.set_aspect('equal')
            plt.tight_layout()
            plt.show()

    def plot_ommatidial_data(self, three_d=False, image_size=10000, scatter=False,
                             app=None, window=None, projected=False,
                             projected_radius=1e5, figsize=None, xmin=None, xmax=None,
                             ymin=None, ymax=None):
        """Plot the ommatidial data (lens area, IO angle, ...) in 2D histograms.

        Parameters
        ----------
        three_d : bool, default=False
            Whether to use pyqtgraph to plot the cross section in 3D.
        image_size : float, default=1e4
            The size of the 2d histogram used for plotting the variables.
        scatter : bool, default=bool
            Whether to plot the variable as a scatterplot as opposed to a 2D histogram
        app : QApplication
            The active QApplication used for the pyqtgraph plots.
        """
        data = self.ommatidial_data
        if projected:
            assert 'dx' in data.keys(), ("No smoothed direction vectors stored. "
                                         f"Try running {self.measure_interommatidia}")
            # get direction and position vectors
            x, y, z = data[['x_', 'y_', 'z_']].values.T
            pos_vectors = np.array([x, y, z]).T
            # get smoothed direction vectors
            dx, dy, dz = data[['dx', 'dy', 'dz']].values.T
            dir_vectors = np.array([dx, dy, dz]).T
            dir_vectors /= np.linalg.norm(dir_vectors, axis=1)[:, np.newaxis]
            # get raw direction vectors
            # dx, dy, dz = data[['dx_', 'dy_', 'dz_']].values.T
            # dir_vectors_raw = np.array([dx, dy, dz]).T
            # dir_vectors_raw /= np.linalg.norm(dir_vectors_raw, axis=1)[:, np.newaxis]
            proj_coords = project_coords(pos_vectors, dir_vectors, radius=projected_radius)
            polar = rectangular_to_spherical(proj_coords)
            # make two plots, side-by-side, of the original and projected polar coordinates
            original_polar = rectangular_to_spherical(pos_vectors)
            # fig, axes = plt.subplots(ncols=2, figsize=(6, 4))
            # give a unique value for each original polar coordinate
            # use the HSV colormap mapping hue to angle relative to the origin
            # in polar coordinates and V to radial distance
            fig, axes = plt.subplots(ncols=2, figsize=(6, 4), sharey=True, sharex=True)
            # give a unique value for each original polar coordinate
            # use the HSV colormap mapping hue to angle relative to the origin
            # in polar coordinates and V to radial distance
            original_polar -= original_polar.mean(0)
            angs = np.arctan2(original_polar[:, 0], original_polar[:, 1])
            dists = np.linalg.norm(original_polar[:, :2], axis=-1)
            hues = (angs + np.pi) / (2 * np.pi)
            vals = dists / dists.max()
            # sats = np.sqrt(1 - (hues**2 + vals**2))
            sats = np.ones(len(hues))
            hsv = np.array([hues, sats, vals]).T
            # hsv /= np.linalg.norm(hsv, axis=1)[:, np.newaxis]
            # get rgb colors
            cs = colors.hsv_to_rgb(hsv)
            # remove centers so both maps are 
            original_center = np.nanmean(original_polar, axis=0)
            original_polar -= original_center[np.newaxis]
            center = np.nanmean(polar, axis=0)
            polar -= center[np.newaxis]
            axes[0].scatter(original_polar[:, 1] * 180 / np.pi,
                            original_polar[:, 0] * 180 / np.pi , c=cs, marker='.',
                            alpha=.25)
            axes[1].scatter(polar[:, 1] * 180 / np.pi, polar[:, 0] * 180 / np.pi, c=cs,
                            alpha=.25, marker='.')
            [ax.set_aspect('equal') for ax in axes]
            sbn.despine(ax=axes[0], trim=True)
            sbn.despine(ax=axes[1], trim=True, left=True)
            # label axes
            axes[0].set_ylabel("Elevation ($\degree$)")
            axes[0].set_xlabel("Azimuth ($\degree$)")
            axes[0].set_title("Spherical")
            axes[1].set_xlabel("Azimuth ($\degree$)")
            axes[1].set_title("Projected")
            # super title
            fig.suptitle("World Referenced Coordinates")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()
            # original_polar -= original_polar.mean(0)
            # angs = np.arctan2(original_polar[:, 0], original_polar[:, 1])
            # dists = np.linalg.norm(original_polar[:, :2], axis=-1)
            # hues = (angs + np.pi) / (2 * np.pi)
            # vals = dists / dists.max()
            # # sats = np.sqrt(1 - (hues**2 + vals**2))
            # sats = np.ones(len(hues))
            # hsv = np.array([hues, sats, vals]).T
            # # hsv /= np.linalg.norm(hsv, axis=1)[:, np.newaxis]
            # # get rgb colors
            # cs = colors.hsv_to_rgb(hsv)
            # axes[0].scatter(original_polar[:, 1], original_polar[:, 0], c=cs)
            # axes[1].scatter(polar[:, 1], polar[:, 0], c=cs, alpha=.25,
            #                 marker='.')
            # [ax.set_aspect('equal') for ax in axes]
            # plt.tight_layout()
            # plt.show()
            # get important coordinate data
            theta, phi = polar.T[:2]
            x, y, z = pos_vectors.T
        else:
            if 'x_' in data.keys():
                x, y, z = data[['x_', 'y_', 'z_']].values.T
                pts = np.array([x, y, z]).T
                polar = rectangular_to_spherical(pts)
                theta, phi, radii = polar.T
            else:
                theta, phi, x, y, z = np.copy(data[['theta', 'phi', 'x', 'y', 'z']].values.T)

        # theta *= 180./ np.pi
        # phi *= 180./ np.pi
        com = stats.circmean(phi)
        # if com > np.pi / 2:
        #     phi[phi < 0] = phi[(phi < 0)] + 2 * np.pi
        pts = np.array([x, y, z]).T
        vars_to_plot = ['size', 'lens_area', 'lens_diameter', 'lens_diameter_adj',
                        'spherical_IOA', 'skewness', 'cross_section_area', 'cross_section_height']
        var_ranges = ['size', 'lens_area', 'lens_diameter', 'lens_diameter',
                      'spherical_IOA', 'skewness', 'cross_section_area', 'cross_section_height']
        # size = data['size'].values
        # low_size, high_size = np.percentile(size, [5, 95])
        # include = (size > low_size) * (size < high_size)
        # scatters_3d = []
        for num, (var, var_range) in enumerate(zip(vars_to_plot, var_ranges)):
            colorvals = data[var].values
            range_vals = data[var_range].values
            no_nans = np.isnan(colorvals) == False
            no_nans_theta = np.isnan(theta) == False
            no_nans_phi = np.isnan(phi) == False
            no_nans = no_nans * no_nans_theta * no_nans_phi
            # figure out reasonable vmin and vmax
            vmin, vmax = np.percentile(range_vals[no_nans], [0, 99])
            # use io angle as the marker size
            io_angles = .8 * .5 * data.spherical_IOA.values
            no_nans *= np.isnan(io_angles) == False
            # remove nans
            theta_unwrapped = np.unwrap(theta[no_nans])
            phi_unwrapped = np.unwrap(phi[no_nans])
            io_angles = io_angles[no_nans]
            # sort theta and phi
            order = np.argsort(phi_unwrapped)
            if figsize is not None:
                fig = plt.figure(figsize=figsize)
            else:
                fig = None
            summary = VarSummary(phi_unwrapped[order] * 180 / np.pi,
                                 theta_unwrapped[order] * 180 / np.pi,
                                 colorvals[no_nans][order].astype(float),
                                 suptitle=f"{var} (N={no_nans.sum()})",
                                 color_label=var,
                                 image_size=image_size,
                                 scatter=scatter, vmin=vmin, vmax=vmax,
                                 marker='.', marker_sizes=io_angles[order])
            if xmin is None:
                xmin = phi_unwrapped.min() * 180 / np.pi
            if xmax is None:
                xmax=phi_unwrapped.max() * 180 / np.pi
            if ymin is None:
                ymin=theta_unwrapped.min() * 180 / np.pi
            if ymax is None:
                ymax=theta_unwrapped.max() * 180 / np.pi
            summary.plot(fig=fig, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
            # plt.tight_layout()
        plt.show()

    def plot_ommatidial_data_3d(self, app=None, window=None, main_window=None, **kwargs):
        """Plot the ommatidial data (lens area, IO angle, ...) in 3D scatterplots.

        Parameters
        ----------
        app : QApplication
            The active QApplication used for the pyqtgraph plots.
        window : QWidget
            The window in which to render the ScatterPlot3d
        main_window : QWidget
            The main window used as parent for the data selection box.
        """
        data = self.ommatidial_data
        area = np.copy(data['lens_area'])
        diameter = 2 * np.sqrt(area / np.pi)
        data['lens_diameter'] = diameter
        theta, phi, x, y, z = np.copy(data[['theta', 'phi', 'x', 'y', 'z']].values.T)
        theta *= 180./ np.pi
        phi *= 180./ np.pi
        com = stats.circmean(phi)
        pts = np.array([x, y, z]).T
        vars_to_plot = ['size', 'lens_area', 'lens_diameter',
                        'spherical_IOA', 'skewness',
                        'cross_section_area', 'cross_section_height']
        # make a function to call for each variable
        def plot_variable(var):
            colorvals = data[var].values
            no_nans = np.isnan(colorvals) == False
            window.setWindowTitle(var)
            scatter = ScatterPlot3d( 
               (pts[no_nans]), colorvals=(colorvals[no_nans]), title=var, size=5,
                app=app, window=window)
        # make a small window for the radio buttons
        choice_window = QDialog(parent=main_window)
        # make a radio button box for choosing the variable to plot
        box_layout = QVBoxLayout()
        buttons = []
        for num, (var) in enumerate(vars_to_plot):
            button = QRadioButton(var.replace("_", " "))
            button.clicked.connect(partial(plot_variable, var=var))
            button.released.connect(window.clear)
            print(var)
            box_layout.addWidget(button)
            buttons += [button]
        # add buttons layout to the choice window
        choice_window.setLayout(box_layout)
        choice_window.show()

    def plot_interommatidial_data(self, three_d=False, valmin=None, valmax=None, scatter=False,
                                  app=None, window=None, main_window=None, image_size=1e5,
                                  projected=False, projected_radius=1e5, xmin=None, xmax=None,
                                  figsize=None, ymin=None, ymax=None, convex=False):
        """Plot the interommatidial data

        Parameters
        ----------
        three_d : bool, default=False
            Whether to use pyqtgraph to plot the cross section in 3D.
        valmin : float, default=None
            Minimum value for the 2D histograms and scatterplot colormaps. If None,
            pyplot use the default minimum.
        valmax : float, default=None
            Maximum value for the 2D histograms and scatterplot colormaps. If None,
            pyplot use the default max.
        scatter : bool, default=False
            Whether to plot the interommatidia as a scatter plot as opposed to a 2D 
            histogram.
        image_size : int, default=1e5
            The size of the image used for 2d raster plots.
        projected : bool, default=False
            Whether to use coordinates projected onto an encompassing sphere (a.k.a. 
            world referenced coordinates). 
        projected_radius : float, default=1e5
            The radius of the sphere used for projecting the world referenced 
            coordinates.
        convex : bool, default=False
            Whether to assume the eye is convex or concave (default).
        """
        # load the interommatidial data
        interommatidial_data = self.interommatidial_data
        orientation = np.copy(interommatidial_data.orientation.values)
        orientation[orientation < 0] = orientation[orientation < 0] + np.pi
        orientation -= np.pi/2
        # order = np.argsort(orientation)
        # orientation[order] = np.unwrap(orientation[order], period=np.pi)
        CMAP = 'Greys'
        # convert to degrees
        angles = interommatidial_data.angle_total.values * 180 / np.pi
        no_nans = np.isnan(angles) == False
        # use default values unless provided
        if valmax is None:
            vmax = np.round(np.percentile(angles[no_nans], 99))
        else:
            vmax = valmax
        if valmin is None:
            vmin = 0
        else:
            vmin = valmin
        # get coordinate locations
        x1, y1, z1 = interommatidial_data[['pt1_x', 'pt1_y', 'pt1_z']].values.T
        x2, y2, z2 = interommatidial_data[['pt2_x', 'pt2_y', 'pt2_z']].values.T
        x, y, z = (x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2
        pos_vectors = np.array([x, y, z]).T
        # get mean direction vectors, raw
        dx1, dy1, dz1 = interommatidial_data[['pt1_dx', 'pt1_dy', 'pt1_dz']].values.T
        dx2, dy2, dz2 = interommatidial_data[['pt2_dx', 'pt2_dy', 'pt2_dz']].values.T
        dx, dy, dz = (dx1 + dx2) / 2, (dy1 + dy2) / 2, (dz1 + dz2) / 2
        dir_vectors_raw = np.array([dx, dy, dz]).T
        dir_vectors_raw /= np.linalg.norm(dir_vectors_raw, axis=1)[:, np.newaxis]
        # get mean direction vectors, smoothed
        dx1, dy1, dz1 = interommatidial_data[['pt1_dx_', 'pt1_dy_', 'pt1_dz_']].values.T
        dx2, dy2, dz2 = interommatidial_data[['pt2_dx_', 'pt2_dy_', 'pt2_dz_']].values.T
        dx, dy, dz = (dx1 + dx2) / 2, (dy1 + dy2) / 2, (dz1 + dz2) / 2
        dir_vectors = np.array([dx, dy, dz]).T
        dir_vectors /= np.linalg.norm(dir_vectors, axis=1)[:, np.newaxis]
        if projected:
            # project coordinates onto a sphere centered at (0, 0, 0) with a radius
            # defined by projected_radius
            # proj_coords = []
            # for p_vector, d_vector in zip(pos_vectors, dir_vectors):
            #     const = np.dot(p_vector, d_vector)
            #     diff = const ** 2 - np.linalg.norm(p_vector) ** 2 + projected_radius ** 2
            #     if diff >= 0:
            #         dists = np.array([-const - np.sqrt(diff),
            #                           -const + np.sqrt(diff)])
            #         dist = dists[np.argmin(abs(dists))]
            #         proj_pt = p_vector + d_vector * dist
            #     else:
            #         proj_pt = np.empty(3)
            #         proj_pt[:] = np.nan
            #     proj_coords += [proj_pta]
            # proj_coords = np.array(proj_coords)
            proj_coords = project_coords(pos_vectors, dir_vectors, radius=projected_radius,
                                         convex=convex)
            proj_coords_raw = project_coords(pos_vectors, dir_vectors_raw,
                                             radius=projected_radius, convex=convex)
            polar = rectangular_to_spherical(proj_coords)
            theta, phi, radii = polar.T            
            # easy fix for any obvious outliers
            for vals in [phi, theta]:
                order = np.argsort(vals)
                if np.any(np.diff(vals[order]) > np.pi/2):
                    vals[order] = np.unwrap(vals[order], period=np.pi)
            polar_raw = rectangular_to_spherical(proj_coords_raw)
            theta_r, phi_r, radii_r = polar_raw.T            
            # easy fix for any obvious outliers
            for vals in [phi_r, theta_r]:
                order = np.argsort(vals)
                if np.any(np.diff(vals[order]) > np.pi/2):
                    vals[order] = np.unwrap(vals[order], period=np.pi)
            # make two plots, side-by-side, of the original and projected polar coordinates
            original_polar = rectangular_to_spherical(pos_vectors)
            fig, axes = plt.subplots(ncols=3, figsize=(10, 5), sharey=True, sharex=True)
            # give a unique value for each original polar coordinate
            # use the HSV colormap mapping hue to angle relative to the origin
            # in polar coordinates and V to radial distance
            original_polar -= original_polar.mean(0)
            angs = np.arctan2(original_polar[:, 0], original_polar[:, 1])
            dists = np.linalg.norm(original_polar[:, :2], axis=-1)
            hues = (angs + np.pi) / (2 * np.pi)
            vals = dists / dists.max()
            # sats = np.sqrt(1 - (hues**2 + vals**2))
            sats = np.ones(len(hues))
            hsv = np.array([hues, sats, vals]).T
            # hsv /= np.linalg.norm(hsv, axis=1)[:, np.newaxis]
            # get rgb colors
            cs = colors.hsv_to_rgb(hsv)
            # remove centers so both maps are 
            original_center = np.nanmean(original_polar, axis=0)
            original_polar -= original_center[np.newaxis]
            center = np.nanmean(polar, axis=0)
            polar -= center[np.newaxis]
            raw_center = np.nanmean(polar_raw, axis=0)
            polar_raw -= raw_center[np.newaxis]
            axes[0].scatter(original_polar[:, 1] * 180 / np.pi,
                            original_polar[:, 0] * 180 / np.pi , c=cs, marker='.',
                            alpha=.25, edgecolors='none')
            # plot raw projected coordinates
            axes[1].scatter(polar_raw[:, 1] * 180 / np.pi, polar_raw[:, 0] * 180 / np.pi, c=cs,
                            alpha=.25, marker='.', edgecolors='none')
            # plot smoothed projected coordinates
            axes[2].scatter(polar[:, 1] * 180 / np.pi, polar[:, 0] * 180 / np.pi, c=cs,
                            alpha=.25, marker='.', edgecolors='none')
            # format axes
            [ax.set_aspect('equal') for ax in axes]
            # set reasonable x and y limits
            x_min, x_max = -120, 120
            [ax.set_xlim(x_min, x_max) for ax in axes]
            [ax.set_ylim(x_min, x_max) for ax in axes]
            sbn.despine(ax=axes[0], trim=True)
            sbn.despine(ax=axes[1], trim=True, left=True)
            sbn.despine(ax=axes[2], trim=True, left=True)
            # label axes
            axes[0].set_ylabel("Elevation ($\degree$)")
            axes[0].set_xlabel("Azimuth ($\degree$)")
            axes[0].set_title("Spherical")
            axes[1].set_xlabel("Azimuth ($\degree$)")
            axes[1].set_title("Projected")
            axes[2].set_xlabel("Azimuth ($\degree$)")
            axes[2].set_title("Projected and Smoothed")
            # super title
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()
            # todo: calculate the projected and polar vertical and horizontal FOVs
            FOV_h = np.diff(np.percentile(theta, [2.5, 97.5]))[0]
            FOV_v = np.diff(np.percentile(phi, [2.5, 97.5])) [0]
            th_proj, ph_proj, _ = polar_raw.T
            no_nans = np.isnan(th_proj) == False
            FOV_h_proj = np.diff(np.percentile(th_proj[no_nans], [2.5, 97.5]))[0]
            FOV_v_proj = np.diff(np.percentile(ph_proj[no_nans], [2.5, 97.5]))[0]
            # todo: calculate the ratio of vertical to horizontal radii of curvature
            # eye length = 
            vert_to_hori = (np.sin(FOV_h/2) / np.sin(FOV_v/2)) * (np.sin(FOV_v_proj/2) / np.sin(FOV_h_proj/2))
            print(f"The vertical radius is {vert_to_hori:.3f} times the horizontal radius using FOV estimates.")
            # breakpoint()
        else:
            # use spherical coordinates
            polar = rectangular_to_spherical(pos_vectors)
            theta, phi, radii = polar.T
        # swap horizontal and vertical angles for the plots below
        # angles_v = np.copy(interommatidial_data.angle_v.values)
        # interommatidial_data.angle_v = np.copy(interommatidial_data.angle_h.values)
        # interommatidial_data.angle_h = angles_v
        # cluster orientations into 3 groups
        clusterer = cluster.KMeans(
            3, init=np.array([0, np.pi/2, np.pi])[:, np.newaxis], n_init=1).fit(
                orientation[:, np.newaxis])
        # define groups as cluster centers +/- 15 degs
        pad = 15 * np.pi / 180
        groups = np.zeros(len(orientation), dtype=float)
        groups.fill(np.nan)
        for group, center in enumerate(clusterer.cluster_centers_):
            diff = abs(orientation - center)
            groups[diff < pad] = group
        group_set = np.unique(groups)
        no_nans = np.isnan(group_set) == False
        group_set = group_set[no_nans]
        group_oris = clusterer.cluster_centers_
        # make bins using the provided value ranges
        BINS = [np.linspace(-90, 90), np.linspace(vmin, vmax, 50)]
        # plot the vertical and horizontal components of the interommatidial angles
        # by their orientation
        fig = plt.figure(figsize=(8, 4))
        gridspec = fig.add_gridspec(ncols=4, nrows=1, width_ratios=[4, 4, 4, 1])
        axes = [fig.add_subplot(gridspec[0, 0]), fig.add_subplot(gridspec[0, 1])]
        # fig, axes = plt.subplots(ncols=2)
        no_nans_ori = np.isnan(orientation) == False
        no_nans_angle_v = np.isnan(interommatidial_data.angle_v) == False 
        no_nans_angle_h = np.isnan(interommatidial_data.angle_h) == False 
        no_nans = np.array(no_nans_ori & no_nans_angle_v & no_nans_angle_h)
        axes[0].hist2d(orientation[no_nans] * 180 / np.pi,
                       interommatidial_data.angle_h.values[no_nans] * 180 / np.pi,
                       color='k', bins=BINS,  cmap=CMAP, edgecolor='none')
        axes[0].set_title('Horizontal')
        axes[0].set_xlabel('Orientation ($\\degree$)')
        axes[0].set_xticks([-60, 0, 60])
        axes[0].set_ylabel("Angle ($\\degree$)")
        axes[1].hist2d((orientation[no_nans] * 180 / np.pi),
                       (interommatidial_data.angle_v[no_nans] * 180 / np.pi),
                       color='k', bins=BINS, cmap=CMAP, edgecolor='none')
        axes[1].set_title('Vertical')
        axes[1].set_xlabel('Orientation ($\\degree$)')
        axes[1].set_xticks([-60, 0, 60])
        # todo: plot the medians +/- IQR bars per elevation and azimuth
        for vals, ax, variable in zip(
                [interommatidial_data.angle_h, interommatidial_data.angle_v],
                axes, ["Horizontal IO Component", "Vertical IO Component"]):
            print()
            print()
            print(variable+ ":")
            for group_num, group_ori, color in zip(
                    group_set, group_oris, [red, green, blue]):
                # grab subset of the data
                ori = 180 / np.pi * group_ori
                # get values of this group
                inds = groups == group_num
                sub_vals = 180 / np.pi * vals[inds].values
                # remove nans
                no_nans_sub = np.isnan(sub_vals) == False
                sub_vals = sub_vals[no_nans_sub]
                low, median, high = np.percentile(sub_vals, [25, 50, 75])
                # bootstrap sub_vals to get 99% CI of the median
                indices = np.arange(len(sub_vals))
                indices_random = np.random.choice(indices, (len(sub_vals), 10000), replace=True)
                meds_random = sub_vals[indices_random]
                meds_random = np.median(meds_random, axis=1)
                low_ci, high_ci = np.percentile(meds_random, [.5, 99.5])
                # make a label for the plotted data per  group
                label = f"N={len(sub_vals)}\nm={median: .2f}\nIQR=[{low: .2f}, {high: .2f}]\nCI=({low_ci: .2f}, {high_ci: .2f})"
                print()
                print(f"group # {group_num}, at {np.round(ori, 2)}$\degree$")
                print(label)
                # plot median +/- IQR
                ax.plot([ori, ori], [low, high], color=color, alpha=.5)
                ax.scatter([ori, ori], [low_ci, high_ci], color=color, marker='_')
                ax.plot(ori, median, marker='.', color=color, label=label)
        [ax.legend(prop={'size': 6}) for ax in axes]
        axes[1].set_yticks([])
        sbn.despine(ax = axes[0], trim=True)
        sbn.despine(ax = axes[1], trim=True, left=True)
        # now the total IO angle
        img_ax = fig.add_subplot(gridspec[(0, 2)])
        colorbar_ax = fig.add_subplot(gridspec[(0, 3)])
        img_ax.hist2d(
            (orientation[no_nans] * 180 / np.pi), angles[no_nans], color='k',
            bins=BINS, cmap=CMAP, edgecolor='none')
        vals = angles
        # plot the medians +/- IQR bars per elevation and azimuth
        for group_num, group_ori, color in zip(
                group_set, group_oris, [red, green, blue]):
            # grab subset of the data
            # todo: get values and plot in the same loop
            ori = 180 / np.pi * group_ori
            # get values of this group
            inds = groups == group_num
            sub_vals = vals[inds]
            # remove nans
            no_nans = np.isnan(sub_vals) == False
            sub_vals = sub_vals[no_nans]
            low, median, high = np.percentile(sub_vals, [25, 50, 75])
            # bootstrap sub_vals to get 99% CI of the median
            indices = np.arange(sum(no_nans))
            indices_random = np.random.choice(indices, (len(sub_vals), 10000), replace=True)
            meds_random = sub_vals[indices_random]
            meds_random = np.median(meds_random, axis=1)
            low_ci, high_ci = np.percentile(meds_random, [.5, 99.5])
            # make a label for the plotted data per  group
            label = f"N={len(sub_vals)}\nm={median: .2f}\nIQR=[{low: .2f}, {high: .2f}]\nCI=({low_ci: .2f}, {high_ci: .2f})"
            # plot median +/- IQR
            img_ax.plot([ori, ori], [low, high], color=color, alpha=.5)
            img_ax.scatter([ori, ori], [low_ci, high_ci], color=color, marker='_')
            img_ax.plot(ori, median, marker='.', color=color, label=label)
        colorbar_ax.hist(angles, bins=(BINS[1]), orientation='horizontal', color='k', alpha=1)
        colorbar_ax.set_ylim(0, vmax)
        colorbar_ax.set_yticks([])
        sbn.despine(ax=colorbar_ax, trim=True, left=True)
        img_ax.legend(prop={'size': 6})
        img_ax.set_title('Total IO Angle')
        img_ax.set_xlabel('Orientation ($\\degree$)')
        img_ax.set_yticks([])
        img_ax.set_ylim(0, vmax)
        img_ax.set_xticks([-60, 0, 60])
        sbn.despine(ax=img_ax, left=True, trim=True)
        plt.tight_layout()
        plt.show()
        # todo: get measurements of the horizontal and vertical IO angle components
        # find the horizontal group using the smallest 
        group_nums = np.arange(3)
        horizontal_group = group_nums[np.argmin(abs(group_oris))]
        diagonal_groups = np.delete(group_nums, horizontal_group)
        horizontal_group = groups == horizontal_group
        diagonal_group = np.in1d(groups, diagonal_groups)
        # calculate the horizontal and vertical IO angles
        angle_h = 180./ np.pi * interommatidial_data['angle_h'].values
        angle_v = 180./ np.pi * interommatidial_data['angle_v'].values
        # IO horizontal = hor. angle of hor. pairs + 1/2 of the hor. angle of diagonal pairs
        io_horizontal = np.concatenate([.5 * angle_h[horizontal_group], angle_h[diagonal_group]])
        # IO vertical   = 2/sqrt(3) * vertical angle of diagonal pairs
        io_vertical = angle_v[diagonal_group]
        # do the same but for interommatidial distances
        # print the median 
        for vals, lbl in zip([io_horizontal, io_vertical], ['Horizontal', 'Vertical']):
            print(f"{lbl} IO components:")
            low, med, high = np.percentile(vals, [25, 50, 75])
            indices = np.arange(len(vals))
            indices_random = np.random.choice(indices, (len(vals), 10000), replace=True)
            meds_random = vals[indices_random]
            meds_random = np.median(meds_random, axis=1)
            low_ci, high_ci = np.percentile(meds_random, [.5, 99.5])
            label = f"N={len(vals)}\nm={med: .2f}\nIQR=[{low: .2f}, {high: .2f}]\nCI=({low_ci: .2f}, {high_ci: .2f})"
            print(label)
        # plot the interommatidial diameters by their orientation like the total IO angles
        pts1 = interommatidial_data[['pt1_x', 'pt1_y', 'pt1_z']].values
        pts2 = interommatidial_data[['pt2_x', 'pt2_y', 'pt2_z']].values
        diams = np.linalg.norm(pts2 - pts1, axis=-1)
        # get radius of intersection using the total IO angles and their corresponding diameters
        radii = diams / (angles * np.pi / 180)
        # todo: remove outliers
        for vals, lbl in zip([diams, radii], ['Distance', 'Radius']):
            no_nans = np.isnan(vals) == False
            no_nans = no_nans * (np.isinf(vals) == False)
            vals = np.copy(vals)[no_nans]
            maxval = np.percentile(vals, 95)
            BINS = [np.linspace(-90, 90), np.linspace(0, maxval, 50)]
            # BINS = [np.linspace(-90, 90), np.linspace(0, 40, 50)]
            # plot one 2D histogram
            fig, axes = plt.subplots(ncols=2, figsize=(3.08, 4), gridspec_kw={'width_ratios':[4, 1]})
            axes[0].hist2d(orientation[no_nans] * 180 / np.pi, vals, color='k', 
                           bins=BINS, cmap=CMAP, edgecolor='none')
            axes[0].set_xlabel('Orientation ($\\degree$)')
            axes[0].set_xticks([-60, 0, 60])
            axes[0].set_ylabel(f"{lbl} ($\mu$m)")
            # plot stats per orientation group
            img_ax = axes[0]
            for group_num, group_ori, color in zip(
                    group_set, group_oris, [red, green, blue]):
                # grab subset of the data
                # todo: get values and plot in the same loop
                ori = 180 / np.pi * group_ori
                # get values of this group
                inds = groups[no_nans] == group_num
                sub_vals = vals[inds]
                # remove nans
                # no_nans = np.isnan(sub_vals) == False
                # sub_vals = sub_vals[no_nans]
                low, median, high = np.percentile(sub_vals, [25, 50, 75])
                # bootstrap sub_vals to get 99% CI of the median
                indices = np.arange(len(sub_vals))
                indices_random = np.random.choice(indices, (len(sub_vals), 10000), replace=True)
                meds_random = sub_vals[indices_random]
                meds_random = np.median(meds_random, axis=1)
                low_ci, high_ci = np.percentile(meds_random, [.5, 99.5])
                # make a label for the plotted data per  group
                label = f"m={median: .2f}\nIQR=[{low: .2f}, {high: .2f}]\nCI=({low_ci: .2f}, {high_ci: .2f})"
                # plot median +/- IQR
                img_ax.plot([ori, ori], [low, high], color=color, alpha=.5)
                img_ax.scatter([ori, ori], [low_ci, high_ci], color=color, marker='_')
                img_ax.plot(ori, median, marker='.', color=color, label=label)
            img_ax.legend(fontsize='xx-small')
            sbn.despine(ax=axes[0], trim=True)
            # plot the flattened histogram of diameters
            no_nans = np.isnan(vals) == False
            axes[1].hist(vals[no_nans], bins=BINS[1], orientation='horizontal', color='k', alpha=1)
            ymin, ymax = axes[0].get_ylim()
            axes[1].set_ylim(ymin, ymax)
            axes[1].set_yticks([])
            sbn.despine(ax=axes[1], trim=True, left=True)
            plt.tight_layout()
            plt.show()
        # IO horizontal = hor. angle of hor. pairs + 1/2 of the hor. angle of diagonal pairs
        # diams_h = np.concatenate([diams[diagonal_group]/2., diams[horizontal_group]/2])
        # IO vertical   = 2/sqrt(3) * vertical angle of diagonal pairs
        # diams_v = np.sqrt(3) * diams[diagonal_group] / 2.
        # todo: measure the intersection radii assuming an isosceles triangle
        # based on that, each radius is the diameters / IO angles
        # radius_h = diams_h / io_horizontal
        # radius_v = diams_v / io_vertical
        # do the same but for interommatidial distances
        # print the median 
        for vals, lbl in zip(
            [diams, radii], 
            ['Diameters', 'Radius']):
            print(f"{lbl}:")
            no_nans = np.isnan(vals) == False
            vals = np.copy(vals[no_nans])
            low, med, high = np.percentile(vals, [25, 50, 75])
            indices = np.arange(len(vals))
            indices_random = np.random.choice(indices, (len(vals), 10000), replace=True)
            meds_random = vals[indices_random]
            meds_random = np.median(meds_random, axis=1)
            low_ci, high_ci = np.percentile(meds_random, [.5, 99.5])
            label = f"m={med: .2f}\nIQR=[{low: .2f}, {high: .2f}]\nCI=({low_ci: .2f}, {high_ci: .2f})"
            print(label)
        # for each interommatidial angle component and total:
        # no_nans = (np.isnan(theta) == False) * (np.isnan(phi) == False)
        # for angs, title in zip([interommatidial_data.angle_h * 180 / np.pi,
        #                         interommatidial_data.angle_v * 180 / np.pi, angles],
        #                        ['Horizontal IO Angle', 'Vertical IO Angle', 'Total IO Angle']):
        #     # go through orientation groups and plot
        #     order = np.argsort(group_set)
        #     for group, ori in zip(group_set[:3], group_oris * 180 / np.pi):
        #         inds = groups == group
        #         include = inds * no_nans
        #         include *= np.isnan(angs) == False
        #         summary = VarSummary(
        #             phi[include] * 180 / np.pi, theta[include] * 180 / np.pi,
        #             angs[include], color_label=title,
        #             suptitle=(title + f" {np.round(ori)}$\\degree$ (N={include.sum()})"),
        #             scatter=scatter, vmin=vmin, vmax=vmax, image_size=image_size) 
        #         summary.plot()
        # plt.show()
        # now, using the oval eye model, plot the horizontal and vertical components
        # the horizontal component is half the horizontal component of horizontal IO pairs
        # use half the horizontal component of group 0 pairs
        group0_horizontal = interommatidial_data.angle_h[groups == 0]/2
        group1_horizontal = interommatidial_data.angle_h[groups == 1]
        group2_horizontal = interommatidial_data.angle_h[groups == 2]
        theta_horizontal = []
        phi_horizontal = []
        for vals, storage in zip([theta, phi], [theta_horizontal, phi_horizontal]):
            for group in group_set:
                storage += [vals[groups == group]]
        theta_horizontal = np.concatenate(theta_horizontal)
        phi_horizontal = np.concatenate(phi_horizontal)
        io_horizontal = np.concatenate([group0_horizontal, group1_horizontal, group2_horizontal])
        # the vertical component is equal to the vertical component of diagonal IO pairs
        group1_vertical = interommatidial_data.angle_v[groups == 1]
        group2_vertical = interommatidial_data.angle_v[groups == 2]
        theta_vertical = []
        phi_vertical = []
        for vals, storage in zip([theta, phi], [theta_vertical, phi_vertical]):
            for group in [1, 2]:
                storage += [vals[groups == group]]
        theta_vertical = np.concatenate(theta_vertical)
        phi_vertical = np.concatenate(phi_vertical)
        io_vertical = np.concatenate([group1_vertical, group2_vertical])
        # now make VarSummary of each
        thetas = np.concatenate([theta_horizontal, theta_vertical])
        phis = np.concatenate([phi_horizontal, phi_vertical])
        no_nans = np.isnan(phis) == False
        breakpoint()
        if any([xmin is None, xmax is None]):
            xmin, xmax = 180 / np.pi * np.percentile(phis[no_nans], [0, 100])
            x_range = xmax - xmin
            xpad = .05 * x_range
            xmin -= xpad
            xmax += xpad
        if any([ymin is None, ymax is None]):
            ymin, ymax = 180 / np.pi * np.percentile(thetas[no_nans], [0, 100])
            y_range = ymax - ymin
            ypad = .05 * y_range
            ymin -= ypad
            ymax += ypad
        for th, ph, colorvals, title in zip(
                [theta_horizontal, theta_vertical],
                [phi_horizontal, phi_vertical],
                [io_horizontal, io_vertical],
                ["$\Delta\phi_H$ ($\degree$)", "$\Delta\phi_V$ ($\degree$)"]):
            no_nans = np.isnan(th) == False
            no_nans *= np.isnan(ph) == False
            no_nans *= np.isnan(colorvals) == False
            summary = VarSummary(
                ph[no_nans] * 180 / np.pi, th[no_nans] * 180 / np.pi,
                colorvals[no_nans] * 180 / np.pi, color_label=title,
                suptitle=title + f" (N={no_nans.sum()})", scatter=scatter,
                image_size=image_size, vmin=vmin, vmax=vmax)
            fig = None
            if figsize is not None:
                fig = plt.figure(figsize=figsize)
            summary.plot(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, fig=fig,
                         vmin=vmin, vmax=vmax)
        plt.show()


    def plot_interommatidial_data_3d(self, app=None, window=None, main_window=None,
                                     valmin=0, valmax=None, **kwargs):
        """Plot the interommatidial data in 3D scatterplots.

        Parameters
        ----------
        app : QApplication
            The active QApplication used for the pyqtgraph plots.
        window : QWidget
            The window in which to render the ScatterPlot3d
        main_window : QWidget
            The main window used as parent for the data selection box.
        """
        interommatidial_data = self.interommatidial_data
        orientation = abs(interommatidial_data.orientation)
        CMAP = 'Greys'
        # convert important angles to degrees
        interommatidial_data.angle_total *= 180. / np.pi
        interommatidial_data.angle_v *= 180. / np.pi
        interommatidial_data.angle_h *= 180. / np.pi
        # only use non-nan values
        no_nans = np.isnan(interommatidial_data.angle_total) == False
        # use default values unless provided
        if valmax is None:
            vmax = np.round(np.percentile(interommatidial_data.angle_total[no_nans], 99))
        else:
            vmax = valmax
        vmin = valmin
        # get coordinate locations
        x1, y1, z1 = interommatidial_data[['pt1_x', 'pt1_y', 'pt1_z']].values.T
        x2, y2, z2 = interommatidial_data[['pt2_x', 'pt2_y', 'pt2_z']].values.T
        x, y, z = (x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2
        pts = np.array([x, y, z]).T
        # make bins using the provided value ranges
        BINS = [np.linspace(0, 180), np.linspace(vmin, vmax, 50)]
        vars_to_plot = ['angle_total', 'angle_v', 'angle_h']
        # grab the coordinates for IO pairs and split into groups based on their orientation
        th1, ph1 = interommatidial_data[['pt1_th', 'pt1_ph']].values.T
        th2, ph2 = interommatidial_data[['pt2_th', 'pt2_ph']].values.T
        th1, ph1 = np.unwrap(th1), np.unwrap(ph1)
        th2, ph2 = np.unwrap(th2), np.unwrap(ph2)
        th, ph = (th1 + th2) / 2, (ph1 + ph2) / 2
        oris = interommatidial_data[['ori_dx', 'ori_dy']].values
        oris_dx, oris_dy = oris.T
        # Assume 6 clusters of orientations
        # swap vertical and horizontal io angles
        oris = interommatidial_data[['ori_dx', 'ori_dy']].values
        oris_dx, oris_dy = oris.T
        # get complementary angle of any orientations with dy < 0
        oris[oris[:, 0] < 0] = -oris[oris[:, 0] < 0]
        # Assume 3 clusters of orientations
        clusterer = cluster.KMeans(3).fit(oris)
        group_means = clusterer.cluster_centers_
        # calculate the orientations
        group_oris = np.arctan2(group_means[:, 1], group_means[:, 0])
        groups = clusterer.labels_
        group_set = np.unique(groups)
        # make a function to call for each variable
        def update_plot():
            if self.group is not None:
                include = groups == self.group
            else:
                include = np.ones(len(interommatidial_data), dtype=bool)
            colorvals = interommatidial_data[self.var][include].values
            no_nans = np.isnan(colorvals) == False
            scatter = ScatterPlot3d(
                pts[include][no_nans], colorvals=(colorvals[no_nans]),
                title=self.var, size=10,
                app=app, window=window)
        def set_variable(var, window):
            self.var = var
            window.setWindowTitle(var)
            update_plot()
        # add option to select one of the 3 orientation groups or select all
        self.group = None
        self.var = vars_to_plot[0]
        def set_group(group_num, window):
            self.group = group_num
            inds = groups == self.group
            dx, dy = oris_dx[inds], oris_dy[inds]
            group_orientation = group_oris[group_set == self.group]
            group_orientation *= 180 / np.pi
            window.setWindowTitle(f"{np.round(group_orientation, 2)} degs")
            update_plot()
        # make a small window for the radio buttons
        choice_window = QDialog(parent=main_window)
        choice_window.resize(300, 300)
        # group
        general_layout = QHBoxLayout()
        groups_layout = QVBoxLayout()
        variables_layout = QVBoxLayout()
        buttons_group = []
        group_group = QButtonGroup()
        group_variables = QButtonGroup()
        for group in set(groups):
            button = QRadioButton(str(group))
            group_group.addButton(button)
            button.clicked.connect(partial(
                set_group, group_num=group, window=choice_window))
            button.released.connect(window.clear)
            print(group)
            groups_layout.addWidget(button)
            buttons_group += [button]
        button = QRadioButton('All')
        group_group.addButton(button)
        button.clicked.connect(partial(set_group, group_num=None, window=choice_window))
        button.released.connect(window.clear)
        groups_layout.addWidget(button)
        buttons_group += [button]
        general_layout.addLayout(groups_layout)
        # make a radio button box for choosing the variable to plot
        buttons_variable = []
        for num, (var) in enumerate(vars_to_plot):
            button = QRadioButton(var.replace("_", " "))
            group_variables.addButton(button)
            button.clicked.connect(partial(set_variable, var=var, window=window))
            button.released.connect(window.clear)
            print(var)
            variables_layout.addWidget(button)
            buttons_variable += [button]
        general_layout.addLayout(variables_layout)
        # add buttons layout to the choice window
        choice_window.setLayout(general_layout)
        choice_window.show()


    def ommatidia_detecting_algorithm(self, polar_clustering=True, display=False,
                                      test=False, three_d=False, stage=None,
                                      prefiltered=False, regular=True, manual_edit=False,
                                      window_length=np.pi/3, neighborhood_smoothing=3,
                                      thickness=.5):
        """Apply the 3D ommatidia detecting algorithm (ODA-3D).
        

        Parameters
        ----------
        polar_clustering : bool, default=True
            Whether to use spectral clustering or to simply use 
            the nearest cluster center for finding ommatidial clusers.
        display : bool, default=False
            Whether to display the data. Combine with three_d to 
            plot in 3D.
        test : bool, default=False
            Whether to run segments designed for troubleshooting.
        three_d : bool, default=False
            Whether to use pyqtgraph to plot the data in 3D.
        stage : int, default=None
            Which stage to start from. If None, the program will ask for 
            user input. Optionally, one can put 'last' to start up from the last
            successfully loaded dataset.
        prefiltered : bool, default=False
            Whether the stack to import was prefiltered. If not, then user input
            is required.
        regular : bool, default=True
            Whether to assume the ommatidial lattice is regular.
        manual_edit : bool, default=False
            Whether to allow the user to manually correct the ommatidial coordinates
            for each segment.
        window_length : int, default=np.pi/3
            The length of the sliding window used in finding ommatidial clusters. Smaller
            is better for eyes with a large field of view and a lot of ommatidia. 
        neighborhood_smoothing : int, default=3
            The order of nearby ommatidia to use for smoothing their direction vectors. 1
            means to only use those immediately adajcent to each one. 2 includes second order
            neighbors and so on.
        thickness : float, default=.5
            The proportion of the residuals used for generating the  eye raster
            for running the ODA.
        """
        conditions = {'stack':'points' in dir(self), 
                      'cross-sections':'theta' in dir(self), 
                      'cluster labels':'labels' in dir(self), 
                      'ommatidial data':'ommatidial_data' in dir(self), 
                      'interommatidial data':'interommatidial_data' in dir(self)}
        # if the stage was not manually entered,
        if stage is None:
            choices = []
            # and saved data were loaded,
            if any([cond for cond in conditions.values()]):
                # offer the user to start from a particular stage
                print('The following datasets were found for this stack:')
                for num, (dataset, loaded) in enumerate(conditions.items()):
                    if loaded:
                        print(f"{num + 1}. {dataset.capitalize()}")
                        choices += [num + 1]
            while stage not in choices + [0, 'last']:
                stage = input(f"Enter the number {choices} to load from that stage anprocessing, enter 'last', or press 0 to start over: ")
                try:
                    stage = int(stage)
                except:
                    pass
        # if 'last' was entered, load the most advanced stage
        if stage == 'last':
            stage = 0
            for num, (dataset, loaded) in enumerate(conditions.items()):
                if loaded:
                    stage = num
        self.display = display
        if stage < 1:
            self.database.clear()
            self.save_database()
            if not prefiltered:
                self.gui = StackFilter(self.fns)
                low, high = self.gui.get_limits()
            else:
                first_img = load_image(self.fns[0])
                high = np.iinfo(first_img.dtype).max
                low = math.ceil(high / 2)
            self.import_stack(low, high)
            self.save_database()
            if self.display:
                self.plot_raw_data(three_d)
        print('Stack imported.')
        if stage < 2:
            self.get_cross_sections(chunk_process=(len(self.points) > 1000000))
            self.save_database()
            if self.display:
                self.plot_cross_section(three_d, residual_proportion=thickness)
        print('\nCross-section loaded.')
        if stage < 3:
            self.find_ommatidial_clusters(polar_clustering=polar_clustering,
                                          window_length=window_length, test=test,
                                          regular=regular, manual_edit=manual_edit,
                                          thickness=thickness)
            self.save_database()
            # if manual_edit:
            #     # make a raster image of the polar coordinates
            #     pts = Points(self.pts[:], sphere_fit=False, rotate_com=False,
            #                  spherical_conversion=False, polar=self.polar[:])
            #     # calculate pixel size based on shortest distances
            #     dists_tree = spatial.KDTree(pts.polar[:, :2])
            #     dists, inds = dists_tree.query((pts.polar[:, :2]), k=2)
            #     min_dist = 2 * np.mean(dists[:, 1])
            #     raster, (theta_vals, phi_vals) = pts.rasterize(pixel_length=min_dist)
            #     pixel_size = phi_vals[1] - phi_vals[0]
            #     # generate a boolean mask by smoothing the thresholded raster image
            #     mask = raster > 0
            #     mask = ndimage.gaussian_filter(mask.astype(float), 2)
            #     mask /= mask.max()
            #     thresh = 0.1
            #     mask = mask > thresh
            #     mask = 255 * mask.astype(int)
            #     raster = 255 * (raster / raster.max())
            #     raster = raster.astype('uint8')
            #     # use OmmatidiaGUI to fix coordinates
            #     fix_ommatidia = OmmatidiaGUI(raster, coords_arr=pts.polar[:, :2])
            #     self.ommatidial_inds = fix_ommatidia.coords
            #     self.ommatidia = self.pixel_size * self.ommatidial_inds
            if self.display:
                self.plot_ommatidial_clusters(three_d)
        print('\nOmmatidial clusters loaded.')
        if stage < 4:
            self.measure_ommatidia()
            self.save_database()
            if self.display:
                plot_scatter = len(self.ommatidial_data) < 100000
                self.plot_ommatidial_data(three_d, scatter=plot_scatter)
        print('\nOmmatidial data loaded.')
        if stage < 5:
            self.measure_interommatidia(test=test,
                                        neighborhood_smoothing=neighborhood_smoothing)
            self.save_database()
            if self.display:
                plot_scatter = len(self.interommatidial_data) < 100000
                self.plot_interommatidial_data(three_d=three_d, scatter=plot_scatter)
        print()

    def plot_data_3d(self):
        """Use pyqtgraph to plot in 3D any of the results.
        """
        # import widgets for a GUI
        from PyQt5.QtWidgets import (
            QApplication,
            QLabel,
            QMainWindow,
            QPushButton,
            QVBoxLayout,
            QHBoxLayout,
            QWidget)
        # what datasets are available?
        conditions = {'stack':'points' in dir(self), 
                      'cross-sections':'theta' in dir(self), 
                      'cluster labels':'labels' in dir(self), 
                      'ommatidial data':'ommatidial_data' in dir(self), 
                      'interommatidial data':'interommatidial_data' in dir(self)}
        # these are the functions
        functions = [
            self.plot_raw_data,
            self.plot_cross_section,
            self.plot_ommatidial_clusters,
            self.plot_ommatidial_data_3d,
            self.plot_interommatidial_data_3d]
        # make a main window with options for displaying different graphs
        app = QApplication([])
        main_window = QMainWindow()
        main_window.resize(640, 480)
        # make a GLViewWidget for displaying each of the scatterplots
        display_window = gl.GLViewWidget()
        # make a layout to place the display window inside the main window
        layout = QHBoxLayout()
        # main_window.addItem(display_window)
        display_window.show()
        # make a frame for placing all the buttons
        box_layout = QVBoxLayout()
        # make buttons for all the available datasets
        buttons = []
        for (title, success), function in zip(conditions.items(), functions):
            if success:
                button = QRadioButton(title)
                button.clicked.connect(partial(function, three_d=True, app=app,
                                               window=display_window,
                                               main_window=main_window))
                button.released.connect(display_window.clear)
                print(function)
                box_layout.addWidget(button)
                buttons += [button]
        # add sliders for minimum and maximum values in the colormap
        # class MplCanvas(FigureCanvasQTAgg):
        #     def __init__(self, parent=None, width=2, height=4, dpi=100):
        #         # fig = plt.Figure(figsize=(width, height), dpi=dpi)
        #         # self.axes = fig.add_subplot(111)
        #         fig, self.axes = fig.suplots(ncols=3)
        #         super(MplCanvas, self).__init__(fig)

        # plt_plot = MplCanvas(main_window)
        # # setup attributes to modify in the respective functions, updating the values of
        # # the slider
        # self.vmin, self.vmax = 0, 255
        # self.vmin_slider = Slider(plt_plot.axes[0], 'min', self.vmin, self.vmax_possible,
        #                           valinit=self.vmin, valfmt='%d', color='k',
        #                           orientation='vertical')
        # self.vmin_slider.on_changed(

        # # make a function to run everytime one of the buttons is clicked
        # def update_sliders(self):
            

        # plt_plot.axes.scatter(range(10), range(10))
        # box_layout.addWidget(plt_plot)


        # format widget
        # widget = QWidget()
        # widget.setLayout(box_layout)
        # place buttons inside main window
        # layout.addWidget(widget, 1)
        layout.addLayout(box_layout, 1)
        # place GLViewWidget inside main window too
        layout.addWidget(display_window, 5)
        widget = QWidget()
        widget.setLayout(layout)
        main_window.setCentralWidget(widget)
        # resize window to show both the buttons and GLViewWidget
        # run exec loop
        main_window.show()
        app.exec()

    def stats_summary(self, projected_radius=1e5, concave=False):
        """Calculate important statistics for whatever data is available.


        Parameters
        ----------
        projected_radius : float, default=1e5
            The radius to use for getting projected coordinates.
        concave : bool, default=False
            Whether to assume the eye is concave or convex (default).
        """
        # make pandas dataframe to store the statistics
        columns = ['variable', 'mean', 's.d.', 'N', 'min', '1/4', 'mid', '3/4', 'max']
        data = []
        stats_summary = pd.DataFrame([], columns=columns)
        # whole eye data and ommatidial data
        if 'ommatidial_data' in dir(self):
            # 1. whole eye
            # a. Surface Area - sum of the areas formed by the triangles of the centers of 
            # neighboring ommatidia
            pts = self.ommatidial_data[['x', 'y', 'z']].values
            theta, phi, radius = self.ommatidial_data[['theta', 'phi', 'radius']].values.T
            polar = np.array([theta, phi, radius]).T
            # use tesselation to calculate the total surface area and solid angle (FOV)
            tess = spatial.Delaunay(polar[:, :2])
            # get vertices in 3D using tess.simplices
            verts_polar = polar[tess.simplices]
            verts_rect = pts[tess.simplices]
            verts_rect /= 1000
            # the area of a triangle is half the determinant of the triangle
            verts_polar_norm = np.copy(verts_polar)
            verts_polar_norm[..., 2] = 1
            # calculate the distances between all triplets of points
            diffs = verts_rect[:, np.newaxis] - verts_rect[:, :, np.newaxis]
            dists = np.linalg.norm(diffs, axis=-1)
            sides = dists[:, [1, 2, 2], [0, 0, 1]]
            # remove triangles with outlier side lengths
            thresh = np.percentile(sides.max(1), 99)
            include = np.any(sides < thresh, axis=-1)
            sides = sides[include]
            # calculate the total area of the tesselation
            solid_angles = np.array([.5 * abs(np.linalg.det(triangle)) for triangle in verts_polar_norm])
            areas = np.array([.5 * abs(np.linalg.det(triangle)) for triangle in verts_rect])
            # surface area is the sum of the areas of the triangles making up 
            surface_area = sum(areas)
            stats_summary.loc[len(stats_summary)]= ['surface area (mm^2)', surface_area, np.nan, 1, np.nan, np.nan, np.nan, np.nan, np.nan]
            # projected: use projected polar coordinates and same tesselation approach
            x, y, z = self.ommatidial_data[['x_', 'y_', 'z_']].values.T
            pos_vectors = np.array([x, y, z]).T
            dx, dy, dz = self.ommatidial_data[['dx', 'dy', 'dz']].values.T
            dir_vectors = np.array([dx, dy, dz]).T
            dir_vectors /= np.linalg.norm(dir_vectors, axis=1)[:, np.newaxis]
            proj_coords = project_coords(pos_vectors, dir_vectors, radius=projected_radius, convex=True)
            proj_polar = rectangular_to_spherical(proj_coords)
            # todo: use the tessalated coordinates to measure the major and minor eye diameters
            # use the spherical polar coordinates to measure the spherical FOVs
            # use convex hull of 95% kde
            for coords, lbl in zip(
                [polar, proj_polar],
                ['spherical', 'projected']):
                # avoid nans
                no_nans = np.any(np.isnan(coords), axis=1) == False
                kde = stats.gaussian_kde(coords[no_nans, :2].T)
                # get grid of points around 
                th, ph = coords.T[:2, no_nans]
                width, height = th.max() - th.min(), ph.max() - ph.min()
                th_grid = np.linspace(th.min() - .05*width,  th.max() + .05*width, 200)
                ph_grid = np.linspace(ph.min() - .05*height, ph.max() + .05*height, 200)
                th_grid, ph_grid = np.meshgrid(th_grid, ph_grid)
                grid = np.array([th_grid.flatten(), ph_grid.flatten()])
                density = kde(grid).reshape(200, 200)
                density /= density.max()
                # get convex hull of the 95% density region
                xs, ys = np.where(density > .5)
                try:
                    pts = np.array([xs, ys]).T
                    chull_coords = spatial.ConvexHull(pts)
                except:
                    breakpoint()
                contour_coords = pts[chull_coords.vertices].T
                contour_th = th_grid[contour_coords[0], contour_coords[1]]
                contour_ph = ph_grid[contour_coords[0], contour_coords[1]]
                contour_coords = np.array([contour_th, contour_ph]).T
                # measure the major diameter using the longest distance among polar coordinates
                diffs = contour_coords[np.newaxis] - contour_coords[:, np.newaxis]
                dists = np.linalg.norm(diffs, axis=-1)
                max_dist = dists.max()
                ind = np.where(dists == max_dist)[0]
                major_vector = contour_coords[ind]
                major_vector_diff = major_vector[1] - major_vector[0]
                major_vector_unit = major_vector_diff / np.linalg.norm(major_vector_diff)
                major_vector_reciprocal = np.array([-major_vector_unit[1], major_vector_unit[0]])
                major_vector_unit = np.array([major_vector_unit, major_vector_reciprocal])
                # todo: get major and minor FOV's by rotating and projecting all of the 
                # coords coordinates onto those vectors
                coords_rotated = np.dot(
                    coords[:, :2] - coords[:, :2].mean(0), 
                    major_vector_unit.T) + coords[:, :2].mean(0)
                major_rotated = np.dot(
                    major_vector - major_vector.mean(0),
                    major_vector_unit.T) + major_vector.mean(0)
                # test: plot the original and rotated coordinates
                # fig, axes = plt.subplots(ncols=2)
                # original coordinates:
                # axes[0].scatter(coords[:, 0], coords[:, 1])
                # axes[0].plot(major_vector[:, 0], major_vector[:, 1], '-ok')
                # # rotated:
                # axes[1].scatter(coords_rotated[:, 0], coords_rotated[:, 1])
                # axes[1].plot(major_rotated[:, 0], major_rotated[:, 1], '-ok')
                # [ax.set_aspect('equal') for ax in axes]
                # plt.tight_layout()
                # plt.show()
                # use the rotated coordinates to measure the field of view based on 
                # the distributions of each dimension
                xs, ys = coords_rotated.T
                fov_major = np.diff(np.percentile(xs, [5, 95]))
                fov_minor = np.diff(np.percentile(ys, [5, 95]))
                # calculate the solid angle of each pixel in the density grid
                pxl_solid_angle = abs((th_grid[0, 1] - th_grid[0, 0])*(ph_grid[1, 0] - ph_grid[0, 0]))
                fov = sum(density > .5).sum() * pxl_solid_angle
                # the FOV is the sum of all the solid angles in the density grid
                stats_summary.loc[len(stats_summary)] = [
                    f"FOV_{lbl}", fov, np.nan, 1, fov_minor[0],
                    np.nan, np.nan, np.nan, fov_major[0]]
            # todo: use the projected polar coordinates to measure the projected FOVs
            # instead of using the tesselation, use the 95% kde
            proj_polar[:, 2] = 1
            include = np.any(np.isnan(proj_polar), axis=1) == False
            proj_tess = spatial.Delaunay(proj_polar[include][:, :2])
            verts_proj_polar = proj_polar[include][proj_tess.simplices]
            # remove triangles with outlier side lengths
            # calculate the side lengths
            diffs = verts_proj_polar[:, np.newaxis] - verts_proj_polar[:, :, np.newaxis]
            dists = np.linalg.norm(diffs, axis=-1)
            sides = dists[:, [1, 2, 2], [0, 0, 1]]
            thresh = np.percentile(np.nanmax(sides, 1), 99)
            include = sides.max(1) < thresh
            verts_proj_polar = verts_proj_polar[include]
            # add up the areas of the projected triangles
            proj_areas = np.array([.5 * abs(np.linalg.det(triangle))
                                   for triangle in verts_proj_polar])
            fov_projected = proj_areas.sum()
            # c. Radius
            no_nans = np.any(np.isnan(proj_polar), axis=1) == False
            chull_polar_projected = spatial.ConvexHull(proj_polar[no_nans][:, :2])
            contour_proj = proj_polar[:, :2][no_nans][chull_polar_projected.vertices]
            ellipse_proj = LSqEllipse()
            ellipse_proj.fit(contour_proj.T)
            center, width, height, phi = ellipse_proj.parameters()
            fov_proj_minor, fov_proj_major = min(width, height), max(width, height)
            stats_summary.loc[len(stats_summary.index)] = [
                'FOV_projected', fov_projected, np.nan, 1, fov_proj_minor, np.nan, np.nan,
                np.nan, fov_proj_major]
            # 2. ommatidial data - columns=[N, mean, s.d., min, 1/4, mid, 3/4, max]
            # a. number of ommatidia - len(self.ommatidial_data)
            ommatidia_count = len(self.ommatidial_data)
            stats_summary.loc[len(stats_summary.index)] = [
                'ommatidia_count', ommatidia_count, np.nan, 1, np.nan, np.nan,
                np.nan, np.nan, np.nan]
            # b. ommatidial density - ommatidia count / FOV
            polar_density = ommatidia_count / fov_projected
            density = ommatidia_count / surface_area
            stats_summary.loc[len(stats_summary)] = [
                "polar_density_(facets/sr)", polar_density, np.nan, 1, np.nan, np.nan, np.nan,
                np.nan, np.nan]
            stats_summary.loc[len(stats_summary)] = [
                'density_(facets/mm^2)', density, np.nan, 1, np.nan, np.nan, np.nan,
                np.nan, np.nan]
            for var in ['radius', 'lens_diameter', 'lens_diameter_adj','lens_area',
                        'skewness', 'spherical_IOA', 'cross_section_area',
                        'cross_section_height']:
                vals = np.copy(self.ommatidial_data[var].values)
                no_nans = np.isnan(vals) == False
                vals = vals[no_nans]
                key_vals = np.percentile(vals, [0, 25, 50, 75, 100])
                stats_summary.loc[len(stats_summary)] = [
                    var, vals.mean(), vals.std(), len(vals),
                    key_vals[0], key_vals[1], key_vals[2], key_vals[3], key_vals[4]]
        self.stats = stats_summary
        return self.stats
        # # interommatidial data
        # if 'interommatidial_data' in dir(self):
        #     # get the 3 orientation groups
        #     io_data = self.interommatidial_data
        #     angs = io_data.orientation.values
        #     coords = np.array([np.cos(angs), np.sin(angs)]).T
        #     # find the 6 2D clusters and unwrap for the three lattice axes
        #     init_angs = (np.arange(6) * 2 * np.pi / 6) - np.pi/2
        #     init_coords = np.array([np.cos(init_angs), np.sin(init_angs)]).T
        #     clusterer = cluster.KMeans(6, init=init_coords, n_init=1).fit(coords)
        #     lbls = clusterer.labels_
        #     lbls[lbls > 2] -= 3
            

        #     coords[coords[:, 0] < 0] = -coords[coords[:, 0] < 0]
        #     centers = clusterer.cluster_centers_
        #     centers = centers[centers[:, 0] >= 0]
        #     dist_tree = spatial.KDTree(centers)
        #     dists, inds = dist_tree.query(coords)


