import copy
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox
from matplotlib.backend_bases import NavigationToolbar2
import numpy as np
import os
import PIL
import scipy
import seaborn as sbn
import sys
from tempfile import mkdtemp

from PyQt5.QtWidgets import QWidget, QFileDialog, QApplication
from pyqtgraph.Qt import QtCore, QtGui
from PyQt5.QtWidgets import QFileDialog
import pyqtgraph.opengl as gl
import pyqtgraph as pg


APP = None
# MAIN_WINDOW = NONE
blue, green, yellow, orange, red, purple = [
    (0.3, 0.45, 0.69), (0.33, 0.66, 0.41), (0.83, 0.74, 0.37),
    (0.78, 0.5, 0.16), (0.77, 0.31, 0.32), (0.44, 0.22, 0.78)]


def rgb_to_hsv(rgb):
    """Convert image from RGB to HSV."""
    if rgb.ndim == 3:
        ret = matplotlib.colors.rgb_to_hsv(rgb)
    else:
        l, w = rgb.shape
        ret = np.repeat(rgb, 3, axis=(-1))
    return ret

# make a class for plotting a heatmap with distplots along the axes
class VarSummary():
    def __init__(self, xs, ys, colorvals, cmap='viridis', image_size=10**5,
                 color='k', center=True, color_label="Color", suptitle="Title",
                 vmin=None, vmax=None, scatter=False, marker='o', marker_sizes=None):
        """Useful for plotting scatterplots or heatmaps of 2d coordinates.

        
        Plots a 2d scatterplot or pcolormesh with the provided x, y, and corresponding
        color values. This will plot those with a colorbar histogram to the side. This
        is particularly useful for plotting ommatidial coordinates with a particular 
        variable (like lens area or IO angle) color coded.

        Parameters
        ----------
        xs, ys, colorvals : array-like, length=N
            The x- and y-coordinates and associated values to be colored.
        cmap : str, default='viridis'
            The name of the pyplot colormap for plotting colorvals. 
        

        """
        self.marker_sizes = marker_sizes
        self.marker = marker
        self.vmin = vmin
        self.vmax = vmax
        self.x = xs
        self.y = ys
        self.pts = np.array([xs, ys]).T
        self.center = center
        if self.center:
            self.x_offset = self.x.mean()
            self.y_offset = self.y.mean()
            self.x -= self.x_offset
            self.y -= self.y_offset
            self.pts -= self.pts.mean(0)
        self.colorvals = colorvals
        self.image_size = image_size
        self.cmap = cmap
        self.color = color
        self.color_label = color_label
        self.suptitle = suptitle
        self.scatter = scatter

    def plot(self, fig=None, gridspec=None, inset=False, xmin=None, xmax=None,
             ymin=None, ymax=None, vmin=None, vmax=None, margins=True):
        """Plot the coordinates as a scatterplot or 2D histogram.


        Parameters
        ----------
        fig : plt.figure, default=None
            The figure you want to plot on. Defaults to making a new figure.
        gridspec : plt.figure, default=None
            The figure you want to plot on. Defaults to making a new figure.
        inset : bool, default=False
            Whether to plot a zoomed inset.
        xmin, xmax, ymin, ymax, vmin, vmax : float, default=None
            Optionally, you can provide bounds on the x values, y values, and colorvals.
        margins : bool, default=False
            Whether to plot row and column projected data in the margins.
        """
        if xmin is None:
            self.xmin = self.x.min()
        else:
            # self.xmin = xmin - self.x_offset
            self.xmin = xmin
        if xmax is None:
            self.xmax = self.x.max()
        else:
            # self.xmax = xmax - self.x_offset
            self.xmax = xmax
        if ymin is None:
            self.ymin = self.y.min()
        else:
            # self.ymin = ymin - self.y_offset
            self.ymin = ymin
        if ymax is None:
            self.ymax = self.y.max()
        else:
            # self.ymax = ymax - self.y_offset
            self.ymax = ymax
        if vmin is not None:
            self.vmin = vmin
        elif self.vmin is None:
            self.vmin = self.colorvals.min()
        if vmax is not None:
            self.vmax = vmax
        elif self.vmax is None:
            self.vmax = self.colorvals.max()
        # x_range = self.x.max() - self.x.min()
        # y_range = self.y.max() - self.y.min()
        x_range = self.xmax - self.xmin
        y_range = self.ymax - self.ymin
        # figure out side lengths needed for input image size
        ratio = y_range / x_range
        x_len = int(np.round(np.sqrt(self.image_size/ratio)))
        # get x and y ranges corresponding to image size
        xs = np.linspace(self.xmin, self.xmax, x_len)
        self.raster_pixel_length = xs[1] - xs[0]
        ys = np.arange(self.ymin, self.ymax, self.raster_pixel_length)
        xs = xs[:-1] + (self.raster_pixel_length / 2.)
        ys = ys[:-1] + (self.raster_pixel_length / 2.)
        xvals, yvals = np.meshgrid(xs, ys)
        self.xpad = .05 * (abs(self.xmax - self.xmin))
        self.ypad = .05 * (abs(self.ymax - self.ymin))
        self.xmin, self.xmax = self.xmin - self.xpad, self.xmax + self.xpad
        self.ymin, self.ymax = self.ymin - self.ypad, self.ymax + self.ypad
        # split figure into axes using a grid
        ratio = (self.ymax - self.ymin) / (self.xmax - self.xmin)
        width = 1 + 4 + .3
        height = 1 + 4 * ratio
        scale = 7/width
        if fig is not None:
            self.fig = fig
        else:
            self.fig = plt.figure(figsize=(scale * width, scale * height),
                                  dpi=150)
        if gridspec is None:
            if margins:
                self.gridspec = self.fig.add_gridspec(
                    ncols=3, nrows=2,
                    width_ratios=[1, 4, .3],
                    height_ratios=[4*ratio, 1],
                    wspace=0, hspace=0)
            else:
                self.gridspec = self.fig.add_gridspec(
                    ncols=2, nrows=1,
                    width_ratios=[4, .3],
                    height_ratios=[4*ratio],
                    wspace=0, hspace=0)
        else:
            self.gridspec = gridspec
        # calculate ideal range for colorvals to share between vertical and horizontal axes
        no_nans = np.isnan(self.colorvals) == False
        no_infs = np.isinf(self.colorvals) == False
        no_nans = no_nans * no_infs
        if self.vmin is None and self.vmax is None:
            cmin, cmax = self.colorvals[no_nans].min(), self.colorvals[no_nans].max()
            pad = .025 * (cmax - cmin)
            cmin -= pad
            cmax += pad
            self.cmin = cmin
            self.cmax = cmax
        else:
            cmin = self.vmin
            cmax = self.vmax
        # generate ticks for colorvals
        crange = cmax - cmin
        scale = np.round(np.log10(crange))
        cticks = np.linspace(np.round(cmin,  - int(scale - 1)),
                             np.round(cmax,  - int(scale - 1)), 4)[1:-1]
        cticks = np.round(cticks, int(scale - 1))
        # plot 2d heatmap
        if margins:
            self.heatmap_ax = self.fig.add_subplot(self.gridspec[0, 1])
        else:
            self.heatmap_ax = self.fig.add_subplot(self.gridspec[0, 0])            
        no_nans = np.isnan(self.colorvals) == False
        grid = scipy.interpolate.griddata(self.pts[no_nans],
                                          self.colorvals[no_nans],
                                          np.array([xvals, yvals]).T,
                                          method='nearest')
        grid = grid.astype(float)
        mask = np.histogram2d(self.pts[:, 0], self.pts[:, 1],
                              bins=[grid.shape[0], grid.shape[1]])
        grid[mask[0] == 0] = np.nan
        no_nans = np.isnan(grid) == False
        self.plot_heatmap(xs, ys, grid, inset=inset, margins=margins)
        # self.heatmap_ax.set_title(self.suptitle)
        # make colorbar/histogram
        if margins:
            self.colorbar_ax = self.fig.add_subplot(self.gridspec[0, 2])
        else:
            self.colorbar_ax = self.fig.add_subplot(self.gridspec[0, 1])
        bins = np.linspace(cmin, cmax, 101)
        counts, bin_edges = np.histogram(self.colorvals, bins=bins)
        # self.histogram = sbn.distplot(self.colorvals, kde=False, color=self.color,
        #                               ax=self.colorbar_ax, vertical=True, bins=bins,
        #                               axlabel=False)
        self.histogram = sbn.histplot(y=self.colorvals, kde=False, color=self.color,
                                      ax=self.colorbar_ax, bins=bins, fill=True,
                                      alpha=.25)
        bin_edges = np.repeat(bins, 2)[1:-1]
        heights = np.repeat(counts, 2)
        self.colorbar_ax.plot(heights, bin_edges, color='w')
        vals = np.linspace(cmin, cmax, 100)
        self.colorbar_ax.pcolormesh([0, counts.max()], vals,
                                    np.repeat(vals[:, np.newaxis], 2, axis=-1),
                                    cmap=self.cmap, zorder=0)
        # self.colorbar_ax.set_xlabel("Count")
        sbn.despine(ax=self.colorbar_ax, bottom=False)
        self.colorbar_ax.set_xticks([])
        self.colorbar_ylabel = self.colorbar_ax.set_ylabel(
            self.color_label, rotation=270, labelpad=20)
        self.colorbar_ax.yaxis.set_label_position("right")
        self.colorbar_ax.yaxis.tick_right()
        self.colorbar_ax.set_xlim(0, counts.max())
        # self.colorbar_ylabel.set_rotation(270)
        if margins:
            # plot colorval spread using CIs along vertical
            self.vertical_ax = self.fig.add_subplot(self.gridspec[0, 0],
                                                    sharey=self.heatmap_ax)

            lows, mids, highs = [], [], []
            lows_ci, highs_ci = [], []
            # use actual values, spaced out in 15 degree bins
            bin_width = 5
            num_bins_vertical = self.y.ptp() / bin_width
            bins_vertical = np.arange(int(num_bins_vertical) + 2).astype(float)
            bins_vertical *= bin_width
            bins_vertical += self.y.min()
            # center the bins by splitting the difference
            diff = bins_vertical.max() - self.y.max()
            bins_vertical -= diff/2
            # go through each bin and calculate median, IQR, and 99% CI
            for bin_low, bin_high in zip(bins_vertical[:-1], bins_vertical[1:]):
                include = (self.y > bin_low) * (self.y <= bin_high)
                sub_vals = np.asarray(self.colorvals[include])
                no_nans = np.isnan(sub_vals) == False
                if np.any(no_nans):
                    low, mid, high = np.percentile(sub_vals[no_nans], [25, 50, 75])
                    # bootstrap subvals for CI interval of the median
                    inds = np.arange(len(sub_vals))
                    rand_inds = np.random.choice(inds, (1000, len(sub_vals)))
                    rand_vals = sub_vals[rand_inds]
                    rand_meds = np.median(rand_vals, axis=0)
                    low_ci, high_ci = np.percentile(rand_meds, (.5, 99.5))
                else:
                    low, low_ci, med, high_ci, high = np.repeat(np.nan, 5)
                # store values
                for val, storage in zip(
                        [low, low_ci, mid, high_ci, high],
                        [lows, lows_ci, mids, highs_ci, highs]):
                    storage += [val]
            bins_vertical_labels = .5 * (bins_vertical[:-1] + bins_vertical[1:])
            # plot
            no_nans = np.isnan(self.colorvals) == False
            self.vertical_ax.hist2d(self.colorvals[no_nans],
                                    self.y[no_nans], bins=[15, ys.shape[0]],
                                    cmap='Greys')
            self.vertical_ax.plot(mids, bins_vertical_labels, color=red)
            for y, low, high, low_ci, high_ci in zip(bins_vertical_labels, lows, highs,
                                                     lows_ci, highs_ci):
                self.vertical_ax.plot([low, high], [y, y], color=red, alpha=.5)
                self.vertical_ax.plot([low_ci, high_ci], [y, y], color=red, alpha=1)
                self.vertical_ax.scatter([low_ci, high_ci], [y, y], marker='|', color=red, alpha=1)
            # self.vertical_ax.fill_betweenx(
            #     ys, lows, highs, alpha=.5, color=red, edgecolor="none", lw=0)
            self.vertical_ax.set_ylim(self.ymin, self.ymax)
            self.vertical_ax.set_xlim(cmin, cmax)
            self.vertical_ax.set_xticks(cticks)
            self.vertical_ax.set_ylabel("Elevation ($^\circ$)")
            sbn.despine(ax=self.vertical_ax)
            # get descriptive stats on the horizontal axis
            lows, mids, highs = [], [], []
            lows_ci, highs_ci = [], []
            # use actual values, spaced out in 15 degree bins
            bin_width = 5
            num_bins_horizontal = self.x.ptp() / bin_width
            bins_horizontal = np.arange(int(num_bins_horizontal) + 2).astype(float)
            bins_horizontal *= bin_width
            bins_horizontal += self.x.min()
            # center the bins by splitting the difference
            diff = bins_horizontal.max() - self.x.max()
            bins_horizontal -= diff/2
            # go through each bin and calculate median, IQR, and 99% CI
            for bin_low, bin_high in zip(bins_horizontal[:-1], bins_horizontal[1:]):
                include = (self.x > bin_low) * (self.x <= bin_high)
                if np.any(include):
                    sub_vals = np.asarray(self.colorvals[include])
                    no_nans = np.isnan(sub_vals) == False
                    low, mid, high = np.percentile(sub_vals[no_nans], [25, 50, 75])
                    # bootstrap subvals for CI interval of the median
                    inds = np.arange(len(sub_vals))
                    rand_inds = np.random.choice(inds, (1000, len(sub_vals)))
                    rand_vals = sub_vals[rand_inds]
                    rand_meds = np.median(rand_vals, axis=0)
                    low_ci, high_ci = np.percentile(rand_meds, (.5, 99.5))
                else:
                    low, low_ci, mid, high_ci, high = np.repeat(np.nan, 5)
                # store values
                for val, storage in zip(
                        [low, low_ci, mid, high_ci, high],
                        [lows, lows_ci, mids, highs_ci, highs]):
                    storage += [val]
            bins_horizontal_labels = .5 * (bins_horizontal[:-1] + bins_horizontal[1:])
            # self.horizontal_ax.set_xlabel(self.color_label)
            # plot expected colorvals using bootstrapped CIs along horizontal
            self.horizontal_ax = self.fig.add_subplot(self.gridspec[1, 1],
                                                      sharex=self.heatmap_ax)
            no_nans = np.isnan(self.colorvals) == False
            self.horizontal_ax.hist2d(self.x[no_nans], self.colorvals[no_nans],
                                      bins=[xs.shape[0], 15],
                                      cmap='Greys')
            for x, low, high, low_ci, high_ci, mid in zip(bins_horizontal_labels, lows, highs,
                                                          lows_ci, highs_ci, mids):
                self.horizontal_ax.plot([x, x], [low, high], color=red, alpha=.5)
                self.horizontal_ax.plot([x, x], [low_ci, high_ci], color=red, alpha=1)
                self.horizontal_ax.scatter([x, x], [low_ci, high_ci], marker='_', color=red, alpha=1)
                self.horizontal_ax.plot(x, mid, marker='.', color=red, alpha=1)
            self.horizontal_ax.plot(bins_horizontal_labels, mids, color=red)
            # self.horizontal_ax.fill_between(
            #     xs, lows, highs, alpha=.2, color=self.color, edgecolor="none", lw=0)
            self.horizontal_ax.set_xlim(self.xmin, self.xmax)
            # self.horizontal_ax.set_ylabel(self.color_label)
            self.horizontal_ax.set_yticks(cticks)
            sbn.despine(ax=self.horizontal_ax)
            self.horizontal_ax.set_ylim(cmin, cmax)
            self.horizontal_ax.set_xlabel("Azimuth ($^\circ$)")
        plt.suptitle(self.suptitle)
        # plt.tight_layout()

    def plot_heatmap(self, xs, ys, grid, inset=False, inset_width=20, margins=True):
        # self.heatmap = self.heatmap_ax.scatter(
        #     self.x, self.y, c=self.colorvals, cmap=self.cmap,
        #     vmin=self.cmin, vmax=self.cmax)
        # breakpoint()
        if self.vmin is not None:
            cmin = self.vmin
        else:
            cmin = self.cmin
        if self.vmax is not None:
            cmax = self.vmax
        else:
            cmax = self.cmax
        # make a label with the important colorval statistics
        # no_nans = np.isnan(self.colorvals) == False
        # low, median, high = np.percentile(self.colorvals[no_nans], [25, 50, 75])
        # # bootstrap sub_vals to get 99% CI of the median
        # indices = np.arange(len(self.colorvals[no_nans]))
        # indices_random = np.random.choice(indices, (len(self.colorvals[no_nans]), 1000), replace=True)
        # meds_random = self.colorvals[no_nans][indices_random]
        # meds_random = np.median(meds_random, axis=1)
        # low_ci, high_ci = np.percentile(meds_random, [.5, 99.5])
        # # make a label for the plotted data per  group
        # label = f"m={median:.2f}\nIQR=[{low:.2f}, {high:.2f}]\nCI=({low_ci:.2f}, {high_ci:.2f})"
        print()
        print(self.suptitle + ":")
        # print(label)
        if inset:
            xmin_inset, xmax_inset = -inset_width/2., inset_width/2.
            ymin_inset, ymax_inset = xmin_inset, xmax_inset
            padding = .1 * xmax_inset
            self.heatmap_axins = zoomed_inset_axes(
                self.heatmap_ax, 4, loc='upper right')
            self.heatmap_axins.set_xlim(xmin_inset, xmax_inset)
            self.heatmap_axins.set_ylim(xmin_inset, xmax_inset)
            # self.heatmap_axins.set_xticks([])
            # self.heatmap_axins.set_yticks([])
            self.heatmap_ax.indicate_inset_zoom(self.heatmap_axins, edgecolor='k')
            mark_inset(self.heatmap_ax, self.heatmap_axins, loc1=2, loc2=4,
                       fc='none', ec='k', lw=.5)
        if self.scatter:
            # use the density-based area for plotting the scatterplot
            coords = np.array([self.x, self.y]).T
            self.dist_tree = scipy.spatial.KDTree(coords)
            dists, inds = self.dist_tree.query(coords, k=4)
            dists = dists[:, 1:]
            dists = dists.mean(-1)
            radii = dists/2
            radii = np.pi * radii ** 2
            # self.heatmap = SizedScatter(
            #     self.x, self.y, self.heatmap_ax, size=radii, linewidth=0)
            # convert radii to 
            if self.marker_sizes is None or len(self.x) != len(self.marker_sizes):
                self.heatmap = self.heatmap_ax.scatter(
                    self.x, self.y, c=self.colorvals, vmin=cmin, vmax=cmax,
                    cmap=self.cmap, marker=self.marker)
                if inset:
                    xs, ys = self.x, self.y
                    include = (xs >= xmin_inset - padding) * (xs <= xmax_inset + padding)
                    include *= (ys >= ymin_inset - padding) * (ys <= ymax_inset + padding)
                    if np.any(include):
                        cmap = plt.get_cmap(self.cmap)
                        colorvals = np.copy(self.colorvals)
                        colorvals[colorvals < self.vmin] = 0
                        colorvals -= self.vmin
                        colorvals /= (self.vmax - self.vmin)
                        colors = cmap(colorvals)
                        breakpoint()
                        self.heatmap_axins.scatter(xs[include], ys[include], c=colors[include],
                                                   marker=self.marker)
            else:
                # plot a bunch of circles with colors from colorvals and areas from marker_sizes
                cmap = plt.get_cmap(self.cmap)
                colorvals = np.copy(self.colorvals)
                
                colorvals[colorvals < self.vmin] = 0
                colorvals -= self.vmin
                colorvals /= (self.vmax - self.vmin)
                colors = cmap(colorvals)
                # add zoomed inset
                # if inset:
                #     xmin_inset, xmax_inset = -inset_width/2., inset_width/2.
                #     ymin_inset, ymax_inset = xmin_inset, xmax_inset
                #     self.heatmap_axins = zoomed_inset_axes(
                #         self.heatmap_ax, 5, loc='upper right')
                #     self.heatmap_axins.set_xlim(xmin_inset, xmax_inset)
                #     self.heatmap_axins.set_ylim(xmin_inset, xmax_inset)
                #     # self.heatmap_axins.set_xticks([])
                #     # self.heatmap_axins.set_yticks([])
                #     self.heatmap_ax.indicate_inset_zoom(self.heatmap_axins, edgecolor='k')
                #     mark_inset(self.heatmap_ax, self.heatmap_axins, loc1=2, loc2=4,
                #                fc='none', ec='k', lw=.5)
                self.circles = []
                for num, (color, radius, x, y) in enumerate(zip(
                        colors, self.marker_sizes, self.x, self.y)):
                    circle = plt.Circle((x, y), radius=radius, color=color)
                    self.circles += [circle]
                    self.heatmap_ax.add_artist(circle)
                    if inset:
                        include = np.all([
                            x >= xmin_inset - padding, x <= xmax_inset + padding,
                            y >= ymin_inset - padding, y <= ymax_inset + padding])
                        if include:
                            circle = plt.Circle((x, y), radius=radius, color=color)
                            self.heatmap_axins.add_artist(circle)
                # for x, y, c
            # todo: use plt.Circles instead so that we can specify the area in data units
            # self.circles = []
            # colorvals = np.copy(self.colorvals)
            # colorvals[colorvals < cmin] = cmin
            # colorvals[colorvals > cmax] = cmax
            # cvals = plt.cm.get_cmap(self.cmap)(colorvals)
            # for x, y, cval, radius in zip(self.x, self.y, cvals, dists):
            #     self.circles += [plt.Circle((x, y), radius=radius, linewidth=0)]
            # self.heatmap = matplotlib.collections.PatchCollection(self.circles)
            # self.heatmap_ax.add_collection(self.heatmap)
                                 
        else:
            self.heatmap = self.heatmap_ax.pcolormesh(
                xs, ys, grid.T, cmap=self.cmap, antialiased=True,
                vmin=cmin, vmax=cmax)
            if inset:
                xs, ys = self.x, self.y
                include = (xs >= xmin_inset - padding) * (xs <= xmax_inset + padding)
                include *= (ys >= ymin_inset - padding) * (ys <= ymax_inset + padding)
                if np.any(include):
                    cmap = plt.get_cmap(self.cmap)
                    colorvals = np.copy(self.colorvals)
                    colorvals[colorvals < self.vmin] = 0
                    colorvals -= self.vmin
                    colorvals /= (self.vmax - self.vmin)
                    colors = cmap(colorvals)
                    self.heatmap_axins.scatter(xs[include], ys[include], c=colors[include],
                                               marker=self.marker)

        # self.heatmap_ax.legend()
        self.heatmap_ax.set_aspect('equal')
        if inset:
            self.xmax += inset_width
        self.heatmap_ax.set_xlim(self.xmin, self.xmax)
        self.heatmap_ax.set_ylim(self.ymin, self.ymax)
        if margins:
            sbn.despine(ax=self.heatmap_ax, bottom=True, left=True)
        else:
            sbn.despine(ax=self.heatmap_ax, bottom=False, left=False, trim=True)
        self.heatmap_ax.label_outer()
        if margins:
            self.heatmap_ax.tick_params(axis=u'both', which=u'both',length=0)


class VarSummary_lines(VarSummary):
    def __init__(self, xs, ys, colorvals, cmap='viridis', image_size=10**5,
                 color='k', center=True, color_label="Color", suptitle="Title",
                 xs1=None, xs2=None, ys1=None, ys2=None):
        self.xs1, self.xs2, self.ys1, self.ys2 = xs1, xs2, ys1, ys2
        VarSummary.__init__(self, xs, ys, colorvals, cmap='viridis',
                            image_size=10**5, color='k', center=True,
                            color_label="Color", suptitle="Title")
        if self.center:
            self.xs1 -= self.x_offset
            self.xs2 -= self.x_offset
            self.ys1 -= self.y_offset
            self.ys2 -= self.y_offset

    def plot_heatmap(self, *args):
        cmap = plt.cm.get_cmap(self.cmap)
        self.heatmap = []
        for x1, x2, y1, y2, cval in zip(self.xs1, self.xs2, self.ys1,
                                        self.ys2, self.colorvals):
            prop = (cval - self.cmin)/(self.cmax - self.cmin)
            self.heatmap_ax.plot([x1, x2], [y1, y2], color=cmap(prop))
        self.heatmap_ax.set_aspect('equal')
        self.heatmap_ax.set_xlim(self.xmin, self.xmax)
        self.heatmap_ax.set_ylim(self.ymin, self.ymax)
        sbn.despine(ax=self.heatmap_ax, bottom=True, left=True)
        self.heatmap_ax.label_outer()
        self.heatmap_ax.tick_params(axis=u'both', which=u'both',length=0)



class SizedScatter():
    def __init__(self,x,y,ax,size=1,**kwargs):
        """Makes a Scatterplot with appropriately sized markers.

        Taken from https://stackoverflow.com/questions/48172928/scale-matplotlib-pyplot-axes-scatter-markersize-by-x-scale/48174228#48174228
        """
        self.n = len(x)
        self.ax = ax
        self.ax.figure.canvas.draw()
        self.size_data=size
        self.size = size
        self.sc = ax.scatter(x,y,s=self.size,**kwargs)
        self._resize()
        self.cid = ax.figure.canvas.mpl_connect('draw_event', self._resize)

    def _resize(self,event=None):
        ppd=72./self.ax.figure.dpi
        trans = self.ax.transData.transform
        breakpoint()
        # size =  (trans((np.ones(len(self.size_data), self.size_data[:, np.newaxis]) - trans(0)[np.newaxis])*ppd[1]
        
        if s != self.size:
            self.sc.set_sizes(s**2*np.ones(self.n))
            self.size = s
            self._redraw_later()
    
    def _redraw_later(self):
        self.timer = self.ax.figure.canvas.new_timer(interval=10)
        self.timer.single_shot = True
        self.timer.add_callback(lambda : self.ax.figure.canvas.draw_idle())
        self.timer.start()

class ScatterPlot3d():
    """Plot 3d datapoints using pyqtgraph's GLScatterPlotItem."""
    def __init__(self, arr, color=None, size=1, window=None,
                 colorvals=None, cmap=plt.cm.viridis, title="3D Scatter Plot",
                 app=APP):
        self.title = title
        self.arr = arr
        if app is None:
            app = QApplication([])
        self.app = app
        self.color = color
        self.cmap = cmap
        self.size = size
        self.window = window
        self.n, self.dim = self.arr.shape
        assert self.dim == 3, ("Input array should have shape "
                               "N x 3. Instead it has "
                               "shape {} x {}.".format(
                                   self.n,
                                   self.dim))
        if colorvals is not None:
            assert len(colorvals) == self.n, print("input colorvals should "
                                                   "have the same lengths as "
                                                   "input array")
            if np.any(colorvals < 0) or np.any(colorvals > 1):
                colorvals = (colorvals - colorvals.min()) / \
                    (colorvals.max() - colorvals.min())
            self.color = np.array([self.cmap(c) for c in colorvals])
        elif color is not None:
            assert len(color) == 4, print("color input should be a list or tuple "
                                          "of RGBA values between 0 and 1")
            if isinstance(self.color, (tuple, list)):
                self.color = np.array(self.color)
            if self.color.max() > 1:
                self.color = self.color / self.color.max()
            self.color = tuple(self.color)
        else:
            self.color = (1, 1, 1, 1)
        self.plot()

    def plot(self):
        if self.window is None:
            self.window = gl.GLViewWidget()  # todo: consider using QDialog
            self.window.setWindowTitle(self.title)
        self.scatter_GUI = gl.GLScatterPlotItem(
            pos=self.arr, size=self.size, color=self.color)
        self.window.addItem(self.scatter_GUI)

    def show(self):
        # self.window.exec_()
        self.window.show()
        # self.app.exec_()


class tracker_window():

    def __init__(self, dirname="./"):
        # m.pyplot.ion()
        self.dirname = dirname
        self.load_filenames()
        self.num_frames = len(self.filenames)
        self.range_frames = np.array(range(self.num_frames))
        self.curr_frame_index = 0
        self.data_changed = False
        # the figure
        self.load_image()
        # figsize = self.image.shape[1]/90, self.image.shape[0]/90
        h, w = self.image.shape[:2]
        if w > h:
            fig_width = 8
            fig_height = h/w * fig_width
        else:
            fig_height = 8
            fig_width = w/h * fig_height
        # start with vmin and vmax at extremes
        self.vmin = 0
        self.vmax = np.iinfo(self.image.dtype).max
        self.vmax_possible = self.vmax
        # self.figure = plt.figure(1, figsize=(
        #     figsize[0]+1, figsize[1]+2), dpi=90)
        self.figure = plt.figure(1, figsize=(fig_width, fig_height), dpi=90)
        # xmarg, ymarg = .2, .1
        # fig_left, fig_bottom, fig_width, fig_height = .15, .1, .75, .85
        fig_left, fig_bottom, fig_width, fig_height = .1, .1, .75, .8
        axim = plt.axes([fig_left, fig_bottom, fig_width, fig_height])
        self.implot = plt.imshow(self.image, cmap='viridis', vmin=self.vmin, vmax=self.vmax)
        self.xlim = self.figure.axes[0].get_xlim()
        self.ylim = self.figure.axes[0].get_ylim()
        self.axis = self.figure.get_axes()[0]
        self.figure.axes[0].set_xlim(*self.xlim)
        self.figure.axes[0].set_ylim(*self.ylim)
        self.image_data = self.axis.images[0]
        # title
        self.title = self.figure.suptitle(
            '%d - %s' % (self.curr_frame_index + 1, self.filenames[self.curr_frame_index].rsplit('/')[-1]))

        # the slider controlling frames
        axframe = plt.axes([fig_left, 0.04, fig_width, 0.02])
        self.curr_frame = Slider(
            axframe, 'frame', 1, self.num_frames, valinit=1, valfmt='%d', color='k')
        self.curr_frame.on_changed(self.change_frame)
        # the vmin slider
        vminframe = plt.axes([fig_left + fig_width + .02, 0.1, .02, .05 + .7])
        self.vmin = Slider(
            vminframe, 'min', 0, self.vmax_possible,
            valinit=0, valfmt='%d', color='k', orientation='vertical')
        self.vmin.on_changed(self.show_image)
        # the vmax slider
        vmaxframe = plt.axes([fig_left + fig_width + .1, 0.1, .02, .05 + .7])
        self.vmax = Slider(
            vmaxframe, 'max', 0, self.vmax_possible, valinit=self.vmax_possible,
            valfmt='%d', color='k', orientation='vertical')
        self.vmax.on_changed(self.show_image)
        # limit both sliders
        self.vmin.slidermax = self.vmax
        self.vmax.slidermin = self.vmin
        # the colorbar in between
        self.cbar_ax = plt.axes([fig_left + fig_width + .06, 0.1, .02, .05 + .7])
        self.colorvals = np.arange(self.vmax_possible)
        self.cbar = self.cbar_ax.pcolormesh(
            [0, 10], self.colorvals,
            np.repeat(self.colorvals[:, np.newaxis], 2, axis=-1),
            cmap='viridis', vmin=0, vmax=self.vmax_possible, shading='nearest')
        self.cbar_ax.set_xticks([])
        self.cbar_ax.set_yticks([])
        # connect some keys
        # self.cidk = self.figure.canvas.mpl_connect(
        #     'key_release_event', self.on_key_release)
        # self.cidm = self.figure.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        # self.cidm = self.figure.canvas.mpl_connect('', self.on_mouse_release)
        # self.figure.canvas.toolbar.home = self.show_image

        # change the toolbar functions
        NavigationToolbar2.home = self.show_image
        NavigationToolbar2.save = self.save_data

    def load_filenames(self):
        ls = os.listdir(self.dirname)
        self.filenames = []
        img_extensions = ('.png', '.jpg', '.bmp', '.jpg', '.jpeg', '.tif', '.tiff')
        for f in ls:
            if f.lower().endswith(img_extensions) and f[0] not in [".", "_"]:
                self.filenames += [os.path.join(self.dirname, f)]
        self.filenames.sort()

    def load_image(self):
        # print(self.curr_frame_index)
        self.image = PIL.Image.open(self.filenames[self.curr_frame_index])
        self.image = np.asarray(self.image)

    def show_image(self, *args):
        # print('show_image')
        # first plotthe image
        self.im = np.copy(self.image)
        colorvals = np.copy(self.colorvals)
        # remove values > vmax
        self.im[self.im > self.vmax.val] = 0
        self.figure.axes[0].get_images()[0].set_clim([self.vmin.val, self.vmax.val])
        self.figure.axes[0].get_images()[0].set_data(self.im)
        colorvals[colorvals > self.vmax.val] = 0
        self.cbar.set_array(np.repeat(colorvals[:, np.newaxis], 2, axis=-1))
        self.cbar.set_clim([self.vmin.val, self.vmax.val])
        # and the title
        self.title.set_text('%d - %s' % (self.curr_frame_index + 1,
                                         self.filenames[self.curr_frame_index].rsplit('/')[-1]))
        plt.draw()

    def change_frame(self, new_frame):
        # print('change_frame {} {}'.format(new_frame, int(new_frame)))
        self.curr_frame_index = int(new_frame)-1
        self.load_image()
        self.show_image()
        if self.data_changed:
            self.save_data()
            self.data_changed = False

    def nudge(self, direction):
        self.show_image()
        # self.change_frame(mod(self.curr_frame, self.num_frames))
        self.data_changed = True

    def on_key_release(self, event):
        # frame change
        if event.key in ("pageup", "alt+v", "alt+tab"):
            self.curr_frame.set_val(
                np.mod(self.curr_frame_index, self.num_frames))
        elif event.key in ("pagedown", "alt+c", "tab"):
            self.curr_frame.set_val(
                np.mod(self.curr_frame_index + 2, self.num_frames))
            print(self.curr_frame_index)
        elif event.key == "alt+pageup":
            self.curr_frame.set_val(
                np.mod(self.curr_frame_index - 9, self.num_frames))
        elif event.key == "alt+pagedown":
            self.curr_frame.set_val(
                np.mod(self.curr_frame_index + 11, self.num_frames))
        elif event.key == "home":
            self.curr_frame.set_val(1)
        elif event.key == "end":
            self.curr_frame.set_val(self.num_frames)
        # marker move
        elif event.key == "left":
            self.nudge(-1)
        elif event.key == "right":
            self.nudge(1)
        elif event.key == "up":
            self.nudge(-1j)
        elif event.key == "down":
            self.nudge(1j)
        elif event.key == "alt+left":
            self.nudge(-10)
        elif event.key == "alt+right":
            self.nudge(10)
        elif event.key == "alt+up":
            self.nudge(-10j)
        elif event.key == "alt+down":
            self.nudge(10j)

    def update_sliders(self, val):
        self.show_image()

    def on_mouse_release(self, event):
        self.change_frame(0)

    def save_data(self):
        print('save')
        for fn, val in zip(self.objects_to_save.keys(), self.objects_to_save.values()):
            np.save(fn, val)


class StackFilter():
    """Import image filenames filter images using upper and lower contrast bounds."""

    def __init__(self, fns=os.listdir("./")):
        """Import images using fns, a list of filenames."""
        self.fns = fns
        self.folder = os.path.dirname(self.fns[0])
        self.vals = None
        self.imgs = None

    def load_images(self, low_bound=0, upper_bound=np.inf):
        print("Loading images:\n")
        first_img = None
        for fn in self.fns:
            try:
                first_img = load_image(fn)
                break
            except:
                pass
        breakpoint()
        width, height = first_img.shape
        # self.imgs = np.zeros((len(self.fns), width, height), dtype=first_img.dtype)
        # use a memmap to store the stack of images
        memmap_fn = os.path.join(mkdtemp(), 'temp_volume.dat')
        self.imgs = np.memmap(
            memmap_fn, mode='w+',
            shape=(len(self.fns), width, height), dtype=first_img.dtype)
        for num, fn in enumerate(self.fns):
            try:
                img = np.copy(load_image(fn))
                keep = np.logical_and(img >= low_bound, img <= upper_bound)
                img[keep == False] = 0
                self.imgs[num] = img
            except:
                print(f"{fn} failed to load.")
            print_progress(num, len(self.fns))
        # self.imgs = np.array(imgs, dtype=first_img.dtype)
 
    def contrast_filter(self):
        self.contrast_filter_UI = tracker_window(dirname=self.folder)
        plt.show()
        # grab low and high bounds from UI
        self.low = int(np.round(self.contrast_filter_UI.vmin.val))
        self.high = int(np.round(self.contrast_filter_UI.vmax.val))
        print("Extracting coordinate data: ")
        self.load_images(low_bound=self.low, upper_bound=self.high)
        #  inds_to_remove = np.logical_or(self.imgs <= self.low, self.imgs > self.high)
        # self.imgs[inds_to_remove] = 0
        # np.logical_and(self.imgs <= self.high, self.imgs > self.low, out=self.imgs)
        # self.imgs = self.imgs.astype(bool, copy=False)
        # try:
        #     ys, xs, zs = np.where(self.imgs > 0)
        #     vals = self.imgs[ys, xs, zs]
        # except:
        # print(
        #     "coordinate data is too large. Using a hard drive memory map instead of RAM.")
        self.imgs_memmap = np.memmap(os.path.join(self.folder, "volume.npy"),
                                     mode='w+', shape=self.imgs.shape, dtype=bool)
        self.imgs_memmap[:] = self.imgs[:]
        del self.imgs
        self.imgs = None
        # xs, ys, zs, vals = [], [], [], []
        total_vals = self.imgs_memmap.sum()
        self.arr = np.zeros((total_vals, 3), dtype='uint16')
        self.vals = np.zeros(total_vals, dtype=self.imgs_memmap.dtype)
        last_ind = 0
        for depth, img in enumerate(self.imgs_memmap):
            num_vals = img.sum()
            y, x = np.where(img > 0)
            z = np.repeat(depth, len(x))
            arr = np.array([x, y, z])
            self.arr[last_ind: last_ind + num_vals] = arr.T
            self.vals[last_ind: last_ind + num_vals] = img[y, x]
            last_ind = last_ind + num_vals
            print_progress(depth + 1, len(self.imgs_memmap))
        # self.arr = np.array([xs, ys, zs], dtype=np.uint16).T
        # self.vals = np.array(vals)

    def get_limits(self):
        self.limits_UI = tracker_window(dirname=self.folder)
        plt.show()
        # grab low and high bounds from UI
        self.low = int(np.round(self.limits_UI.vmin.val))
        self.high = int(np.round(self.limits_UI.vmax.val))
        return self.low, self.high

    def pre_filter(self):
        self.contrast_filter_UI = tracker_window(dirname=self.folder)
        plt.show()
        # grab low and high bounds from UI
        self.low = self.contrast_filter_UI.vmin.val
        self.high = self.contrast_filter_UI.vmax.val
        folder = os.path.join(self.folder, 'prefiltered_stack')
        if not os.path.isdir(folder):
            os.mkdir(folder)
        # self.load_images(low_bound=self.low, upper_bound=self.high)
        # inds_to_remove = np.logical_or(self.imgs <= self.low, self.imgs > self.high)
        # self.imgs[inds_to_remove] = 0
        # self.imgs = self.imgs.astype('uint8', copy=False)
        # np.multiply(self.imgs, 255, out=self.imgs)
        print("Saving filtered images:\n")
        # for num, (fn, img) in enumerate(zip(self.fns, self.imgs)):
        #     base = os.path.basename(fn)
        #     new_fn = os.path.join(folder, base)
        #     save_image(new_fn, img)
        #     print_progress(num + 1, len(self.fns))
        for num, fn in enumerate(self.contrast_filter_UI.filenames):
            base = os.path.basename(fn)
            new_fn = os.path.join(folder, base)
            img = np.copy(load_image(fn))
            keep = np.logical_and(img >= self.low, img <= self.high)
            img[keep == False] = 0
            save_image(new_fn, img)
            print_progress(num + 1, len(self.fns))


class OmmatidiaGUI():
    def __init__(self, img_fn=None, img_arr=None, coords_fn=None,
                 coords_arr=None, pixel_size=1):
        """GUI for editing sets of coordinates superimposed on an image.


        Parameters
        ----------
        img_fn : str
            Filename of the image to use.
        img_arr : np.ndarry
            Array of the image to use. 
        coords_fn : str, shape=(N, 2)
            Filename of the data with coordinates.
        coords_arr : np.ndarry, shape=(N, 2)
            Array of the data with coordinates.
        """
        self.pixel_size = pixel_size
        self.img_fn = img_fn
        # load image if file was provided
        if self.img_fn is not None:
            self.img = load_image(img_fn)
        else:
            self.img = img_arr
        # load coordinates if file was provided
        if coords_arr is not None:
            self.coords_fn = "./coords.npy"
            self.coords = self.pixel_size * coords_arr
        else:
            self.coords_fn = coords_fn
            self.coords = self.pixel_size * np.load(coords_fn)
        # todo: if self.coords is empty, make sure it has the right shape
        if len(self.coords) == 0:
            self.coords = np.array([[], []]).T
        # make image and colorbar subplots
        self.fig = plt.figure(figsize=(12, 12))
        self.fig.suptitle(img_fn)
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 6, 1]) 
        self.button_ax = plt.subplot(gs[0])
        self.img_ax = plt.subplot(gs[1])
        self.img_ax.set_aspect('equal')
        self.cbar_ax = plt.subplot(gs[2])
        # plot the image as a pcolormesh so that coords are to scale
        height, width = self.img.shape[:2]
        xvals = self.pixel_size * np.arange(width)
        yvals = self.pixel_size * np.arange(height)
        xvals, yvals = np.meshgrid(xvals, yvals)
        if self.img.ndim == 3:
            self.img_ax.pcolormesh(xvals, yvals, self.img[..., 0],
                                   cmap='gray', shading='auto')
        elif self.img.ndim == 2:
            self.img_ax.pcolormesh(xvals, yvals, self.img,
                                   cmap='gray', shading='auto')
        self.img_ax.set(xlim=(0, self.pixel_size * width),
                        ylim=(0, self.pixel_size * height))
        self.fig.canvas.draw_idle()
        # self.img_ax.imshow(self.img)
        # plot the coordinates
        self.scatter = None
        self.plot_coords()
        # set pyplot to interactive mode
        # setup the window with radio buttons for the 3 interactive states
        self.radio = RadioButtons(self.button_ax, ('add', 'delete', 'clear'))
        self.radio.on_clicked(self.select_button)
        # format the radio button axis 
        self.button_ax.set_aspect('equal')
        self.select_function = self.add_coord
        # assign the select_function to the canvas selection
        self.mouse_press = self.fig.canvas.mpl_connect(
            'button_release_event', self.onclick)
        # assign hotkeys
        self.key_press = self.fig.canvas.mpl_connect(
            'key_release_event', self.keypress)
        plt.show()

    def plot_coords(self):
        """Plot all of the points in self.coords."""        
        self.cbar_ax.cla()
        # get the x and y limits before updating anything
        # xlim, ylim = self.img_ax.get_xlim(), self.img_ax.get_ylim()
        ys, xs = self.coords.T
        # calculate mean distance to nearest 3 neighbors
        if len(xs) > 1:
            self.dist_tree = scipy.spatial.KDTree(self.coords)
            dists, inds = self.dist_tree.query(self.coords, k=4)
            dists = dists[:, 1:]
            # diams = dists.mean(1)
            diams = np.array([d[np.isinf(d) == False].mean() for d in dists])
            # update the histogram colobar of diameters
            colorbar_histogram(diams, vmin=diams.min(), vmax=diams.max(),
                               ax=self.cbar_ax)
        # if a scatterplot already exists, remove it
        if self.scatter is not None:
            try:
                self.scatter.remove()
            except:
                pass
        if len(xs) > 1:
            self.scatter = self.img_ax.scatter(xs, ys, c=diams, marker='o')
        elif len(xs) == 1:
            self.scatter = self.img_ax.scatter(xs, ys, marker='o')
        
        # if self.scatter is None:
        #     if len(xs) > 1:
        #         self.scatter = self.img_ax.scatter(xs, ys, c=diams, marker='o')
        #     elif len(xs) == 1:
        #         self.scatter = self.img_ax.scatter(xs, ys, marker='o')
        # else:
        #     self.scatter.remove()
        #     if len(xs) > 1:
        #         self.scatter = self.img_ax.scatter(xs, ys, c=diams, marker='o')
        #     elif len(xs) > 0:
        #         self.scatter = self.img_ax.scatter(xs, ys, marker='o')
        #         # self.img_ax.set_xlim(xlim)
        #         # self.img_ax.set_xlim(ylim)
        self.fig.canvas.draw_idle()

    def add_coord(self, coord):
        """Add the selected coordinate to the list of coordinates and replot."""
        # only add point if its further than a pixel away from another coordinate
        if len(self.coords) > 0:
            if 'dist_tree' not in dir(self):
                self.dist_tree = scipy.spatial.KDTree(self.coords)
            nearest_dist, nearest_ind = self.dist_tree.query(coord, k=1)        
            success = False
            if nearest_dist > self.pixel_size:
                self.coords = np.vstack([self.coords, coord])
                success = True
                print(f"({coord[0]}, {coord[1]}) added to stored coordinates.")
            else:
                print(f"{coord} is too close to another stored point.")
        else:
            self.coords = np.vstack([self.coords, coord])
            success = True
        return success

    def delete_coord(self, coord):
        """Delete the nearest stored coordinate."""
        if len(self.coords) > 1:
            nearest_dist, nearest_ind = self.dist_tree.query(coord, k=1)
            self.coords = np.delete(self.coords, nearest_ind, axis=0)
        else:
            self.clear_coords()

    def clear_coords(self, coord):
        """Delete all stored coordinates."""
        nearest_dist, nearest_ind = self.dist_tree.query(coord, k=1)
        self.coords = np.array([[], []]).T

    def save_coords(self):
        """Save the edited array of coordinates."""
        np.save(self.coords_fn, self.coords / self.pixel_size)

    def select_button(self, label):
        """Set the interactive state using the radio buttons.


        Parameters
        ----------
        label : str
            The label output of self.radio.on_clicked(), used for selecting
            the interactive function (add_coord, delete_coord, or move_coord).
        """
        self.state = label
        if label == 'add':
            self.select_function = self.add_coord
        elif label == 'delete':
            self.select_function = self.delete_coord
        else:
            self.select_function = self.clear_coords
        self.fig.canvas.draw_idle()
        print(self.state)

    def onclick(self, event):
        """Run whenever the mouse click is released."""
        if event.inaxes == self.img_ax and self.fig.canvas.manager.toolbar.mode == '':
            x, y = event.xdata, event.ydata
            coord = [y, x]
            self.select_function(coord)
            self.save_coords()
            self.plot_coords()

    def keypress(self, event):
        """Run whenever a key is released."""
        if event.key == 'a':
            self.radio.set_active(0)
            self.select_button('add')
        elif event.key == 'd':
            self.radio.set_active(1)
            self.select_button('delete')


class ColorSelector:

    def __init__(self, image, bw=False, hue_only=False):
        """Initialize the ColorSelector GUI.

        Make a pyplot interactive figure with the original image to be 
        sampled, the processed image based on the sample region, and the hues,
        saturations, and values of the sample region.

        Parameters
        ----------
        image : np.ndarray
            The 2D image we want to filter.
        bw : bool, default=False
            Whether the image is grayscale.
        hue_only : bool, default=False
            Whether to use only the hue channel.
        """
        self.bw = bw
        self.hue_only = hue_only
        if isinstance(image, str):
            image = scipy.ndimage.imread(image)
        self.image = image
        self.image_hsv = rgb_to_hsv(self.image)
        self.mask = np.ones((self.image.shape[:2]), dtype=bool)
        self.color_range = np.array([[0, 0, 0], [1, 1, 255]])
        self.fig = matplotlib.pyplot.figure(figsize=(8, 8),
          num='Color Selector')
        self.grid = matplotlib.gridspec.GridSpec(6,
          2, width_ratios=[1, 3])
        self.original_image_ax = self.fig.add_subplot(self.grid[:3, 1])
        self.original_image_ax.set_xticks([])
        self.original_image_ax.set_yticks([])
        matplotlib.pyplot.title('Original Image')
        matplotlib.pyplot.imshow(self.image.astype('uint8'))
        self.masked_image_ax = self.fig.add_subplot(self.grid[3:, 1])
        self.masked_image_ax.set_xticks([])
        self.masked_image_ax.set_yticks([])
        matplotlib.pyplot.title('Masked Image')
        self.masked_im = self.masked_image_ax.imshow(self.image.astype('uint8'))
        self.plot_color_stats(init=True)

    def get_color_stats(self):
        """Calculate the histograms of the hues, saturations, and values.

        
        Atributes
        ---------
        hsv : np.ndarray
            The hues, saturations, and values per pixel of the filtered image.
        hue_dist : list
            The bin size and values of the hues in the sample.
        sat_dist : list
            The bin size and values of the saturations in the sample.
        val_dist : list
            The bin size and values of the values in the sample.
        """
        self.sample_hsv = self.image_hsv[self.mask]
        self.hue_dist = list(np.histogram((self.sample_hsv[:, 0]),
          255, range=(0, 1), density=True))
        self.hue_dist[0] = np.append(self.hue_dist[0], self.hue_dist[0][0])
        self.sat_dist = np.histogram((self.sample_hsv[:, 1]),
          255, range=(0, 1), density=True)
        self.val_dist = np.histogram((self.sample_hsv[:, 2]),
          255, range=(0, 255), density=True)

    def plot_color_stats(self, init=False):
        """Initialize or update the plots for hues, saturations, and values.

        Parameters
        ----------
        init : bool, default=False
            Whether to initialize the plots.
        """
        self.get_color_stats()
        if init:
            self.hues = self.fig.add_subplot((self.grid[0:2, 0]), polar=True)
            matplotlib.pyplot.title('Hues')
            radii, theta = np.array([0, self.image.size]), np.linspace(0, 2 * np.pi, 256)
            colorvals = np.arange(256) / 256
            colorvals = np.array([colorvals, colorvals])
            self.hues.pcolormesh(theta, radii, colorvals, cmap='hsv', shading='nearest')
            self.hues.set_xticks([])
            self.hues.set_xticklabels([])
            self.hues.set_rticks([])
            self.sats = self.fig.add_subplot(self.grid[2:4, 0])
            self.sats.set_xticks([0, 0.5, 1])
            self.sats.set_yticks([])
            matplotlib.pyplot.title('Saturations')
            xs, ys = self.sat_dist[1], np.array([0, self.image.size])
            self.sats.pcolormesh(xs, ys, colorvals, cmap='Blues', shading='nearest')
            self.vals = self.fig.add_subplot(self.grid[4:, 0])
            self.vals.set_xticks([0, 128, 255])
            self.vals.set_yticks([])
            matplotlib.pyplot.title('Values')
            xs, ys = self.val_dist[1], np.array([0, self.image.size])
            self.vals.pcolormesh(xs, ys, (colorvals[:, ::-1]), cmap='Greys',
              shading='nearest')
            self.h_line, = self.hues.plot(2 * np.pi * self.hue_dist[1], self.hue_dist[0], 'k')
            self.s_line, = self.sats.plot(self.sat_dist[1][1:], self.sat_dist[0], 'r')
            self.sats.set_xlim(0, 1)
            self.v_line, = self.vals.plot(self.val_dist[1][1:], self.val_dist[0], 'r')
            self.vals.set_xlim(0, 255)
            self.satspan = self.sats.axvspan((self.color_range[0][1]),
              (self.color_range[1][1]), color='k',
              alpha=0.3)
            self.valspan = self.vals.axvspan((self.color_range[0][2]),
              (self.color_range[1][2]), color='k',
              alpha=0.3)
            self.fig.tight_layout()
        else:
            self.h_line.set_ydata(self.hue_dist[0])
            self.s_line.set_ydata(self.sat_dist[0])
            self.v_line.set_ydata(self.val_dist[0])
            self.satspan.set_xy(self.get_axvspan(self.color_range[0][1], self.color_range[1][1]))
            self.valspan.set_xy(self.get_axvspan(self.color_range[0][2], self.color_range[1][2]))
        self.hues.set_rlim(rmin=(-0.5 * self.hue_dist[0].max()),
          rmax=(1 * self.hue_dist[0].max()))
        self.sats.set_ylim(ymin=0, ymax=(self.sat_dist[0].max()))
        self.vals.set_ylim(ymin=0, ymax=(self.val_dist[0].max()))

    def select_color(self, dilate_iters=5):
        """Generate a mask based on the selected colors and dilated.

        Parameters
        ----------
        dilate_iters : int, default=5
            Number of iterations to apply the binary dilation to the mask.

        Attributes
        ----------
        lows : np.ndarray
            Minimum hue, saturation, and value of the region distribution.
        highs : np.ndarray
            Maximum hue, saturation, and value of the region distribution.
        mask : np.ndarray
            2D masking boolean array of image using selected color range.

        Returns
        -------
        keyed : np.ndarray
            The image including only the pixels within the selected color range.
        """
        self.lows, self.highs = self.color_range.min(0), self.color_range.max(0)
        hue_low, hue_high = self.lows[0], self.highs[0]
        include = np.logical_and(
            self.image_hsv > self.lows[(np.newaxis, np.newaxis)],
            self.image_hsv < self.highs[(np.newaxis, np.newaxis)])
        # if hue is out of bounds:
        if hue_low < 0:
            hue_low = 1 + hue_low
            include[..., 0] = np.logical_or(
                self.image_hsv[(..., 0)] > hue_low, self.image_hsv[(..., 0)] < hue_high)
        # use hues if option selected
        if self.hue_only:
                self.mask = include[..., 0]
        # use vals if grascale image
        elif self.bw:
            self.mask = include[..., 2]
        # use intersection of constraints otherwise
        else:
            self.mask = np.product(include, axis=(-1)).astype(bool)
        # dilate the mask
        if dilate_iters > 0:
            self.mask = scipy.ndimage.morphology.binary_dilation((self.mask),
              iterations=dilate_iters).astype(bool)
        # make masked version of image
        keyed = self.image.copy()
        keyed[self.mask == False] = [0, 0, 0]
        return keyed

    def onselect(self, eclick, erelease):
        """Update image based on rectangle between eclick and erelease."""
        self.select = self.image[
            int(eclick.ydata):int(erelease.ydata),
            int(eclick.xdata):int(erelease.xdata)]
        self.select_hsv = self.image_hsv[
            int(eclick.ydata):int(erelease.ydata),
            int(eclick.xdata):int(erelease.xdata)]
        if self.select.shape[0] != 0:
            if self.select.shape[1] != 0:
                means = self.select_hsv.mean((0, 1))
                standard_dev = self.select_hsv.std((0, 1))
                h_mean = scipy.stats.circmean(self.select_hsv[(..., 0)].flatten(), 0, 1)
                h_std = abs(scipy.stats.circstd(self.select_hsv[(..., 0)].flatten(), 0, 1))
                means[0], standard_dev[0] = h_mean, h_std
                self.color_range = np.array([
                 means - 3 * standard_dev, means + 3 * standard_dev])
                self.masked_image = self.select_color()
                # fig = plt.figure()
                # plt.imshow(self.masked_image.astype('uint8'))
                # plt.show()
                self.masked_im.set_array(self.masked_image.astype('uint8'))
                self.plot_color_stats()
                self.fig.canvas.draw()

    def toggle_selector(self, event):
        """Keyboard shortcuts to close the window and toggle the selector."""
        print(' Key pressed.')
        if event.key in ('Q', 'q'):
            if self.RS.active:
                matplotlib.pyplot.close()
        if event.key in ('A', 'a'):
            if not self.RS.active:
                print(' RectangleSelector activated.')
                self.RS.set_active(True)

    def get_axvspan(self, x1, x2):
        """Get corners for updating the axvspans."""
        return np.array([
         [
          x1, 0.0],
         [
          x1, 1.0],
         [
          x2, 1.0],
         [
          x2, 0.0],
         [
          x1, 0.0]])

    def displaying(self):
        """True if the GUI is currently displayed."""
        return matplotlib.pyplot.fignum_exists(self.fig.number)

    def start_up(self):
        """Run when ready to display."""
        self.RS = matplotlib.widgets.RectangleSelector((self.original_image_ax),
          (self.onselect), drawtype='box')
        matplotlib.pyplot.connect('key_press_event', self.toggle_selector)
        matplotlib.pyplot.show()


def colorbar_histogram(colorvals, vmin, vmax, ax=None, bin_number=100, fill_color='k',
                       line_color='w', colormap='viridis'):
    """Plot a colorbar with a histogram skyline superimposed.

    Parameters
    ----------
    colorvals : array-like
        List of values corresponding to colors drawn from the colormap.
    vmin : float
        Minimum colorvalue to include in the histogram and colorbar.
    vmin : float
        Maximum colorvalue to include in the histogram and colorbar.
    ax : matplotlib.axes._subplots.AxesSubplot
        The pyplot axis in which to plot the histogram and colorbar.
    bin_number : int, default=100
        The number of bins to use in plotting the histogram.
    fill_color : matplotlib color, default='k'
        Color for filling the space under the histogram. Default is black.
    line_color : matplotlib color, default='w'
        Color for the histogram skyline.
    colormap : matplotlib colormap, default='viridis'
        Colormap of colorvals to colors.
    """
    if not all([vmin < np.inf, vmax < np.inf, not np.isnan(vmin), not np.isnan(vmax)]):
        raise AssertionError('Input vmin and vmax should be finite floats')
    else:
        if ax is None:
            ax = plt.gca()
    colorvals = np.asarray(colorvals)
    bins = np.linspace(vmin, vmax, bin_number + 1)
    counts, bin_edges = np.histogram(colorvals, bins=bins)
    bin_edges = np.repeat(bins, 2)[1:-1]
    heights = np.repeat(counts, 2)
    ax.plot(heights, bin_edges, color=line_color)
    ax.fill_betweenx(bin_edges, heights, color=fill_color, alpha=0.3)
    vals = np.linspace(vmin, vmax)
    C = vals
    X = np.array([0, counts.max()])
    Y = np.repeat((vals[:, np.newaxis]), 2, axis=(-1))
    ax.pcolormesh(X, C, Y, cmap=colormap, zorder=0,
      vmin=vmin,
      vmax=vmax,
      shading='nearest')
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    ax.yaxis.set_label_position('right')
    ax.yaxis.tick_right()
    ax.set_ylim(vmin, vmax)
    ax.set_xlim(0, counts.max())

def load_image(fn):
    """Import an image as a numpy array using the PIL."""
    return np.asarray(PIL.Image.open(fn))

def save_image(fn, arr):
    """Save an image using the PIL."""
    img = PIL.Image.fromarray(arr)
    if os.path.exists(fn):
        os.remove(fn)
    return img.save(fn)

