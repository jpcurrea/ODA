#!/usr/bin/env python
"""A command line tool to preview ODA results in 3D."""
from ODA import CTStack
import sys

PIXEL_SIZE = .325               # mm
BIN_SIZE = 4                    # voxel bin size
PIXEL_SIZE *= BIN_SIZE          # adjust pixel size by bin size


if __name__=="__main__":
    # check for user specified directory name
    # use current directory by default
    folder = "./"
    if len(sys.argv) > 1:
        folder = sys.argv[1]
    stack = CTStack(dirname=folder, img_extension='.tif', bw=True,
                    pixel_size=PIXEL_SIZE, depth_size=PIXEL_SIZE)
    stack.plot_data_3d()
