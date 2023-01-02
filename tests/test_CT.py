"""Test script for processing a CT stack.
"""
# import sys
# sys.path.append("../src/")
from ODA import *

# set important parameters
PIXEL_SIZE = .325
PIXEL_SIZE *= 4                 # the bin length
DEPTH_SIZE = PIXEL_SIZE
POLAR_CLUSTERING = True         # spherical eye
# load the CT stack in this same tests folder
ct_directory = "d_mauritiana_ct_stack"
ct_stack = CTStack(dirname=ct_directory, img_extension=".tif", bw=True,
                   pixel_size=PIXEL_SIZE, depth_size=DEPTH_SIZE)
ct_stack.ommatidia_detecting_algorithm(polar_clustering=True)
# plot the data using the 3D GUI
ct_stack.plot_data_3d()