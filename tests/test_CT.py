"""Test script for processing a CT stack.
"""
import sys
# sys.path.append("../src/ODA/")
# from analysis_tools import *
from volume_gui import *
import os
from subprocess import call
from ODA import *

# set important parameters
PIXEL_SIZE = .325 * 4      # pixel length * bin length
DEPTH_SIZE = PIXEL_SIZE    # cube voxels
POLAR_CLUSTERING = True    # spherical eye
# load the CT stack in this same tests folder
ct_directory = "d_mauritiana_ct_stack"
ct_stack = CTStack(dirname=ct_directory, img_extension=".tif", bw=True,
                   pixel_size=PIXEL_SIZE, depth_size=DEPTH_SIZE)
ct_stack.ommatidia_detecting_algorithm(polar_clustering=True)

# plot the data using the 3D GUI
# note: use only one of the following by commenting one and uncommenting the other

# for the older 3D scatterplots:
# ct_stack.plot_data_3d()

# for the newer volume rendering:
cwd = os.getcwd()
path = os.path.join(cwd, 'volume_gui.py')
del ct_stack
gui = VolumeGUI(raw_stack_folder=ct_directory, dirname=ct_directory, 
                img_extension='.tif', bw=True,
                pixel_size=PIXEL_SIZE, depth_size=PIXEL_SIZE, 
                zoffset=0, bin_length=1, raw_bin_length=1)