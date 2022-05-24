"""Test script for processing a CT stack.
"""
import sys
sys.path.append("h:\\My Drive\\backup\\Desktop\\programs\\eye_tools\\src\\eye_tools")
# sys.path = sys.path[1:]
# import analysis_tools as et
from analysis_tools import *
# import logging
# logging.basicConfig(level=logging.DEBUG)


# PIXEL_SIZE = 4.50074
PIXEL_SIZE = .325
PIXEL_SIZE *= 8                 # the bin length
DEPTH_SIZE = PIXEL_SIZE

# ct_directory = "/home/pbl/Downloads/moth_eye/"
ct_directory = "H:\\My Drive\\backup\\Downloads\\Tam16_M_17_01\\prefilter"
img_ext = ".tif"
# make a ct stack object using the ct stack folder
# ct_stack = CTStack(dirname=ct_directory, img_extension=img_ext, bw=True,
#                    pixel_size=PIXEL_SIZE, depth_size=DEPTH_SIZE)
# ct_stack.prefilter(low=5000)
# make a CT stack using the prefiltered_stack
# ct_directory = "../../../research/method__compound_eye_tools/data/CT/manuscript/manduca_sexta/_prefiltered_stack"
# make a ct stack object using the ct stack folder
# stack = Stack(dirname=ct_directory, img_extension=img_ext, bw=True)


ct_stack = CTStack(dirname=ct_directory, img_extension=img_ext, bw=True,
                   pixel_size=PIXEL_SIZE, depth_size=DEPTH_SIZE)
ct_stack.ommatidia_detecting_algorithm(
    polar_clustering=True, display=False, test=False, three_d=False)
# 

# ct_stack.ommatidial_data.spherical_IOA *= 2
# include = ct_stack.ommatidial_data['size'] > 1
# ct_stack.ommatidial_data = ct_stack.ommatidial_data[include]
# ct_stack.plot_ommatidial_data(image_size=500, scatter=True)
# ct_stack.plot_interommatidial_data(three_d=True)
