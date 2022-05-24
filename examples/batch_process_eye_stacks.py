#!/usr/bin/env python

"""Batch process all folders of eye stacks and save.


Assumes the following folders of stacks of .jpg images and binary mask images:

.\
|--batch_process_eye_stacks.py
|--stack_1\
   |--img_001.jpg
   |--img_002.jpg
   |...
   |--mask.png
|--stack_1_ommatidia.jpg (outcome)
|--stack_2\
   |--img_001.jpg
   |--img_002.jpg
   |...
   |--mask.png
|--stack_2_ommatidia.jpg (outcome)
|--ommatidia_data.csv
|--_hidden_folder\
   |(skipped files)
   |...
"""
import os
from scipy import misc
from analysis_tools import *
import pandas as pd


# Custom parameters
PIXEL_SIZE = (488.84896*2)**-1  # measured manually
DEPTH_SIZE = .004*3            # from the mircroscope manual
BRIGH_PEAK = False             # True assumes a bright point for every peak
HIGH_PASS = True               # True adds a high-pass filter to the low-pass used in the ODA
SQUARE_LATTICE = True          # True assumes only two fundamental gratings

# make dictionary to store relevant information
values = {
    "dirname":[], "surface_area":[], "eye_length":[], "eye_width":[],
    "radius":[], "fov_hull":[], "fov_long":[], "fov_short":[], 
    "ommatidia_count":[], "ommatidial_diameter":[], "ommatidial_diameter_std":[],
    "ommatidial_diameter_fft":[], "io_angle":[], "io_angle_std":[],
    "io_angle_fft":[]
}
params = ["dirname", "surface_area", "eye_length"]

# load filenames and folders
fns = os.listdir(os.getcwd())
img_fns = [fn for fn in fns if fn.endswith(".jpg")]
folders = [fn for fn in fns if os.path.isdir(fn)]
folders = [os.path.join(os.getcwd(), f) for f in folders]
# for each folder
for folder in folders:
    # skip hidden folders
    base = os.path.basename(folder)
    if not base.startswith("_"):
        print(folder)
        # get stack name from the folder name
        path, base = os.path.split(folder)
        stack_name = f"{base}_ommatidia.jpg"
        ommatidia_fig_fn = os.path.join(path, stack_name)
        # get the eye stack and save the image with the ommatidia superimposed
        st = EyeStack(folder, f_type=".TIF", mask_fn=os.path.join(folder, "mask.png"),
                      pixel_size=PIXEL_SIZE, depth_size=DEPTH_SIZE)
        st.oda_3d(high_pass=HIGH_PASS, plot_fn=ommatidia_fig_fn,
                  bright_peak=BRIGH_PEAK, square_lattice=SQUARE_LATTICE)
        # rename relevant parameters so they can be accessed using getattr
        st.radius = st.sphere.radius
        st.eye_width = st.eye.eye_width
        st.eye_length = st.eye.eye_length
        st.ommatidia_count = len(st.eye.ommatidia)
        st.ommatidial_diameter = st.eye.ommatidial_diameter
        st.ommatidial_diameter_std = st.eye.ommatidial_diameter.std()
        st.ommatidial_diameter_fft = st.eye.ommatidial_diameter_fft
        st.io_angle = st.io_angle * 180 / np.pi
        st.io_angle_std = st.io_angles.std() * 180 / np.pi
        st.io_angle_fft = st.io_angle_fft * 180 / np.pi
        fov_long, fov_short = max(st.fov_long, st.fov_short), min(st.fov_long, st.fov_short)
        st.fov_long = fov_long
        st.fov_short = fov_short
        # store relevant parameters
        for key in values.keys():
            values[key] += [getattr(st, key)]
        print()

dataframe = pd.DataFrame(values)
dataframe.to_csv("eye_stack_data.csv", index=False)

