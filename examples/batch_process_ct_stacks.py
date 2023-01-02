"""Batch process all folders of eye stacks and save.

Assumes the following folders of stacks of .jpg images and binary mask images:

.\
|--batch_process_ct_stacks.py
|--stack_1\
    |--img_001.jpg
    |--img_002.jpg
    |...
    |--pixel.txt                 (contains pixel length)
    |--_prefiltered_stack\
        |--img_001.jpg
        |--img_002.jpg
        |--...
    |--ommatidial_data.csv       (outcome)
    |--interommatidial_data.csv  (outcome)
|--stack_1_stats.csv             (outcome)
|--stack_2\
    |--img_001.jpg
    |--img_002.jpg
    |...
    |--pixel.txt                 (contains pixel length)
    |--_prefiltered_stack\
        |--img_001.jpg
        |--img_002.jpg
        |--...
    |--ommatidial_data.csv       (outcome)
    |--interommatidial_data.csv  (outcome)
|--stack_2_stats.csv             (outcome)
|...
|--_hidden_folder\
    |(skipped files)
    |...
"""
from ODA import *
import cProfile

fns = os.listdir()

folders = [
    'drosophila_mauritiana\\_prefiltered_stack',
    'manduca_sexta\\_prefiltered_stack',
    'deilephila_elpenor\\_prefiltered_stack',
    'apis_mellifera\\_prefiltered_stack',
    ]

pixel_lengths = [4*.325, 4.50074, 3.325, 1.6]
window_lengths = [np.pi/3, np.pi/3, np.pi/3, np.pi/10]

polar_clustering = [True, True, True, False]

ind = 0
for folder, pixel_length, polar_cluster, window_length in zip(
        folders[ind:], pixel_lengths[ind:], polar_clustering[ind:],
        window_lengths[ind:]):
    print(folder)
    # make a CTStack object
    stack = CTStack(dirname=folder, img_extension='.tif', bw=True,
                    pixel_size=pixel_length, depth_size=pixel_length)
    # start a profiler for measuring the program's performance
    profiler = cProfile.Profile()
    profiler.enable()
    stack.ommatidia_detecting_algorithm(neighborhood_smoothing=5, window_length=window_length,
                                        polar_clustering=polar_clustering, prefiltered=True, stage=0)
    # stop the profiler
    profiler.disable()
    fn_stats = folder + "_profiler_log.txt"
    profiler.dump_stats(fn_stats)
    stack.stats_summary()
    print(stack.stats)
    fn = f"{folder}_stats.csv"
    stack.stats.to_csv(fn)