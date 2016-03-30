# Refer to INSTRUCTIONS.txt for details on how to use this mini-package.
# Remember, whitespace matters in Python, so don't mess with tabs in this file.

import sys, os
import numpy as np
from skimage.io import imread, imsave
from motion import motion_correct

### Paths ###
input_directory     =   '/jukebox/path_to_my_input_folder'     # a folder containing the tif files to be corrected. do not edit contents once job is submitted!
output_directory    =   '/jukebox/path_to_my_output_folder'    # a folder in which to save the motion-corrected movies

### Parameters ###
template    =   np.median   # a function used to calculate the template, examples: np.mean, np.median. Beware: median is expensive for large datasets
max_shift   =   [8, 8]      # maximum shift in pixels allowed in each dimension (y, x)
crop        =   False       # True / False, whether or not to crop the movie such that borders introduced by motion correction are removed
shift_threshold = 3.0
max_iters = 100


### Motion correct ###

# structure params
correct_params = dict(shift_threshold=shift_threshold, max_iters=max_iters)
compute_params = dict(template=template, max_shift=max_shift)
apply_params = dict(crop=crop)

# determine job id
jobid = int(sys.argv[1])

# collect file names (assumes files are not changing while job runs)
filenames = sorted([f for f in os.listdir(input_directory) if f.endswith('.tif')])

# job number checking
if jobid >= len(filenames):
    print('Job ID is greater than number of files.')
    sys.exit(0)

# file names
fname = filenames[jobid]
path = os.path.join(input_directory, fname)
fname_cor = '{}_mc.tif'.format(os.path.splitext(fname)[0])
path_cor = os.path.join(output_directory, fname_cor)
print('Motion correcting file: {}'.format(fname)); sys.stdout.flush()

# load in tif, correct, and save
mov = imread(path)
mov_cor, template, vals = motion_correct(mov, compute_kwargs=compute_params, apply_kwargs=apply_params, **correct_params)
imsave(path_cor, mov_cor)
print('Done file: {}'.format(fname))
