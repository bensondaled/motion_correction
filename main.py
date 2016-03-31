# Refer to INSTRUCTIONS.txt for details on how to use this mini-package.
# Remember, whitespace matters in Python, so don't mess with tabs in this file.

import sys, os, time
import numpy as np
from skimage.io import imread, imsave
from scipy.io import savemat
from motion import motion_correct

### Paths ###
input_directory     =   '/jukebox/wang/deverett/'#'/jukebox/path_to_my_input_folder'     # a folder containing the tif files to be corrected. do not edit contents once job is submitted!
output_directory    =   '/jukebox/wang/deverett/'#'/jukebox/path_to_my_output_folder'    # a folder in which to save the motion-corrected movies

### Saving Parameters ###
save_tif         =     True
save_vals_func   =     savemat # savemat / np.save / False

### Algorithm Parameters ###
template        =   np.median           # a function used to calculate the template, examples: np.mean, np.median. Beware: median is expensive for large datasets
max_shift       =   [8, 8]              # maximum shift in pixels allowed in each dimension (y, x)
reslice         =   slice(None, None)   # take every n, for example slice(1,None,2) takes every 2 starting at 2nd
resample        =   50                  # nonoverlapping rolling mean of this window size before computing template
symmetrize      =   True                # True / False
crop            =   False               # True / False, whether or not to crop the movie such that borders introduced by motion correction are removed
shift_threshold =   3.0                 # when absolute shifts in both axes are less than this, iterations stop
max_iters       =   2                   # max number of iterations


### Motion correct ###

# structure params
compute_params = dict(template=template, max_shift=max_shift, reslice=reslice, resample=resample, symmetrize=symmetrize)
apply_params = dict(crop=crop, reslice=reslice)
correct_params = dict(shift_threshold=shift_threshold, max_iters=max_iters)

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
fname_vals = '{}_mcvals'.format(os.path.splitext(fname)[0])
path_cor = os.path.join(output_directory, fname_cor)
path_vals = os.path.join(output_directory, fname_vals)
print('Motion correcting file: {}'.format(fname)); sys.stdout.flush()

# load in tif, correct
t0 = time.time()
mov = imread(path)
mov_cor, template, vals = motion_correct(mov, compute_kwargs=compute_params, apply_kwargs=apply_params, **correct_params)
print('Motion correction took {:0.3f} seconds.'.format(time.time()-t0)); sys.stdout.flush()

# save out results
if save_tif:
    imsave(path_cor, mov_cor)
if save_vals_func:
    if save_vals_func == savemat:
        vals = dict(vals=vals)
    save_vals_func(path_vals, vals)
print('Done file: {}'.format(fname))
