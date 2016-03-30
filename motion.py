import numpy as np
import cv2
PF_numeric_types = [int, float, np.float16, np.float32, np.float64, np.float128, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]

def motion_correct(mov, max_iters=1, shift_threshold=1., in_place=True, verbose=True, compute_kwargs={}, apply_kwargs={}):
    """Perform motion correction using template matching.

    max_iters : int
        maximum number of iterations
    shift_threshold : float
        absolute max shift value below which to exit
    in_place : bool
        perform on same memory as supplied
    verbose : bool
        show status
    compute_kwargs : dict
        kwargs for compute_motion_correction
    apply_kwargs : dict
        kwargs for apply_motion_correction
   
    Returns
    --------
    corrected movie, template, values

    Note that this function is a convenience function for calling compute_motion_correction followed by apply_motion_correction, multiple times and combining results
    """
    if not in_place:
        mov = mov.copy()
  
    all_vals = []
    for it in range(max_iters):
        if verbose:
            print('Iteration {}'.format(it))
        template,vals = compute_motion(mov, **compute_kwargs)
        mov = apply_motion_correction(mov, vals, **apply_kwargs)
        maxshifts = np.abs(vals[:,[0,1]].max(axis=0))
        all_vals.append(vals)
        if verbose:
            print('Shifts: {}'.format(str(maxshifts)))
        if np.all(maxshifts < shift_threshold):
            break

    # combine values from iterations
    all_vals = np.array(all_vals)
    return_vals = np.empty([all_vals.shape[1],all_vals.shape[2]])
    return_vals[:,[0,1]] = all_vals[:,:,[0,1]].sum(axis=0)
    return_vals[:,2] = all_vals[:,:,2].mean(axis=0)

    return mov,template,return_vals


def apply_motion_correction(mov, shifts, interpolation=cv2.INTER_LINEAR, crop=False, in_place=False):
    """Apply shifts to mov in order to correct motion

    Parameters
    ----------
    mov : pyfluo.Movie
        input movie
    shifts : np.ndarray
        obtained from the function compute_motion, list of [x_shift, y_shift] for each frame. if more than 2 columns, assumes first 2 are the desired ones
    interpolation : def
        interpolation flag for cv2.warpAffine, defaults to cv2.INTER_LINEAR
    crop : bool / int
        whether to crop image to borders of correction. if True, crops to maximum adjustments. if int, crops that number of pixels off all sides
    in_place : bool
        in place

    This supports the correction of single frames as well, given a single shift
    """
    if not in_place:
        mov=mov.copy()

    if shifts.dtype.names:
        shifts = shifts[['y_shift','x_shift']].view((float, 2))
    elif shifts.shape[-1] == 3:
        shifts = shifts[:,[0,1]]

    if mov.ndim==2:
        mov = mov[None,...]
    if shifts.ndim==1:
        shifts = shifts[None,...]

    assert shifts.ndim==2 and shifts.shape[1]==2

    t,h,w=mov.shape
    for i,frame in enumerate(mov):
        sh_x_n, sh_y_n = shifts[i]
        M = np.float32([[1,0,sh_y_n],[0,1,sh_x_n]])                 
        mov[i] = cv2.warpAffine(frame,M,(w,h),flags=interpolation)

    if crop:
        if crop == True:
            ymax = min([0, min(shifts[:,0])]) or None
            xmax = min([0, min(shifts[:,1])]) or None
            ymin = max(shifts[:,0])
            xmin = max(shifts[:,1])
        elif any([isinstance(crop, dt) for dt in PF_numeric_types]):
            ymax,xmax = -crop,-crop
            ymin,xmin = crop,crop
        mov = mov[:, ymin:ymax, xmin:xmax]

    return mov.squeeze()

def compute_motion(mov, max_shift=(5,5), template=np.median, template_matching_method=cv2.TM_CCORR_NORMED, reslice=slice(None,None), resample=1, symmetrize=True):
        """Compute template, shifts, and correlations associated with template-matching-based motion correction

        Parameters
        ----------
        mov : pyfluo.Movie
            input movie
        max_shift : int / list-like
            maximum number of pixels to shift frame on each iteration (by axis if list-like)
        template : np.ndarray / def
            if array, template to be used. if function, that used to compute template (defaults to np.median)
        template_matching_method : opencv constant
            method parameter for cv2.matchTemplate
        reslice : slice
            used to reslice movie, example: slice(1,None,2) gives every other frame starting from 2nd frame
        resample : int
            average over n frames in nonoverlapping windows before running template operation
        symmetrize : bool
            enforces that for a given axis (x,y), max(shifts) = |min(shifts)|
        
        Returns
        -------
        template: np.ndarray
            the template used
        shifts : np.ndarray
            one row per frame, see array's dtype for details
        """
      
        # Parse movie
        mov = mov.astype(np.float32)
        mov = mov[reslice]
        n_frames,h_i, w_i = mov.shape

        # Parse max_shift param
        if type(max_shift) in [int,float]:
            ms_h = max_shift
            ms_w = max_shift
        elif type(max_shift) in [tuple, list, np.ndarray]:
            ms_h,ms_w = max_shift
        else:
            raise Exception('Max shift should be given as value or 2-item list')
       
        # Parse/generate template
        if callable(template):
            if resample > 1:
                resample = int(resample)
                resampled = np.array([np.mean(mov[i:i+resample], axis=0) for i in np.arange(0,len(mov),resample)])
            else:
                resampled = mov
            template=template(resampled, axis=0)            
        elif not isinstance(template, np.ndarray):
            raise Exception('Template parameter should be an array or function')
        template = template.astype(np.float32)
        template=template[ms_h:h_i-ms_h,ms_w:w_i-ms_w]
        h,w = template.shape
        
        vals = np.zeros([n_frames,3])

        for i,frame in enumerate(mov):

            res = cv2.matchTemplate(frame, template, template_matching_method)
            avg_metric = np.mean(res)
            top_left = cv2.minMaxLoc(res)[3]
            sh_y,sh_x = top_left
            bottom_right = (top_left[0] + w, top_left[1] + h)
        
            if (0 < top_left[1] < 2 * ms_h-1) & (0 < top_left[0] < 2 * ms_w-1):
                # if max is internal, check for subpixel shift using gaussian peak registration
                log_xm1_y = np.log(res[sh_x-1,sh_y])             
                log_xp1_y = np.log(res[sh_x+1,sh_y])             
                log_x_ym1 = np.log(res[sh_x,sh_y-1])             
                log_x_yp1 = np.log(res[sh_x,sh_y+1])             
                four_log_xy = 4*np.log(res[sh_x,sh_y])
    
                sh_x_n = -(sh_x - ms_h + (log_xm1_y - log_xp1_y) / (2 * log_xm1_y - four_log_xy + 2 * log_xp1_y))
                sh_y_n = -(sh_y - ms_w + (log_x_ym1 - log_x_yp1) / (2 * log_x_ym1 - four_log_xy + 2 * log_x_yp1))
            else:
                sh_x_n = -(sh_x - ms_h)
                sh_y_n = -(sh_y - ms_w)
                    
            vals[i] = [sh_x_n, sh_y_n, avg_metric]

        if symmetrize:
            vals[:,[0,1]] -= (vals[:,[0,1]].max(axis=0) + vals[:,[0,1]].min(axis=0)) / 2.
            assert np.all(np.round(vals[:,[0,1]].max(axis=0),3) == np.round(np.abs(vals[:,[0,1]].min(axis=0)),3))

        return template, vals
