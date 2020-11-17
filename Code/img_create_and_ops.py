# Image manipulation, creation etc. 
# # ----------------------------------------------------------------------------------
# # ----------------------------------------------------------------------------------

import cv2
import numpy as np

def make_bw_img(arr, dilate=False):
    """
    Returns an image made from the array. If dilate is True, returns a dilated image too
    """

    dilate_kernel = np.ones((4, 4), np.uint8)
    im = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if dilate:
        im_d = cv2.dilate(im, dilate_kernel)
        return im_d
    else:
        return im