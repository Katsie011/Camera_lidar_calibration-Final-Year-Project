# Matrix Operations:
# -------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------

import numpy as np
from img_create_and_ops import *

import numpy as np
import cv2
from matplotlib import pyplot as plt




def project_velo_to_cam0(cal):
    """
    Return: the projection matrix for Velo --> Cam0
    In: pykitti calibration object
    """
    P_velo2cam_ref = (cal.T_cam0_velo)
    R_ref2rect = np.linalg.inv((cal.R_rect_00))
    P_rect2cam2 = (cal.P_rect_00)

    proj_mat = P_rect2cam2 @ R_ref2rect @ P_velo2cam_ref
    return proj_mat


def project_cam0_to_velo(cal):
    """
    Return: the projection matrix for Cam0 --> Velo
    In: pykitti calibration object
    """
    R_ref2rect = (cal.R_rect_00)
    R_ref2rect_inv = np.linalg.inv(R_ref2rect)

    velo2cam_ref = data.calib.T_cam0_velo
    P_cam_ref2velo = np.linalg.inv(velo2cam_ref)

    proj_mat = R_ref2rect_inv @ P_cam_ref2velo
    return proj_mat


def project_camera_to_lidar(points, proj_mat):
    """
    Args:
        points:     3D points in camera coordinate [3, npoints]
        proj_mat:   Projection matrix [3, 4]
    Returns:
        points in lidar coordinate:     [3, npoints]
    """
    points = points.T
    len_pts = points.shape[1]
    # Change to homogenous coordinate
    #     print(points, np.ones((1, len_pts)))
    points = np.vstack((points, np.ones((1, len_pts))))
    points = proj_mat @ points
    return points[:3, :].T


def project_to_2Dimage(points, proj_mat):
    """
    Apply the perspective projection
    Args:
        pts_3d:     3D points in camera coordinate [npoints, 3]
        proj_mat:   Projection matrix [3, 4]
    """
    len_pts = points.shape[0]
    # Change to homogenous coordinate with form (x, y, z, 1)T

    #     print(points, np.ones((1, len_pts)))
    points = np.hstack((points, np.ones((len_pts, 1))))
    points = points @ proj_mat.T
    #     print(points)
    points[:, :2] /= points[:, 2][:, np.newaxis]
    return points[:, :2]





# Rendering Lidar:
# -------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------



def sample_img_at_lidar(inds, img):
    """
    Creates a synthetic lidar image from camera image
    by sampling the camera at the corresponding pixel
    locations.
    
    Returns: Synthetic camera lidar
    In: pixel indicies [n, 2], Image to be sampled
    """
    img = np.array(img)
    cam_lidar = np.zeros(inds.shape[0], dtype=np.uint8)
    inds = np.floor(inds).astype(int)
#     print(img.shape)
    
    for i in range(inds.shape[0]):
        tx = inds[i][1]
        ty = inds[i][0]
        cam_lidar[i] = img[ty, tx]
    
    return cam_lidar

def pts_to_img(inds, pix, img_shape, dilate=False):
    """
    Renders a black and white image from points.
    
    Returns: Image
    In: Pixel locations:   [n, 2]
        Pixel intensities: [n, ]
        Image Shape:       [height, width]
        Dilate the pixels: [bool]
    """
    pix = pix.astype(np.uint8)
    img_arr = np.zeros([img_shape[0], img_shape[1]], dtype=np.uint8)
    inds= np.floor(inds).astype(int)
    
    for i in range(inds.shape[0]):
        tx = inds[i][1]
        ty = inds[i][0]
        img_arr[ty,tx] = pix[i]
    
    return make_bw_img(img_arr, dilate=dilate)
# Rendering Lidar:
# -------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------


def sample_img_at_lidar(inds, img):
    """
    Creates a synthetic lidar image from camera image
    by sampling the camera at the corresponding pixel
    locations.
    
    Returns: Synthetic camera lidar
    In: pixel indicies [n, 2], Image to be sampled
    """
    img = np.array(img)
    cam_lidar = np.zeros(inds.shape[0], dtype=np.uint8)
    inds = np.floor(inds).astype(int)
#     print(img.shape)
    
    for i in range(inds.shape[0]):
        tx = inds[i][1]
        ty = inds[i][0]
        cam_lidar[i] = img[ty, tx]
    
    return cam_lidar



def  proj_lidar_to_image0(pts_velo, intens, proj_mat, img_height, img_width):
    """
    Returns pixel locations in the image, depths of lidar pts in those locations and the lidar intensities
    """
    # apply projection
    pts_2d = project_to_2Dimage(pts_velo, proj_mat)
#     print(pts_2d)
    # Filter lidar points to be within image FOV
    inds = np.where((pts_2d[:, 0] < img_height) & (pts_2d[:, 0] >= 0) &
                    (pts_2d[:, 1] < img_width) & (pts_2d[:, 1] >= 0) &
                    (pts_velo[:, 0] > 0)
                    )[0]

    # Filter out pixels points
    pixel_locations = pts_2d[inds,:]
    

    # Retrieve depth from lidar
    imgfov_pc_velo = pts_velo[inds, :]
    intensities = intens[inds]
#     plt.scatter(imgfov_pc_velo[:,0],imgfov_pc_velo[:,1],c=imgfov_pc_velo[:,2])
    
    imgfov_pc_velo = np.hstack((imgfov_pc_velo, 
                                np.ones((imgfov_pc_velo.shape[0], 1))))
    
    depths = (proj_mat @ imgfov_pc_velo.transpose()).T[:,2]
    
    return pixel_locations, depths, intensities




def render_lidar_on_image0(pts_velo,intens, img, proj_velo2cam0, img_height, img_width, plt_depth=False, plt_pts_on_img = False):
    
    pixel_locations, depths, intensities = proj_lidar_to_image0(pts_velo, intens, proj_velo2cam0, img_height, img_width)
    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
    img_l = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    if plt_pts_on_img:
        for i in range(pixel_locations.shape[0]):
            if plt_depth:
                colour = cmap[int(640.0 / depths[i]), :]
            else:
                colour = cmap[int(255*intensities[i]), :]

            cv2.circle(img_l, (int(np.round(pixel_locations[i, 0])),
                             int(np.round(pixel_locations[i, 1]))),
                       1, color=tuple(colour), thickness=1)
#     plt.imshow(img)
#     plt.axis('off')
#     plt.show()
    if plt_depth:
        ints = depths
    else:
        ints = intensities
        
        
    pixel_locations = np.floor(np.vstack((pixel_locations[:, 1], pixel_locations[:, 0])).T).astype(int)
    return img_l, pixel_locations, ints