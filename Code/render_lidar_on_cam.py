# # Rendering Lidar:
# # -------------------------------------------------------------------------------------------------------
# # -------------------------------------------------------------------------------------------------------

# import numpy as np
# import cv2
# from matplotlib import pyplot as plt



# def sample_img_at_lidar(inds, img):
#     """
#     Creates a synthetic lidar image from camera image
#     by sampling the camera at the corresponding pixel
#     locations.
    
#     Returns: Synthetic camera lidar
#     In: pixel indicies [n, 2], Image to be sampled
#     """
#     img = np.array(img)
#     cam_lidar = np.zeros(inds.shape[0], dtype=np.uint8)
#     inds = np.floor(inds).astype(int)
# #     print(img.shape)
    
#     for i in range(inds.shape[0]):
#         tx = inds[i][1]
#         ty = inds[i][0]
#         cam_lidar[i] = img[ty, tx]
    
#     return cam_lidar



# def  proj_lidar_to_image0(pts_velo, intens, proj_mat, img_height, img_width):
#     """
#     Returns pixel locations in the image, depths of lidar pts in those locations and the lidar intensities
#     """
#     from Lidar_Cam_Operations import project_to_2Dimage
#     # apply projection
#     pts_2d = project_to_2Dimage(pts_velo, proj_mat)
# #     print(pts_2d)
#     # Filter lidar points to be within image FOV
#     inds = np.where((pts_2d[:, 0] < img_height) & (pts_2d[:, 0] >= 0) &
#                     (pts_2d[:, 1] < img_width) & (pts_2d[:, 1] >= 0) &
#                     (pts_velo[:, 0] > 0)
#                     )[0]

#     # Filter out pixels points
#     pixel_locations = pts_2d[inds,:]
    

#     # Retrieve depth from lidar
#     imgfov_pc_velo = pts_velo[inds, :]
#     intensities = intens[inds]
# #     plt.scatter(imgfov_pc_velo[:,0],imgfov_pc_velo[:,1],c=imgfov_pc_velo[:,2])
    
#     imgfov_pc_velo = np.hstack((imgfov_pc_velo, 
#                                 np.ones((imgfov_pc_velo.shape[0], 1))))
    
#     depths = (proj_mat @ imgfov_pc_velo.transpose()).T[:,2]
    
#     return pixel_locations, depths, intensities




# def render_lidar_on_image0(pts_velo,intens, img, proj_velo2cam0, img_height, img_width, plt_depth=False):
    
#     pixel_locations, depths, intensities = proj_lidar_to_image0(pts_velo, intens, proj_velo2cam0, img_height, img_width)
#     cmap = plt.cm.get_cmap('hsv', 256)
#     cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
    
# #     for i in range(pixel_locations.shape[0]):
# #         if plt_depth:
# #             colour = cmap[int(640.0 / depths[i]), :]
# #         else:
# #             colour = cmap[int(255*intensities[i]), :]
            
# #         cv2.circle(img, (int(np.round(pixel_locations[i, 0])),
# #                          int(np.round(pixel_locations[i, 1]))),
# #                    1, color=tuple(colour), thickness=1)
# #     plt.imshow(img)
# #     plt.axis('off')
# #     plt.show()
#     return img, pixel_locations, intensities