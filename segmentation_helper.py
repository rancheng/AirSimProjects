import airsim
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

client = airsim.VehicleClient()
client.confirmConnection()

responses = client.simGetImages([
    airsim.ImageRequest("0", airsim.ImageType.Segmentation, pixels_as_float=False, compress=False), # segmentation image in int
    airsim.ImageRequest("1", airsim.ImageType.DepthPerspective, pixels_as_float=True, compress=False), # depth in perspective projection
    airsim.ImageRequest("2", airsim.ImageType.Scene, pixels_as_float=False, compress=False) # scene vision image in uncompressed RGBA array
    ])
np.shape(responses[0])
r0 = responses[0]
r1 = responses[1]
r2 = responses[2]
(x_m, y_m) = np.meshgrid(range(0, r1.width), range(0, r1.height))
if len(r0.image_data_float) > 1:
    img0 = np.array(r0.image_data_float)
    img0 = img0.reshape(r0.height, r0.width)
else:
    img0 = np.frombuffer(r0.image_data_uint8, dtype=np.uint8) #get numpy array
    img0 = img0.reshape(r0.height, r0.width, 4)
if len(r1.image_data_float) > 1:
    img1 = np.array(r1.image_data_float)
    img1 = img1.reshape(r1.height, r1.width)
    # img1 = img1 * 200
    # img1[img1 > 255] = 255
else:
    img1 = np.frombuffer(r1.image_data_uint8, dtype=np.uint8) #get numpy array
    img1 = img1.reshape(r1.height, r1.width, 4)
if len(r2.image_data_float) > 1:
    img2 = np.array(r2.image_data_float)
    img2 = img2.reshape(r2.height, r2.width)
else:
    img2 = np.frombuffer(r2.image_data_uint8, dtype=np.uint8) #get numpy array
    img2 = img2.reshape(r2.height, r2.width, 4)
figure = plt.figure()
cv2.imwrite('segl.png', img0)
cv2.imwrite('depth.png', img1)
cv2.imwrite('rgb.png', img2)
figure
plt.subplot(311)
plt.imshow(img0)
plt.subplot(312)
plt.imshow(img1, cmap=plt.cm.gray)
plt.subplot(313)
plt.imshow(img2)
plt.show()


# render the 3d point cloud
# if len(img1.shape) == 2:
#     figure = plt.figure()
#     ax = figure.gca(projection='3d')
#     cam_info = client.simGetCameraInfo('1')
#     proj_mat = np.array(cam_info.proj_mat.matrix)
#     x_m_flat = np.squeeze(x_m.reshape((-1, 1)))
#     y_m_flat = np.squeeze(y_m.reshape((-1, 1)))
#     z_m_flat = np.squeeze(img1.reshape((-1, 1)))
#     dummy_1 = np.ones(len(x_m_flat))
#     xyz_zip = zip(x_m_flat, y_m_flat, z_m_flat, dummy_1)
#     x_res = []
#     y_res = []
#     z_res = []
#     for tmp_point in xyz_zip:
#         tmp_point = np.array(tmp_point).reshape((-1, 1))
#         tmp_proj = np.squeeze(np.matmul(proj_mat, tmp_point))
#         x_res.append(tmp_proj[3])
#         y_res.append(tmp_proj[0])
#         z_res.append(tmp_proj[2])
#     x_m = x_m * 2 * img1 / r1.width
#     y_m = y_m * 2 * img1 / r1.width
#     img1[img1 > 0.4] = 0
#     # ax.scatter3D(x_res, y_res, z_res)
#     plt.scatter(x_res, y_res, z_res)
#     plt.xlabel('x')
#     plt.ylabel('y')
#     # ax.plot_surface(x_m, y_m, img1, rstride=1, cstride=1, cmap=plt.cm.gray, linewidth=0)
#     plt.show()
