import vtk
import numpy as np
import matplotlib.pyplot as plt
import nrrd
import cv2
import itertools
import matplotlib.pyplot as plt
import pickle
import os

# NAME = 'Normal_1'
# data, header = nrrd.read(f'C:/Users/최재원/Desktop/ASOCADataAccess/ASOCADataAccess/Normal/Annotations/{NAME}.nrrd')
# NAME = 'Diseased_6'
# data, header = data, header = nrrd.read(f'C:/Users/Public/Pycharm/preprocessing/ASOCAData/Diseased/Annotations/{NAME}.nrrd')
# data = np.moveaxis(data, -1, 0)
#
# CTCA_data, CTCA_header = nrrd.read(f'C:/Users/최재원/Desktop/ASOCADataAccess/ASOCADataAccess/Diseased/CTCA/{NAME}.nrrd')
# CTCA_data = np.moveaxis(CTCA_data, -1, 0)
#
# vtp_file_path = "C:/Users/최재원/Desktop/ASOCADataAccess/ASOCADataAccess/Diseased/Centerlines/Diseased_1.vtp"


# coordinate_list_1 = []
# for z, img_data in enumerate(data):
#     for y, row in enumerate(img_data):
#         for x, value in enumerate(row):
#             if value == 1:
#                 coordinate_list_1.append((x, y, z))

# vtp_reader = vtk.vtkXMLPolyDataReader()
# vtp_reader.SetFileName(vtp_file_path)
# vtp_reader.Update()
# vtp_poly_data = vtp_reader.GetOutput()
#
# points = vtp_poly_data.GetPoints()
#
# num_points = points.GetNumberOfPoints()
# points_array = np.zeros((num_points, 3))
# for i in range(num_points):
#     points_array[i] = points.GetPoint(i)


# x_1 = np.array([coord[1] for coord in coordinate_list_1])
# y_1 = np.array([coord[0] for coord in coordinate_list_1])
# z_1 = np.array([coord[2] for coord in coordinate_list_1])
# print((np.max(x_1)+np.min(x_1))/2, (np.max(y_1)+np.min(y_1))/2, (np.max(z_1)+np.min(z_1))/2)
# print(np.max(x_1)-np.min(x_1), np.max(y_1)-np.min(y_1), np.max(z_1)-np.min(z_1))
# print(np.max(x_1), np.max(y_1), np.max(z_1))
# print(np.min(x_1), np.min(y_1), np.min(z_1))

#
# x_2 = np.asarray(points_array[:, 0] * 2.586206897 -230.4137932)
# # x_2 = points_array[:, 0]
# # x_2 = np.asarray(points_array[:, 0] * 2.6 - 241.51875, dtype=int)
#
# y_2 = np.asarray(points_array[:, 1] * 2.589473684 -426.0947368)
# # y_2 = points_array[:, 1]
# # y_2 = np.asarray(points_array[:, 1] * 2.6 - 436.0202047, dtype=int)
#
# z_2 = np.asarray(points_array[:, 2]  * 1.590909091 + 295.1363636)
# # z_2 = points_array[:, 2]
# # z_2 = np.asarray(points_array[:, 2] * 1.6 + 301.83643, dtype=int)

#
# print((np.max(x_2)+np.min(x_2))/2, (np.max(y_2)+np.min(y_2))/2, (np.max(z_2)+np.min(z_2))/2)
# print(np.max(x_2)-np.min(x_2), np.max(y_2)-np.min(y_2), np.max(z_2)-np.min(z_2))
# print(np.max(x_2), np.max(y_2), np.max(z_2))
# print(np.min(x_2), np.min(y_2), np.min(z_2))
#
# zz = zip(x_2, y_2, z_2)
# for coord1, coord2 in itertools.combinations(zz, 2):
#     set1 = set(coord1)
#     set2 = set(coord2)
#     common_coords = set1.intersection(set2)
#     zz = [coord for coord in zz if coord not in common_coords]



# coordinates = list(zip(x_2, y_2, z_2))
# page = np.zeros((512, 512, 224))
# for coord in coordinates:
#     x, y, z = coord
#     page[x, y, z] = 255
# page = np.moveaxis(page, -1, 0)

# data = (data*255)
# for k in range(data.shape[0]):
#     zeros = np.zeros((512,512))
#     # centerline = (page[k, :, :])
#     ctdata = (CTCA_data[k, :, :])
#     # ctdata[ctdata < -2500] = 0
#     # m = np.min(ctdata)
#     # M = np.max(ctdata)
#     # nor = (ctdata- m) / (M-m)
#     # nor *= 255
#     # annodata = data[k, :, :]
#     # ct255 = np.where(annodata == 255, 255, ctdata)
#     # ct0 = np.where(annodata == 255, 0, ct255) #ctdata에서 annotation 지운거
#     #
#     # center255 = np.where(centerline == 255, 255, ct0) #centerlines
#     # center0 = np.where(centerline == 255, 0, ctdata) #centerlines에서 annotation 지운거
#     #
#     # image = np.stack((center255, ct0, center0), axis=0)
#     # image = np.moveaxis(image, 0, -1)
#     cv2.imwrite(f"./rep/{k+1}.png", ctdata)


colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#6f290a', '#87c392', '#48ff71', '#ffff45', '#ff45ff', '#45ffff', '#180483']
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

NAME = 'Normal_4'
data, header = nrrd.read(f'C:/Users/Public/Pycharm/preprocessing/ASOCAData/{NAME.split("_")[0]}/Annotations/{NAME}.nrrd')
coordinate_list_1 = []
for z, img_data in enumerate(data):
    for y, row in enumerate(img_data):
        for x, value in enumerate(row):
            if value == 1:
                coordinate_list_1.append((x, y, z))
x_1 = np.array([coord[1] for coord in coordinate_list_1])
y_1 = np.array([coord[0] for coord in coordinate_list_1])
z_1 = np.array([coord[2] for coord in coordinate_list_1])
ax.scatter(z_1,x_1, y_1, c='y', alpha=0.1)
with open(f"C:/Users/최재원/Desktop/ASOCADataAccess/vessel/centerlines_main_3/centerlines_main_3_{NAME}.pkl", "rb") as f:
    groups = pickle.load(f)

# all_coords = [coord for sublist in groups for coord in sublist]
#
# # x, y, z 값들을 추출하여 최댓값 찾기
# x_max = max(coord[0] for coord in all_coords)
# x_min = min(coord[0] for coord in all_coords)
# y_max = max(coord[1] for coord in all_coords)
# y_min = min(coord[1] for coord in all_coords)
# z_max = max(coord[2] for coord in all_coords)
# z_min = min(coord[2] for coord in all_coords)
# ccx = (x_max + x_min)/2
# ccy = (y_max + y_min)/2
# ccz = (z_max + z_min)/2
#
# cccx = ccx - cx
# cccy = ccy - cy
# cccz = ccz - cz
listup = []
new_groups = []
for i in range(len(groups)):
    g1 = np.array(groups[i])
    a = 0
    b = 0
    c = 0
    g1 = g1 - (-12, 0, 5)
    # ax.scatter(g1[:,0] - cccx +a, g1[:,1] - cccy +b, g1[:,2] - cccz +c, c=colors[i], s=1, label=i)
    ax.scatter(g1[:,0], g1[:,1], g1[:,2], c='r', s=1, label=i)
    listup.append(g1 - (-12, 0, 5))
    # ax.scatter(g1[0,0], g1[0,1], g1[0,2], c='k', s=10)
    # listup.append(g1)


# ax.scatter(x_2, y_2, z_2, c='red', s=1)
ax.set_ylabel('Y')
ax.set_xlabel('X')
ax.set_zlabel('D')
plt.show()
with open(f"C:/Users/최재원/Desktop/ASOCADataAccess/vessel/centerlines_main_3/centerlines_main_3_{NAME}.pkl", "wb") as f:
    pickle.dump(listup, f, protocol=pickle.HIGHEST_PROTOCOL)
print("saved")
# with open(f"./centerlines_float.pkl", "wb") as f:
#     pickle.dump(new_groups, f, protocol=pickle.HIGHEST_PROTOCOL)
#     print("Saved")