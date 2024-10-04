from stl import mesh
from mpl_toolkits import mplot3d
import pickle
from matplotlib import pyplot as plt
import vtk
import itertools
from mayavi import mlab
from stl import mesh
import numpy as np
import nrrd
from matplotlib import cm
def distance(point1, point2):
    dis = sum((x - y) ** 2 for x, y in zip(point1, point2)) ** 0.5
    return dis
def far_point(points, stan):
    len_list = []
    points = list(points)
    for i in range(len(points)):
        length = distance(points[i], stan)
        len_list.append(length)
    farpoint = np.argmin(len_list)
    farpoint = points[farpoint]
    return farpoint

def near_point(points, last_point):
    len_list = []
    for i in range(len(points)):
        length = distance(points[i], last_point)
        len_list.append(length)
    near_idx = np.argsort(len_list)
    return near_idx

def nearest_point(points, last_point):
    len_list = []
    for i in range(len(points)):
        length = distance(points[i], last_point)
        len_list.append(length)
    nearest = points[np.argmin(len_list)]
    return nearest

# NAME = 'Diseased_3'
# data, header = nrrd.read(f'C:/Users/Public/Pycharm/preprocessing/ASOCAData/Normal/Annotations/{NAME}.nrrd')
# data = np.moveaxis(data, -1, 0)
# stl_file_path = f'C:/Users/Public/Pycharm/preprocessing/ASOCAData/Normal/SurfaceMeshes/{NAME}.stl'
# vtp_file_path = f"C:/Users/Public/Pycharm/preprocessing/ASOCAData/Normal/Centerlines/{NAME}.vtp"
# your_mesh = mesh.Mesh.from_file(stl_file_path)
#
# coordinate_list_1 = []
# for z1, img_data in enumerate(data):
#     for y1, row in enumerate(img_data):
#         for x1, value in enumerate(row):
#             if value == 1:
#                 coordinate_list_1.append((x1, y1, z1))
# x_1 = np.array([coord[1] for coord in coordinate_list_1])
# y_1 = np.array([coord[0] for coord in coordinate_list_1])
# z_1 = np.array([coord[2] for coord in coordinate_list_1])
#
# x1_M = np.max(x_1)
# x1_m = np.min(x_1)
# x_len = x1_M - x1_m
# y1_M = np.max(y_1)
# y1_m = np.min(y_1)
# y_len = y1_M - y1_m
# z1_M = np.max(z_1)
# z1_m = np.min(z_1)
# z_len = z1_M - z1_m
# x1_av = np.average(x_1)
# y1_av = np.average(y_1)
# z1_av = np.average(z_1)
#
# # unique2 = np.unique(np.stack((your_mesh.x, your_mesh.y, your_mesh.z), -1), axis=0)
# sfx_M = np.max(your_mesh.x)
# sfx_m = np.min(your_mesh.x)
# sfx_len = sfx_M - sfx_m
# sfy_M = np.max(your_mesh.y)
# sfy_m = np.min(your_mesh.y)
# sfy_len = sfy_M - sfy_m
# sfz_M = np.max(your_mesh.z)
# sfz_m = np.min(your_mesh.z)
# sfz_len = sfz_M - sfz_m
# sfx_av = np.average(your_mesh.x)
# sfy_av = np.average(your_mesh.y)
# sfz_av = np.average(your_mesh.z)
#
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
#
# w1 = (x_len / sfx_len)
# w2 = (y_len / sfy_len)
# w3 = (z_len / sfz_len)
# b1 = (x1_av - sfx_av * w1)
# b2 = (y1_av - sfy_av * w2)
# b3 = (z1_av - sfz_av * w3)
#
# x_2 = np.asarray(points_array[:, 0] * w1 + b1)#, dtype=int  * 2.586206897 - 230.4137932
# y_2 = np.asarray(points_array[:, 1] * w2 + b2)# * 2.589473684 - 426.0947368, dtype=int
# z_2 = np.asarray(points_array[:, 2] * w3 + b3)# * 1.590909091 + 295.1363636, dtype=int
#
#
#
# groups = []
# threshold = 3
#
# # unique2 = np.unique(np.stack((x_2, y_2, z_2), -1), axis=0)
# # print(unique2.shape)
# #
# # zz = zip(unique2[:,0], unique2[:, 1], unique2[:, 2])
# zz = zip(x_2, y_2, z_2)
#
# for coord1, coord2 in itertools.combinations(zz, 2):
#     if distance(coord1, coord2) < threshold:
#         added_to_existing_group = False
#         for group in groups:
#             if coord1 in group:
#                 group.add(coord2)
#                 group = sorted(group, key=lambda p: distance(p, coord1))
#                 added_to_existing_group = True
#                 break
#             elif coord2 in group:
#                 group.add(coord1)
#                 group = sorted(group, key=lambda p: distance(p, coord2))
#                 added_to_existing_group = True
#                 break
#         if not added_to_existing_group:
#             groups.append({coord1, coord2})
#
# for i, group1 in enumerate(groups):
#     for j, group2 in enumerate(groups):
#         if i != j:
#             common_coords = group1.intersection(group2)
#             if common_coords:
#                 if len(group1) > len(group2):
#                     group1.difference_update(common_coords)
#                 else:
#                     group2.difference_update(common_coords)
#
#
# colors = ['g', 'r', 'b', 'y','g', 'r', 'b', 'y','g', 'r', 'b', 'y','g', 'r', 'b', 'y', 'g', 'r', 'b', 'y','g', 'r', 'b', 'y','g', 'r', 'b', 'y','g', 'r', 'b', 'y',
#           'g', 'r', 'b', 'y','g', 'r', 'b', 'y','g', 'r', 'b', 'y','g', 'r', 'b', 'y', 'g', 'r', 'b', 'y','g', 'r', 'b', 'y','g', 'r', 'b', 'y','g', 'r', 'b', 'y',
#           'g', 'r', 'b', 'y', 'g', 'r', 'b', 'y', 'g', 'r', 'b', 'y', 'g', 'r', 'b', 'y', 'g', 'r', 'b', 'y', 'g', 'r','b', 'y', 'g', 'r', 'b', 'y', 'g', 'r', 'b', 'y',
#           'g', 'r', 'b', 'y', 'g', 'r', 'b', 'y', 'g', 'r', 'b', 'y', 'g', 'r', 'b', 'y', 'g', 'r', 'b', 'y', 'g', 'r', 'b', 'y', 'g', 'r', 'b', 'y', 'g', 'r', 'b', 'y',
#           'g', 'r', 'b', 'y','g', 'r', 'b', 'y','g', 'r', 'b', 'y','g', 'r', 'b', 'y', 'g', 'r', 'b', 'y','g', 'r', 'b', 'y','g', 'r', 'b', 'y','g', 'r', 'b', 'y',
#           'g', 'r', 'b', 'y','g', 'r', 'b', 'y','g', 'r', 'b', 'y','g', 'r', 'b', 'y', 'g', 'r', 'b', 'y','g', 'r', 'b', 'y','g', 'r', 'b', 'y','g', 'r', 'b', 'y',
#           'g', 'r', 'b', 'y', 'g', 'r', 'b', 'y', 'g', 'r', 'b', 'y', 'g', 'r', 'b', 'y', 'g', 'r', 'b', 'y', 'g', 'r','b', 'y', 'g', 'r', 'b', 'y', 'g', 'r', 'b', 'y',
#           'g', 'r', 'b', 'y', 'g', 'r', 'b', 'y', 'g', 'r', 'b', 'y', 'g', 'r', 'b', 'y', 'g', 'r', 'b', 'y', 'g', 'r', 'b', 'y', 'g', 'r', 'b', 'y', 'g', 'r', 'b', 'y'
#           ]
#
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111, projection='3d')
#
# listup = []
# print(len(groups))
# for i in range(len(groups)):
#     if len(groups[i]) != 1:
#         xs, ys, zs = zip(*groups[i])
#         group_list = []
#         lastpoint = []
#         zm = np.argmin(zs)
#         last = list(groups[i])[zm]
#         lastpoint.append(last)
#
#         points = list(groups[i])
#         # nearidx = near_point(points, lastpoint[i])
#         # points = np.array(points)[nearidx]
#         # listup.append(points)
#
#         group_list.append(lastpoint[0])
#         # filtered = np.delete(points, np.where((points == lastpoint).all(axis=1))[0], axis=0)
#         filtered = [point for point in points if not np.array_equal(point, lastpoint[0])]
#         nearest = nearest_point(filtered, lastpoint[0])
#         group_list.append(nearest)
#         filtered = [point for point in filtered if not np.array_equal(point, nearest)]
#         while len(filtered) !=0:
#             nearest = nearest_point(filtered, nearest)
#             group_list.append(nearest)
#             filtered = [point for point in filtered if not np.array_equal(point, nearest)]
#         listup.append(group_list)
#
#         ax.scatter(xs, ys, zs, c=colors[i], s=1)
#         # ax.scatter(points[19, 0], points[19, 1], points[19, 2], c='k', s=10)
#         ll = np.array(lastpoint)
#         ax.scatter(ll[:,0], ll[:,1], ll[:,2], c='k', s=5)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#6f290a', '#87c392', '#48ff71', '#ffff45', '#ff45ff', '#45ffff', '#180483']
#            #blue      #orange  #green      #red        #purple     #brown      #pink       #gray   #yell
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
with open(f"./CntF/centerlines_float_Normal_1.pkl", "rb") as f:
    groups = pickle.load(f)

# listup = []
# for i in [7]:
#     xs, ys, zs = zip(*groups[i])
#     points = groups[i]
#     group_list = []
#     lastpoint = []
    # xM = np.argmax(xs)
    # xm = np.argmin(xs)
    # yM = np.argmax(ys)
    # ym = np.argmin(ys)
    # if i == 0:
    #     last = list(groups[i])[xm]
    # elif i == 7:
    #     last = list(groups[i])[xM]
    # elif i == 16:
    #     last = list(groups[i])[yM]
    # elif i ==5:
    #     last = list(groups[i])[xM]
    # group_list.append(last)
    # filtered = [point for point in points if not np.array_equal(point, last)]
    # nearest = nearest_point(filtered, last)
    # group_list.append(nearest)
    # filtered = [point for point in filtered if not np.array_equal(point, nearest)]
    # while len(filtered) != 0:
    #     nearest = nearest_point(filtered, nearest)
    #     group_list.append(nearest)
    #     filtered = [point for point in filtered if not np.array_equal(point, nearest)]
    # del groups[i]
    # groups.append(group_list)


for i in range(len(groups)):
    g1 = np.array(groups[i])
    ax.scatter(g1[:,0], g1[:,1], g1[:,2], c=colors[i], s=1, label=i)
    ax.scatter(g1[0,0], g1[0,1], g1[0,2], c='k', s=10)
    ax.legend()

ax.set_ylabel('Y')
ax.set_xlabel('X')
ax.set_zlabel('D')
# plt.legend()
plt.show()
#
# with open(f"./centerlines_float.pkl", "wb") as f:
#     pickle.dump(listup, f, protocol=pickle.HIGHEST_PROTOCOL)

# a = 1