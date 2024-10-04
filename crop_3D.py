import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import nrrd
import pickle
import os
import cv2
import scipy.interpolate as interpolation

NAME = 'Diseased_1'  #
data, header = nrrd.read(f'C:/Users/Public/Pycharm/preprocessing/ASOCAData/Diseased/Annotations/{NAME}.nrrd')
# data = np.moveaxis(data, -1, 0)

CTCA_data, CTCA_header = nrrd.read(f'C:/Users/Public/Pycharm/preprocessing/ASOCAData/Diseased/CTCA/{NAME}.nrrd')
# CTCA_data = np.moveaxis(CTCA_data, -1, 0)

# image = sitk.GetImageFromArray(data)

with open(f"./centerlines_float_Diseased_1.pkl", "rb") as f:
    groups = pickle.load(f)


# groups = groups['groups']

def find_perpendicular_plane(point, vector):
    vector = vector / np.linalg.norm(vector)
    A, B, C = vector
    D = -np.dot(vector, point)
    return A, B, C, D


def generate_plane_coordinates(plane_equation, size=10, center=(0, 0, 0), type=None):
    A, B, C, D = plane_equation
    coord = []
    print("Center : ", center)

    rr1 = size * (1 - A ** 2)
    rr2 = size * (1 - B ** 2)
    rr3 = size * (1 - C ** 2)
    # if rr3 < 16:
    #     rr3 = size

    if A >= 0:
        if B >= 0:
            x = np.linspace(center[0] + rr1 / 2, center[0] - rr1 / 2, num=size)
            y = np.linspace(center[1] - rr2 / 2, center[1] + rr2 / 2, num=size)
            z = np.linspace(center[2] + rr3 / 2, center[2] - rr3 / 2, num=size)
        elif B < 0:
            x = np.linspace(center[0] - rr1 / 2, center[0] + rr1 / 2, num=size)
            y = np.linspace(center[1] - rr2 / 2, center[1] + rr2 / 2, num=size)
            z = np.linspace(center[2] + rr3 / 2, center[2] - rr3 / 2, num=size)
    if A < 0:
        if B > 0:
            x = np.linspace(center[0] + rr1 / 2, center[0] - rr1 / 2, num=size)
            y = np.linspace(center[1] + rr2 / 2, center[1] - rr2 / 2, num=size)
            z = np.linspace(center[2] + rr3 / 2, center[2] - rr3 / 2, num=size)
        elif B <= 0:
            x = np.linspace(center[0] - rr1 / 2, center[0] + rr1 / 2, num=size)
            y = np.linspace(center[1] + rr2 / 2, center[1] - rr2 / 2, num=size)
            z = np.linspace(center[2] + rr3 / 2, center[2] - rr3 / 2, num=size)

    for i in range(len(x)):
        # z = (-A * x[i] - B * y[i] - D) / C
        for j in range(len(y)):
            # x = (-B * y[j] - D) / A
            #     z = (-A * x[i] - B * y[j] - D) / C
            coord.append((x[i], y[i], z[j]))
    print("A != 0 and B != 0")
    return coord


size = 20


def near_point(points, minimum_point):
    len_list = []
    for i in range(len(points)):
        length = distance(points[i], minimum_point)
        len_list.append(length)
    near_idx = np.argsort(len_list)
    return near_idx


def distance(point1, point2):
    dis = sum((x - y) ** 2 for x, y in zip(point1, point2)) ** 0.5
    return dis


def far_point(points, stan):
    len_list = []
    for i in range(len(points)):
        length = distance(points[i], stan)
        len_list.append(length)
    farpoint = np.argmax(len_list)
    return farpoint


for i in range(len(groups)):
    stack_array = []
    points = list(groups[i])
    np_points = np.array(points)
    # center = (data.shape[0]/2, data.shape[1]/2, data.shape[2])
    # minimum_idx = far_point(points, center)
    # min_point = points[minimum_idx]
    # near_idx = near_point(points, min_point)
    # sorted_points = np.array(points)[near_idx]
    sorted_points = np_points[::-1, :]

    n = 1
    for kk in range(0, (len(sorted_points) - n)):
        t0 = np.array(sorted_points[kk])
        t1 = np.array(sorted_points[kk + n])
        vector = t1 - t0
        # if (vec1[1] == vec2[1] == vec3[1] == 0 and vector[1] != 0)\
        #         or (vec1[2] == vec2[2] == vec3[2] == 0 and vector[2] !=0)\
        #         or (vec1[0] == vec2[0] == vec3[0] == 0 and vector[0] != 0):
        #     print("**************************PASS**************************", kk)
        #     pass
        # else:
        # vector = generate_perpendicular_vector(vector)
        # plane_equation = find_perpendicular_plane(t0, vector)
        # if np.array_equal(vector, [0, 0, 1]) or np.array_equal(vector, [0, 0, -1]):
        #     vector = t1 - t0
        #     plane_equation = find_perpendicular_plane(t0, vector)
        # else:
        #     plane_equation = find_perpendicular_plane(t0, vector)
        plane_equation = find_perpendicular_plane(t0, vector)
        coord = generate_plane_coordinates(plane_equation, size=size, center=t0)

        # coord = np.round(coord).astype(int)

        interp_func = interpolation.RegularGridInterpolator((np.arange(data.shape[0]), np.arange(data.shape[1]),
                                                             np.arange(data.shape[2])), data, bounds_error=False,
                                                            fill_value=None)
        CTinterp = interpolation.RegularGridInterpolator((np.arange(CTCA_data.shape[0]), np.arange(CTCA_data.shape[1]),
                                                          np.arange(CTCA_data.shape[2])), CTCA_data, bounds_error=False,
                                                         fill_value=None)

        CTplane = np.zeros((size, size))
        nrplane = np.zeros((size, size))
        for p in range(size):
            for q in range(size):
                coordset = coord[p * size + q]
                x_idx = coordset[0]
                y_idx = coordset[1]
                z_idx = coordset[2]
                if z_idx >= data.shape[-1]:
                    z_idx = data.shape[-1] - 1
                elif z_idx <= 0:
                    z_idx = 0

                # if abs(plane_equation[2]) > 0.7:

                nrplane_data = interp_func([x_idx, y_idx, z_idx])
                # nrplane_data = data[x_idx, y_idx, z_idx]
                nrplane[q, p] = nrplane_data
                CTplane_data = CTinterp([x_idx, y_idx, z_idx])
                # CTplane_data = CTCA_data[x_idx, y_idx, z_idx]
                CTplane[p, q] = CTplane_data

        # nrplane = np.where(nrplane > 0, np.max(CTplane), nrplane)
        # nrplane = nrplane * 255
        #
        # binplane = CTplane - nrplane
        # plane = np.stack((nrplane, binplane, binplane), axis=0)
        # plane = np.moveaxis(plane, 0, -1)

        stack_array.append(CTplane)

        # cv2.imwrite(f"./plane/group_{i}_{kk}.png", CTplane, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        # print("saved : ", i, "---", kk)

    # stack = np.array(stack_array)
    # with open(f"./plane_{i}.pkl", "wb") as f:
    #     pickle.dump(stack, f, protocol=pickle.HIGHEST_PROTOCOL)
        stack = np.array(stack_array)
        stack = np.moveaxis(stack, 0, 2)

        for k in range(int(size/2) - 1, int(size/2) + 1):
            image = stack[k]
            if not os.path.exists(f'C:/Users/Public/Pycharm/preprocessing/rep/{NAME}/{i}'):
                os.makedirs(f'C:/Users/Public/Pycharm/preprocessing/rep/{NAME}/{i}')
            cv2.imwrite(f"C:/Users/Public/Pycharm/preprocessing/rep/{NAME}/{i}/group_{i}_{k}.png", image)