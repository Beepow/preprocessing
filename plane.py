import numpy as np
from scipy.ndimage import map_coordinates
import nrrd
import pickle
import os
import cv2
import matplotlib.pyplot as plt
import sklearn.preprocessing as PrePro

def clamp_rotation(prev_n, current_n, max_angle=np.pi / 6):
    angle = np.arccos(np.clip(np.dot(prev_n, current_n), -1.0, 1.0))
    if angle > max_angle:
        rotation_axis = np.cross(prev_n, current_n)
        rotation_axis /= np.linalg.norm(rotation_axis)
        rotation_matrix = np.eye(3) + np.sin(max_angle) * np.cross(rotation_axis, np.eye(3)) + \
                          (1 - np.cos(max_angle)) * np.outer(rotation_axis, rotation_axis)
        current_n = np.dot(rotation_matrix, prev_n)
    return current_n / np.linalg.norm(current_n)


def extract_plane(image_volume, t0, t1, size=12, resolution=16):
    n = np.array(t1) - np.array(t0)
    n = n / np.linalg.norm(n)

    u = np.cross(n, np.array([1, 0, 0]))
    if np.linalg.norm(u) == 0:
        u = np.cross(n, np.array([0, 1, 0]))
    u = u / np.linalg.norm(u)

    v = np.cross(n, u)
    v = v / np.linalg.norm(v)

    grid_x, grid_y = np.meshgrid(np.linspace(-size, size, resolution), np.linspace(-size, size, resolution))
    gx = grid_x[:,:,np.newaxis] * u
    gy = grid_y[:,:,np.newaxis] * v
    plane_points = t0 + gx + gy

    plane_image = map_coordinates(image_volume, plane_points.transpose(2, 0, 1), order=1, mode='nearest')
    return plane_image

prev_n = None

Train = 0
CT = 1
data_type = 'CTCA' if CT else 'Annotation'
slice_size = 64
overlap = 16
start = 0

if Train:
    state = 'Normal'
    stack = []
    for number in range(1, 21):
        NAME = f'{state}_{number}'
        # data, header = nrrd.read(f'C:/Users/Public/Pycharm/preprocessing/ASOCAData/{state}/Annotations/{NAME}.nrrd')
        data, CTCA_header = nrrd.read(f'C:/Users/Public/Pycharm/preprocessing/ASOCAData/{state}/{data_type}/{NAME}.nrrd')

        with open(f"C:/Users/최재원/Desktop/ASOCADataAccess/vessel/centerlines_main_3/centerlines_main_3_{NAME}.pkl", "rb") as f:
            groups = pickle.load(f)
        # stack_array = [] # 각 사람별로 저장할때 사용

        normalized_image = np.clip(data, -300, 400)

        t = []
        for i in range(len(groups)):
            if i==0:
                if number in (4,5,6,7,10,11,18):
                    pass
                else:
                    stack_array = []
                    points = list(groups[i])
                    np_points = np.array(points)
                    n = 1
                    for kk in range(0, len(np_points) - n):
                        t0 = np.array(np_points[kk])
                        t1 = np.array(np_points[kk+n])

                        n_vec = t1 - t0
                        n_vec = n_vec / np.linalg.norm(n_vec)

                        # 이전 법선 벡터와 비교하여 회전 각도 제한
                        if prev_n is not None:
                            n_vec = clamp_rotation(prev_n, n_vec)
                        prev_n = n_vec
                        extracted_image = extract_plane(normalized_image, t0, t0 + n_vec)
                        stack_array.append(extracted_image)
                    t.append(stack_array)
            elif i==1:
                if number in (8,9,12, 14, 15,17,19):
                    pass
                else:
                    stack_array = []
                    points = list(groups[i])
                    np_points = np.array(points)
                    n = 1
                    for kk in range(0, len(np_points) - n):
                        t0 = np.array(np_points[kk])
                        t1 = np.array(np_points[kk+n])

                        n_vec = t1 - t0
                        n_vec = n_vec / np.linalg.norm(n_vec)

                        # 이전 법선 벡터와 비교하여 회전 각도 제한
                        if prev_n is not None:
                            n_vec = clamp_rotation(prev_n, n_vec)
                        prev_n = n_vec
                        extracted_image = extract_plane(normalized_image, t0, t0 + n_vec)
                        stack_array.append(extracted_image)
                    t.append(stack_array)

            elif i==2:
                if number in (2, 6,8):
                    pass
                else:
                    stack_array = []
                    points = list(groups[i])
                    np_points = np.array(points)
                    n = 1
                    for kk in range(0, len(np_points) - n):
                        t0 = np.array(np_points[kk])
                        t1 = np.array(np_points[kk+n])

                        n_vec = t1 - t0
                        n_vec = n_vec / np.linalg.norm(n_vec)

                        # 이전 법선 벡터와 비교하여 회전 각도 제한
                        if prev_n is not None:
                            n_vec = clamp_rotation(prev_n, n_vec)
                        prev_n = n_vec
                        extracted_image = extract_plane(normalized_image, t0, t0 + n_vec)
                        stack_array.append(extracted_image)
                    t.append(stack_array)


        # for i in range(len(groups)):
        #     fig, axes = plt.subplots(1, 16, figsize=(8, 10))
        #     ss = np.array(t[i])
        #     for j in range(16):
        #         mid_slice = ss[:, :, j]
        #         axes[j].imshow(mid_slice, cmap='gray', aspect='equal')
        #         axes[j].set_title(f'Slice {j}')
        #     plt.savefig(f"C:/Users/Public/Pycharm/preprocessing/plane/Test/{NAME}_{i}.png")
        #     plt.show()

        stack_l = []
        while start + slice_size <= len(stack_array):
            stack_l.append(stack_array[start:start + slice_size])
            start += (slice_size - overlap)
        st_l = np.array(stack_l)
        stack.append(st_l)


    stack = np.concatenate(stack, axis=0)
    print(stack.shape)
    if not os.path.exists(f"C:/Users/최재원/Desktop/ASOCADataAccess/vessel/patch/Train_ves/"):
        os.makedirs(f"C:/Users/최재원/Desktop/ASOCADataAccess/vessel/patch/Train_ves/")
    with open(f"C:/Users/최재원/Desktop/ASOCADataAccess/vessel/patch/{state[0]}_patch.pkl", "wb") as f:
        pickle.dump(stack, f, protocol=pickle.HIGHEST_PROTOCOL)

else:
    for state in ['Diseased', 'Normal']:
        for number in range(1, 21):
            NAME = f'{state}_{number}'
            data, CTCA_header = nrrd.read(f'C:/Users/Public/Pycharm/preprocessing/ASOCAData/{state}/{data_type}/{NAME}.nrrd')

            with open(f"C:/Users/최재원/Desktop/ASOCADataAccess/vessel/centerlines_main_3/centerlines_main_3_{NAME}.pkl",
                      "rb") as f:
                groups = pickle.load(f)

            stack = []
            # stack_array = []  # 각 사람별로 저장할때 사용

            normalized_image = np.clip(data, -300, 400)

            t = []
            for i in range(len(groups)):
                if state == 'Normal':
                    if i == 0:
                        if number in (4, 5, 6, 7, 10, 11, 18):
                            pass
                        else:
                            stack_array = []
                            points = list(groups[i])
                            np_points = np.array(points)
                            n = 1
                            for kk in range(0, len(np_points) - n):
                                t0 = np.array(np_points[kk])
                                t1 = np.array(np_points[kk + n])

                                n_vec = t1 - t0
                                n_vec = n_vec / np.linalg.norm(n_vec)

                                if prev_n is not None:
                                    n_vec = clamp_rotation(prev_n, n_vec)
                                prev_n = n_vec
                                extracted_image = extract_plane(normalized_image, t0, t0 + n_vec)
                                stack_array.append(extracted_image)
                            t.append(stack_array)
                    elif i == 1:
                        if number in (8, 9, 12, 14, 15, 17, 19):
                            pass
                        else:
                            stack_array = []
                            points = list(groups[i])
                            np_points = np.array(points)
                            n = 1
                            for kk in range(0, len(np_points) - n):
                                t0 = np.array(np_points[kk])
                                t1 = np.array(np_points[kk + n])

                                n_vec = t1 - t0
                                n_vec = n_vec / np.linalg.norm(n_vec)

                                if prev_n is not None:
                                    n_vec = clamp_rotation(prev_n, n_vec)
                                prev_n = n_vec
                                extracted_image = extract_plane(normalized_image, t0, t0 + n_vec)
                                stack_array.append(extracted_image)
                            t.append(stack_array)

                    elif i == 2:
                        if number in (2, 6, 8):
                            pass
                        else:
                            stack_array = []
                            points = list(groups[i])
                            np_points = np.array(points)
                            n = 1
                            for kk in range(0, len(np_points) - n):
                                t0 = np.array(np_points[kk])
                                t1 = np.array(np_points[kk + n])

                                n_vec = t1 - t0
                                n_vec = n_vec / np.linalg.norm(n_vec)

                                if prev_n is not None:
                                    n_vec = clamp_rotation(prev_n, n_vec)
                                prev_n = n_vec
                                extracted_image = extract_plane(normalized_image, t0, t0 + n_vec)
                                stack_array.append(extracted_image)
                            t.append(stack_array)

                else:
                    if i == 0:
                        if number in (2, 3, 13, 14, 15, 16, 19):
                            pass
                        else:
                            stack_array = []
                            points = list(groups[i])
                            np_points = np.array(points)
                            n = 1
                            for kk in range(0, len(np_points) - n):
                                t0 = np.array(np_points[kk])
                                t1 = np.array(np_points[kk + n])

                                n_vec = t1 - t0
                                n_vec = n_vec / np.linalg.norm(n_vec)

                                if prev_n is not None:
                                    n_vec = clamp_rotation(prev_n, n_vec)
                                prev_n = n_vec
                                extracted_image = extract_plane(normalized_image, t0, t0 + n_vec)
                                stack_array.append(extracted_image)
                            t.append(stack_array)
                    elif i == 1:
                        if number in (10, 11, 17, 18):
                            pass
                        else:
                            stack_array = []
                            points = list(groups[i])
                            np_points = np.array(points)
                            n = 1
                            for kk in range(0, len(np_points) - n):
                                t0 = np.array(np_points[kk])
                                t1 = np.array(np_points[kk + n])

                                n_vec = t1 - t0
                                n_vec = n_vec / np.linalg.norm(n_vec)

                                if prev_n is not None:
                                    n_vec = clamp_rotation(prev_n, n_vec)
                                prev_n = n_vec
                                extracted_image = extract_plane(normalized_image, t0, t0 + n_vec)
                                stack_array.append(extracted_image)
                            t.append(stack_array)
                    elif i== 2:
                        if number in (111,222):
                            pass
                        else:
                            stack_array = []
                            points = list(groups[i])
                            np_points = np.array(points)
                            n = 1
                            for kk in range(0, len(np_points) - n):
                                t0 = np.array(np_points[kk])
                                t1 = np.array(np_points[kk + n])

                                n_vec = t1 - t0
                                n_vec = n_vec / np.linalg.norm(n_vec)

                                if prev_n is not None:
                                    n_vec = clamp_rotation(prev_n, n_vec)
                                prev_n = n_vec
                                extracted_image = extract_plane(normalized_image, t0, t0 + n_vec)
                                stack_array.append(extracted_image)
                            t.append(stack_array)

            # for i in range(len(groups)):
            #     fig, axes = plt.subplots(1, 16, figsize=(8, 10))
            #     ss = np.array(t[i])
            #     for j in range(16):
            #         mid_slice = ss[:, :, j]
            #         axes[j].imshow(mid_slice, cmap='gray', aspect='equal')
            #         axes[j].set_title(f'Slice {j}')
            #     plt.savefig(f"C:/Users/Public/Pycharm/preprocessing/plane/Test/{NAME}_{i}.png")
            #     plt.show()

            stack_l = []
            while start + slice_size <= len(stack_array):
                stack_l.append(stack_array[start:start + slice_size])
                start += (slice_size - overlap)

            st_l = np.array(stack_l)
            stack.append(st_l)

            stack = np.concatenate(stack, axis=0)
            print(NAME, stack.shape)
            if not os.path.exists(f"C:/Users/최재원/Desktop/ASOCADataAccess/vessel/patch/Train_ves/"):
                os.makedirs(f"C:/Users/최재원/Desktop/ASOCADataAccess/vessel/patch/Train_ves/")
            with open(f"C:/Users/최재원/Desktop/ASOCADataAccess/vessel/patch/{state[0]}_{number}.pkl", "wb") as f:
                pickle.dump(stack, f, protocol=pickle.HIGHEST_PROTOCOL)

