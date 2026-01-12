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


def extract_plane(image_volume, t0, t1, size=16, resolution=16):
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

########################OPTIONS#############

prev_n = None

Train = 1
CT = 1
data_type = 'CTCA' if CT else 'Annotations'
slice_size = 1
overlap = 0
size = 8
resolution = 8

display = 0
############################################

if Train:
    state = 'Normal'
    stack = []
    for number in (1,2,3,4,5,6,7,8,9,10,16,17,18,19,20):
        NAME = f'{state}_{number}'
        data, CTCA_header = nrrd.read(f'C:/Users/Public/Pycharm/preprocessing/ASOCAData/{state}/{data_type}/{NAME}.nrrd')
        anno, Anno_header = nrrd.read(
            f'C:/Users/Public/Pycharm/preprocessing/ASOCAData/{state}/Annotations/{NAME}.nrrd')
        with open(f"./CntF/centerlines_float_{NAME}.pkl", "rb") as f:
            groups = pickle.load(f)

        normalized_image = np.clip(data, -300, 400)
        for_display = []
        stack_array = []
        for group in groups:
            for_display_whole = []
            n = 1
            for kk in range(0, len(group) - n):
                t0 = np.array(group[kk])
                t1 = np.array(group[kk + n])

                n_vec = t1 - t0
                n_vec = n_vec / np.linalg.norm(n_vec)

                if prev_n is not None:
                    n_vec = clamp_rotation(prev_n, n_vec)
                prev_n = n_vec
                extracted_image = extract_plane(normalized_image, t0, t0 + n_vec,size=size, resolution=resolution)
                stack_array.append(extracted_image)
                for_display_whole.append(extracted_image)
            for_display.append(np.array(for_display_whole))

        if display:
            for i, branch in enumerate(for_display):
                fig, axes = plt.subplots(1, resolution*2, figsize=(resolution, 10))
                for j in range(resolution):
                    axes[j].set_yticks([])
                    mid_slice = branch[:, j, :]
                    axes[j].imshow(mid_slice, cmap='gray', aspect='equal')
                    axes[j].set_title(f'{j}')
                for J in range(resolution):
                    axes[resolution+J].set_yticks([])
                    mid_slice = branch[:, :, J]
                    axes[resolution+J].imshow(mid_slice, cmap='gray', aspect='equal')
                    axes[resolution+J].set_title(f'{J}')
                plt.yticks([], [])
                plt.savefig(f"C:/Users/Public/Pycharm/preprocessing/plane/Test/{NAME}_{i}.png")
                plt.show()

        start = 0
        stack_l = []
        while start + slice_size <= len(stack_array):
            stack_l.append(stack_array[start:start + slice_size])
            start += (slice_size - overlap)
        if stack_l:
            stack.append(np.array(stack_l))


    stack = np.concatenate(stack, axis=0)
    print(stack.shape)
    if not os.path.exists(f"C:/Users/최재원/Desktop/CTCA_Data_2501/Train/"):
        os.makedirs(f"C:/Users/최재원/Desktop/CTCA_Data_2501/Train/")
    with open(f"C:/Users/최재원/Desktop/CTCA_Data_2501/Train/{state[0]}_s_{size}_res_{resolution}_len_{slice_size}_withAnno.pkl", "wb") as f:
        pickle.dump(stack, f, protocol=pickle.HIGHEST_PROTOCOL)

else:
    for state in ['Diseased', 'Normal']:
        a=0
        for number in range(1, 21):
            NAME = f'{state}_{number}'
            data, CTCA_header = nrrd.read(
                f'C:/Users/Public/Pycharm/preprocessing/ASOCAData/{state}/{data_type}/{NAME}.nrrd')
            anno, Anno_header = nrrd.read(
                f'C:/Users/Public/Pycharm/preprocessing/ASOCAData/{state}/Annotations/{NAME}.nrrd')

            with open(f"./CntF/centerlines_float_{NAME}.pkl", "rb") as f:
                groups = pickle.load(f)
            # with open(f"./centerlines_{NAME}.pkl", "rb") as f:
            #     groups = pickle.load(f)
            stack = []
            data = np.clip(data, -100, 400)
            vessels = []
            for i in range(len(groups)):
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
                    extracted_image = extract_plane(data, t0, t0 + n_vec, size=size, resolution=resolution)
                    stack_array.append(extracted_image)
                vessels.append(stack_array)

            stack_l = []
            for v in range(len(vessels)):
                one_vessel = np.array(vessels[v])
                start = 0
                while start + slice_size <= len(one_vessel):
                    stack_l.append(one_vessel[start:start + slice_size])
                    start += (slice_size - overlap)
            st_l = np.array(stack_l)
            stack.append(st_l)

            stack = np.concatenate(stack, axis=0)

            # for i in range(len(groups)):
            #     fig, axes = plt.subplots(1, resolution, figsize=(8, 10))
            #     ss = np.array(vessels[i])
            #     for j in range(resolution):
            #         axes[j].set_yticks([])
            #         mid_slice = np.clip(ss[:, :, j], 0, 400)
            #         axes[j].imshow(mid_slice, cmap='gray', aspect='equal')
            #         axes[j].set_title(f'Slice {j}')
            #         plt.imsave(f'C:/Users/Public/Pycharm/preprocessing/plane/nobranch/{NAME}_{i}_{j}.png', mid_slice, cmap='gray')
            #     plt.savefig(f"C:/Users/Public/Pycharm/preprocessing/plane/Test/{NAME}_{i}.png")

                # plt.show()

            # for i in range(stack.shape[0]):
            #     slice_1 = (stack[i, :, :, int(resolution/2)-1])
            #     slice_2 = (stack[i, :, :, int(resolution/2)])
            #     slice_3 = stack[i, :, int(resolution/2)-1, :]
            #     slice_4 = stack[i, :, int(resolution/2), :]
            #     plt.imsave(f'./plane/fig_noclip/{NAME}_{i}_{int(resolution/2)-1}_L.png', slice_1, cmap='gray')
            #     plt.imsave(f'./plane/fig_noclip/{NAME}_{i}_{int(resolution/2)}_L.png', slice_2, cmap='gray')
            #     # plt.imsave(f'./plane/fig_noclip/{NAME}_{i}_{int(resolution/2)-1}_R.png', slice_1, cmap='gray')
            #     # plt.imsave(f'./plane/fig_noclip/{NAME}_{i}_{int(resolution/2)}_R.png', slice_2, cmap='gray')
            #     for j in range(stack.shape[1]):
            #         slice = (stack[i, j, :, :])
            #         plt.imsave(f'./plane/fig/{NAME}_{i}_{j}.png', slice, cmap='gray')

            print(NAME, stack.shape)
            if not os.path.exists(f"C:/Users/최재원/Desktop/CTCA_Data_2501/Test/s_{size}_res_{resolution}_len_{slice_size}/"):
                os.makedirs(f"C:/Users/최재원/Desktop/CTCA_Data_2501/Test/s_{size}_res_{resolution}_len_{slice_size}/")
            with open(
                    f"C:/Users/최재원/Desktop/CTCA_Data_2501/Test/s_{size}_res_{resolution}_len_{slice_size}/{state[0]}_{number}.pkl",
                    "wb") as f:
                pickle.dump(stack, f, protocol=pickle.HIGHEST_PROTOCOL)

            a += stack.shape[0]
        print(state, a)
