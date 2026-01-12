import nrrd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import cv2
from scipy.ndimage import map_coordinates
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

    plane_image = map_coordinates(image_volume, plane_points.transpose(2, 0, 1), order=1, mode='constant')
    return plane_image
prev_n = None
# state = 'Diseased'
# num = '6'
# data_name = state + '_' + num

# ct_data, ct_header = nrrd.read(f'./ASOCAData/{state}/CTCA/{data_name}.nrrd')
# annotation_data, annotation_header = nrrd.read(f'./ASOCAData/{state}/Annotations/{data_name}.nrrd')
#
# with open(f'./CntF/centerlines_float_{data_name}.pkl', 'rb') as f:
#     centerline_data = pickle.load(f)
#
# output_dir = './HL_slice'
# os.makedirs(output_dir, exist_ok=True)
#
# num_slices = ct_data.shape[2]
# for h in range(num_slices):
#     ct_slice = ct_data[:, :, h]
#     ct_slice = np.clip(ct_slice, -100, 400)
#     min_val = np.min(ct_slice)
#     max_val = np.max(ct_slice)
#     ct_slice = (ct_slice - min_val) / (max_val - min_val)
#
#     annotation_slice = annotation_data[:, :, h]
#
#     colored_slice = np.stack([ct_slice] * 3, axis=-1)  # 3채널
#
#     # colored_slice[annotation_slice > 0] = [1, 0, 0]  # 빨간색
#     #
#     # for vessel in centerline_data:
#     #     for (x, y, z) in vessel:
#     #         if int(z) == h:
#     #             colored_slice[int(x), int(y)] = [0, 0, 1]  # 파란색
#
#     plt.imsave(os.path.join(output_dir, f'slice_{h:04d}.png'), colored_slice)
with open(f'./score.txt', 'r') as f:
    lines = f.readlines()

for line in lines:
    line = line.rstrip().split(':')
    print(line[0])
    if line[0][0] == 'N':
        state = 'Normal'
    else:
        state = 'Diseased'
    data_name = f'{state}_{line[0].split()[0][2:]}'

    ct_data, ct_header = nrrd.read(f'./ASOCAData/{state}/CTCA/{data_name}.nrrd')
    annotation_data, annotation_header = nrrd.read(f'./ASOCAData/{state}/Annotations/{data_name}.nrrd')

    with open(f'./CntF/centerlines_float_{data_name}.pkl', 'rb') as f:
        centerline_data = pickle.load(f)
    cnld = np.concatenate(centerline_data, axis=0)
    output_dir = f'./HL_slice/{line[0].rstrip()}'
    os.makedirs(output_dir, exist_ok=True)

    num_slices = line[1].split(', ')
    for h in num_slices:
        h = int(h)

        # ct_data = np.clip(ct_data, 0, 1000)
        t0 = np.array(cnld[h])
        t1 = np.array(cnld[h + 1])

        n_vec = t1 - t0
        n_vec = n_vec / np.linalg.norm(n_vec)

        extracted_image = extract_plane(ct_data, t0, t0 + n_vec, size=8, resolution=64)
        ex_anno = extract_plane(annotation_data, t0, t0 + n_vec, size=8, resolution=64)
        extracted_image = extracted_image - extracted_image.min()
        extracted_image = (extracted_image / extracted_image.max())*255
        # extracted_image = np.where(extracted_image > 250, extracted_image*1.05, extracted_image)
        # extracted_image = (extracted_image / extracted_image.max())*255

        # extracted_image = np.where(ex_anno==0, np.power(extracted_image, 1.3), extracted_image)
        # extracted_image = np.where(ex_anno==1, np.power(extracted_image, 1.315), extracted_image)
        # extracted_image = np.where(extracted_image > 250, extracted_image*1.1, extracted_image)
        extracted_image = np.power(extracted_image, 6)
        plt.imsave(os.path.join(output_dir, f'slice_{h}.png'), extracted_image, cmap='gray')
    print("done")