import numpy as np
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
import nrrd
import pickle
import os


def create_centered_cpr(volume, centerline, range_size=5):
    """
    중심선을 기준으로 좌우 ±range_size 범위와 z축으로 ±range_size 범위의 데이터를 포함하는
    CPR 3D 볼륨을 생성합니다.

    :param volume: 3D numpy array (예: CT 스캔 데이터)
    :param centerline: (N, 3) 형태의 numpy array, 중심선 좌표들 (x, y, z)
    :param range_size: 중심선 좌우와 z축으로 확장할 범위 (기본값은 5)
    :return: Centered CPR 3D 볼륨 (3D numpy array)
    """
    num_points = len(centerline)
    cpr_volume = np.zeros((num_points, 2 * range_size + 1, 2 * range_size + 1))

    for i, (x, y, z) in enumerate(centerline):
        for y_offset in range(-range_size, range_size + 1):
            for z_offset in range(-range_size, range_size + 1):
                sampled_value = map_coordinates(volume, np.array([
                    [x],
                    [y + y_offset],
                    [z + z_offset]
                ]), order=1, mode='nearest')
                cpr_volume[i, y_offset + range_size, z_offset + range_size] = sampled_value

    return cpr_volume


def save_cpr_as_pickle(cpr_volume, output_filename):
    """
    CPR 3D 볼륨을 pickle 파일로 저장합니다.

    :param cpr_volume: CPR 3D 볼륨 (3D numpy array)
    :param output_filename: 저장할 파일 이름
    """
    with open(output_filename, 'wb') as f:
        pickle.dump(cpr_volume, f)
    print(f"CPR volume saved as {output_filename}")

range_size=12
for state in ['Diseased', 'Normal']:
    stack = []
    for number in range(1, 21):
        NAME = f'{state}_{number}'
        data, CTCA_header = nrrd.read(f'C:/Users/Public/Pycharm/preprocessing/ASOCAData/{state}/Annotations/{NAME}.nrrd')

        with open(f"C:/Users/최재원/Desktop/ASOCADataAccess/vessel/cnt_new/centerlines_main_{NAME}.pkl", "rb") as f:
            groups = pickle.load(f)

        volume=[]

        for idx, centerline in enumerate(groups):
            # Centered CPR 3D 볼륨 생성

            mask = (data > -400)
            normalized_image = np.full_like(data, -400, dtype=float)
            normalized_image[mask] = data[mask]
            centered_cpr_volume = create_centered_cpr(normalized_image, centerline, range_size=range_size)


            num_slices = centered_cpr_volume.shape[1]  # 슬라이스 수 확인
            fig, axes = plt.subplots(1, num_slices, figsize=(num_slices, 10))

            for i in range(num_slices):
                mid_slice = centered_cpr_volume[:, i, :]  # 각 슬라이스 가져오기
                axes[i].imshow(mid_slice, cmap='gray', aspect='equal')
                axes[i].set_title(f'Slice {i}')
            plt.show()


            # mid_slice = centered_cpr_volume[:, 2, :]
            # plt.figure(figsize=(6, 12))
            # plt.imshow(mid_slice, cmap='gray', aspect='equal')
            # plt.title(f'Centered CPR Mid-Slice for {NAME} Group {idx + 1}')
            # plt.xlabel('Z-offset')
            # plt.ylabel('Position along centerline')
            # plt.show()

            # CPR 3D 볼륨을 pickle 파일로 저장
            # output_dir = f"./cpr_volumes/{state}_{number}"
            # os.makedirs(output_dir, exist_ok=True)
            # output_filename = os.path.join(output_dir, f"cpr_{idx + 1}.pkl")
            # save_cpr_as_pickle(centered_cpr_volume, output_filename)

            slice_size = 32
            overlap = 16
            start = 0
            while start + slice_size <= len(centered_cpr_volume):
                volume.append(centered_cpr_volume[start:start + slice_size])
                start += (slice_size - overlap)
            if not len(volume) == 0:
                st_l = np.array(volume)
                stack.append(st_l)
    stack = np.concatenate(stack, axis=0)
    output_dir = f"./cpr_volumes/"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f"{state[0]}.pkl")
    save_cpr_as_pickle(stack, output_filename)