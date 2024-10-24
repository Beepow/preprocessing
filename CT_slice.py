import nrrd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

state = 'Diseased'
num = '1'
data_name = state + '_' + num
ct_data, ct_header = nrrd.read(f'./ASOCAData/{state}/CTCA/{data_name}.nrrd')
annotation_data, annotation_header = nrrd.read(f'./ASOCAData/{state}/Annotations/{data_name}.nrrd')

with open(f'./CntF/centerlines_float_{data_name}.pkl', 'rb') as f:
    centerline_data = pickle.load(f)

output_dir = './CT_slice'
os.makedirs(output_dir, exist_ok=True)


num_slices = ct_data.shape[2]
for h in range(num_slices):

    ct_slice = ct_data[:, :, h]
    ct_slice = np.clip(ct_slice, -300, 300)
    min_val = np.min(ct_slice)
    max_val = np.max(ct_slice)
    ct_slice = (ct_slice - min_val) / (max_val - min_val)
    annotation_slice = annotation_data[:, :, h]

    colored_slice = np.stack([ct_slice] * 3, axis=-1)  # 3채널

    # annotation 부분 파란색 적용
    colored_slice[annotation_slice > 0] = [1, 0, 0]  # 빨간색

    for vessel in centerline_data:
        for (x, y, z) in vessel:
            if int(z) == h:
                colored_slice[int(x), int(y)] = [0, 0, 1]  # 파란색

    plt.imsave(os.path.join(output_dir, f'slice_{h:04d}.png'), colored_slice)