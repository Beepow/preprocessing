import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


def read_ndataset(filename):
    with open(filename, 'rb') as f:
        evt_stream = np.frombuffer(f.read(), dtype=np.uint8)

    TD = {}
    TD['x'] = evt_stream[0::5] + 1  # pixel x address
    TD['y'] = evt_stream[1::5] + 1  # pixel y address
    TD['p'] = (evt_stream[2::5] >> 7) + 1  # polarity
    TD['ts'] = ((evt_stream[2::5] & 127) << 16)  # timestamp (microseconds)
    TD['ts'] += (evt_stream[3::5] << 8)
    TD['ts'] += evt_stream[4::5]
    print(len(TD['p']), max(TD['ts']))
    return TD


def create_images_from_events(td_events, output_folder, set_length=33333, N=7, w=240, h=180):
    sort = np.argsort(td_events['ts'])

    al_time = td_events['ts'][sort]
    al_polarity = td_events['p'][sort]
    al_xy = list(zip(td_events['x'][sort], td_events['y'][sort]))
    idx_list = []

    start = 0
    overlap=32
    while start < len(al_time):
        cl_value = min(al_time, key=lambda x: abs(x - start))
        index = al_time.tolist().index(cl_value)
        idx_list.append(index)
        start += set_length - overlap
        if start >= max(al_time):
            break
    cnt = 0

    for i in range(len(idx_list) - 1):
        t0 = time.time()
        Plus = np.zeros((N, h, w))
        Minus = np.zeros((N, h, w))
        mask = np.ones((N, h, w))
        PFrame = np.zeros((h, w))
        MFrame = np.zeros((h, w))
        k = 0
        try:
            for c in range(idx_list[i], idx_list[i + 1]):
                if al_polarity[c] == 2:
                    PFrame[al_xy[c][1], al_xy[c][0]] = 1
                elif al_polarity[c] == 1:
                    MFrame[al_xy[c][1], al_xy[c][0]] = 1
                if c == np.round(idx_list[i] + (idx_list[i + 1] - idx_list[i]) * (k + 1) / N) - 1:
                    Plus[k, ...] = PFrame
                    Minus[k, ...] = MFrame
                    PFrame = np.zeros((h, w))
                    MFrame = np.zeros((h, w))
                    mask[k, :, :] = 2 ** k
                    k += 1
            # frame1 = (np.sum((image1 * mask), 0)).astype(np.int32)
            # frame2 = (np.sum((image2 * mask), 0) + 128).astype(np.int32)
            # frame = frame1 + frame2

            Plus = (np.sum((Plus * mask), 0)).astype(np.int32)
            Minus = (np.sum((Minus * mask), 0)).astype(np.int32)
            empty_green = np.zeros((h, w), dtype=np.int32)
            rgb_frame = np.stack((Plus, empty_green, Minus), axis=-1).astype(np.int32)

            cv2.imwrite(f"{output_folder}/{i}.png", rgb_frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            cnt += 1
            t1 = time.time()
            print(f"End ------------------------------- {t1 - t0:.6f} seconds")

        except Exception as e:
            print(f"error---------{e}")
            continue


def process_all_bin_files(root_folder, output_root):
    for class_folder in os.listdir(root_folder)[4:]:
        class_path = os.path.join(root_folder, class_folder)
        if os.path.isdir(class_path):
            for bin_file in os.listdir(class_path):
                if bin_file.endswith(".bin"):
                    bin_path = os.path.join(class_path, bin_file)
                    print(f"Processing {bin_path}...")

                    bin_output_folder = os.path.join(output_root, class_folder, bin_file.replace(".bin", ""))
                    os.makedirs(bin_output_folder, exist_ok=True)

                    td_events = read_ndataset(bin_path)
                    create_images_from_events(td_events, bin_output_folder, set_length=64, N=8, w=36, h=36)#(max(td_events['ts'])+1)/4


# 경로 설정
train_folder = "C:/data/Train/Train"  # bin 파일들이 들어있는 폴더 경로
output_folder = "C:/data/NMNIST_7"  # 생성된 이미지를 저장할 폴더 경로

# 모든 bin 파일을 처리하고 이미지 생성
process_all_bin_files(train_folder, output_folder)

