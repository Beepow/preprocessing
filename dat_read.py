import numpy as np
import matplotlib.pyplot as plt


def load_ncars_data(filename):
    with open(filename, 'rb') as f:
        # 파일에서 타임스탬프 (4바이트)와 데이터 (4바이트)를 읽어들임
        data = np.fromfile(f, dtype=np.uint32)

    # 데이터는 타임스탬프와 데이터로 구성되므로 짝수로 나누어야 함
    timestamps = data[::2]  # 타임스탬프는 짝수 인덱스
    addresses = data[1::2]  # 데이터는 홀수 인덱스

    # 데이터에서 x, y 좌표와 폴라리티를 추출
    x_mask = 0x3FFF  # x 좌표를 얻기 위한 마스크 (14비트)
    y_mask = 0xFFFC000  # y 좌표를 얻기 위한 마스크 (14비트, 18-31 비트)
    polarity_mask = 0x20000000  # 폴라리티를 얻기 위한 마스크 (29번째 비트)

    x = (addresses & x_mask)  # x 좌표 추출
    y = (addresses & y_mask) >> 14  # y 좌표 추출
    polarity = np.where((addresses & polarity_mask) != 0, 1, -1)  # 폴라리티 추출

    return {'ts': timestamps, 'x': x, 'y': y, 'p': polarity}


def events_to_image(events, img_shape=(128, 128), time_window=100):
    ts, x, y, p = events['ts'], events['x'], events['y'], events['p']

    # 모든 배열의 크기를 동일하게 맞추기 위해 최소 크기 확인
    min_length = min(len(ts), len(x), len(y), len(p))
    ts = ts[:min_length]
    x = x[:min_length]
    y = y[:min_length]
    p = p[:min_length]

    # 타임스탬프 기준으로 모든 데이터를 정렬
    sorted_idx = np.argsort(ts)
    ts = ts[sorted_idx]
    x = x[sorted_idx]
    y = y[sorted_idx]
    p = p[sorted_idx]

    images = []
    start_time = ts[0]
    end_time = ts[-1]

    print(f"Start time: {start_time}, End time: {end_time}")

    # 좌표 값이 img_shape에 맞도록 스케일링 (필요시 조정)
    x = np.clip(x, 0, img_shape[1] - 1)
    y = np.clip(y, 0, img_shape[0] - 1)

    for t in range(start_time, end_time, time_window):
        img = np.zeros(img_shape, dtype=np.int8)
        idx = (ts >= t) & (ts < t + time_window)
        print(f"Time window: {t} - {t + time_window}, Events in this window: {np.sum(idx)}")

        if np.any(idx):
            img[y[idx], x[idx]] += p[idx]
            images.append(img)

    if len(images) == 0:
        print("No images to display.")
    return images


def plot_image_sequence(images):
    if len(images) == 0:
        print("No images to display.")
        return

    fig, ax = plt.subplots(1, len(images), figsize=(15, 5))
    for i in range(len(images)):
        ax[i].imshow(images[i], cmap='gray')
        ax[i].axis('off')
    plt.show()


# 예시 사용
filename = 'C:/data/n-cars_train/cars/obj_004396_td.dat'
events = load_ncars_data(filename)
# 배열 크기 확인
print(f"ts size: {len(events['ts'])}, x size: {len(events['x'])}, y size: {len(events['y'])}, p size: {len(events['p'])}")

# 이벤트를 이미지 시퀀스로 변환
images = events_to_image(events, img_shape=(128, 128), time_window=10000)

# 이미지 시퀀스를 시각화
plot_image_sequence(images)
