import aedat
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image, ImageDraw
import math
import time

NAS = "C:/Users/최재원/Desktop/datasets_v4/"
char = ['dog'] ###'airplane', 'dog', 'cat','truck', 'ship', 'horse', 'frog', 'deer', 'bird', 'automobile'
for ch in range(len(char)):
    for idx in range(len(os.listdir(NAS + char[ch])) - 1)[115:116]:

        print("Start---------------------")
        # data = f"./cifar10_frog_125.aedat4"
        data = f"/cifar10_{char[ch]}_{idx}.aedat4"
        # print(f"DataName : {data}")
        output = f"C:/Users/Public/Pycharm/preprocessing/TBR/{char[ch]}"#/{char[ch]}_{str(i)}
        # output = "./TBR"
        print(f"OutPut : {output}_{idx}")
        # OUT = f"/home/jwchoi/TBR_png/{char[ch]}"
        if not os.path.exists(output):
            os.makedirs(output)

        decoder = aedat.Decoder(NAS + char[ch] + data)
        # decoder = aedat.Decoder(data)
        print(decoder.id_to_stream())

        h = decoder.id_to_stream()[0].get('height')
        w = decoder.id_to_stream()[0].get('width')
        type = decoder.id_to_stream()[0].get('type')

        xy_points = []
        colors = []
        cnt = 1
        al_time = []
        al_polarity = []
        al_xy = []
        idx_list = []
        dataarray = []

        for packet in decoder:
            if "events" in packet:
                timestamp = packet["events"]["t"]
                x = packet["events"]["x"]
                y = packet["events"]["y"]
                polarity = (packet["events"]["on"]).astype(int)
                x = [127 - xi for xi in x]
                y = [127 - yi for yi in y]
                xy_points = list(zip(x,y))

                # print("{} timestamps".format(len(packet["events"]["t"])))
                # print(np.max(timestamp) - np.min(timestamp))

                al_time.extend(timestamp)
                al_polarity.extend(polarity)
                al_xy.extend(xy_points)

        for p in range(len(al_time)):
            set_length = 33333 #30000 ### us, micro second /al_time[-1]
            closest_index = None
            cl_value = min(al_time, key=lambda x: abs(x - p*set_length))
            index = min(al_time.index(cl_value), al_time.index(al_time[-1]))
            idx_list.append(index)

            if p*set_length > al_time[-1]:
                break
        N = 7
        v = 0
        emt = np.full((h, w), 0, dtype=np.uint8)

        for i in range(len(idx_list) -1):
            t0 = time.time()
            # image = Image.new("RGB", (w, h), (0, 0, 0))  ### RGB color
            image1 = np.zeros((N, w, h))### GrayScale
            image2 = np.zeros((N, w, h))
            mask = np.ones((N, w, h))
            f1 = np.zeros((w, h))
            f2 = np.zeros((w, h))
            k = 0
            try:
                for c in range(idx_list[i], idx_list[i + 1]):
                    if al_polarity[c] == 1:
                        f1[al_xy[c]] = 1
                    elif al_polarity[c] == 0:
                        f2[al_xy[c]] = -1

                    if c == np.round(idx_list[i] + (idx_list[i + 1] - idx_list[i]) * (k + 1) / N) - 1:
                        image1[k, ...] = f1
                        image2[k, ...] = f2
                        f1 = np.zeros((w, h))
                        f2 = np.zeros((w, h))
                        mask[k, :, :] = 2 ** k
                        k += 1
                ff = (np.sum((image1 * mask), 0) + 128).astype(np.int32)
                frame1 = (np.sum((image1 * mask), 0)).astype(np.int32)
                frame2 = (np.sum((image2 * mask), 0) + 128).astype(np.int32)
                frame = frame1 + frame2
                # imgcon = np.stack((frame1, emt, frame2), axis=-1)
                    # elif al_polarity[c] == 0:
                    #     image[al_xy[c]] = -1
            except Exception as e:
                print(f"error---------{e}")
                continue

            # if cnt == 10 or cnt ==11 or cnt==12 or cnt==9:
            cv2.imwrite(f"{output}/{idx}_{cnt}.png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])

            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_box_aspect([15, 3, 15])
            indices_P = [i for i, pol in enumerate(al_polarity[idx_list[i]:idx_list[i+1]]) if pol == 1]
            indices_M = [i for i, pol in enumerate(al_polarity[idx_list[i]:idx_list[i+1]]) if pol == 0]
            # x, y = zip(*al_xy[indices])
            xp = [al_xy[i][0] for i in indices_P]
            yp = [al_xy[i][1] for i in indices_P]
            yp = [128 - yi for yi in yp]
            tp = [al_time[i] for i in indices_P]
            xm = [al_xy[i][0] for i in indices_M]
            ym = [al_xy[i][1] for i in indices_M]
            ym = [128 - yi for yi in ym]
            tm = [al_time[i] for i in indices_M]
            ax.scatter(yp, tp, xp, c='r', s=1)
            ax.scatter(ym, tm, xm, c='b', s=1)
            ax.set_yticks([])
            plt.savefig(f"{output}/{set_length}us_{cnt}.png")
            plt.show()

            cnt += 1
            t1 = time.time()
            print("End -------------------------------%10f seconds" % (t1 - t0))


