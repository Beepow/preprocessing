import cv2
import numpy as np
import os

img_path = 'Z:/dataset/crop/train/images'
img_names = os.listdir(img_path)

lab_path = 'Z:/dataset/crop/train/labels'
lab_names = os.listdir(lab_path)
k = 0
i=432

for k in range(0,len(img_names)-1):
    img = cv2.imread(img_path + '/' + img_names[k])
    print(img_names[k])
    Rtxt = ''
    Ltxt = ''
    with open(lab_path +'/'+ img_names[k][:-4] + '.txt', "r") as f:
        while True:
            line = f.readline()
            list = line.split()
            if not line:  # 파일 읽기가 종료된 경우
                break

            if list[0] == '1':
                a = float(list[1]) #bbox 중심 x 좌표
                b = float(list[2]) #bbox 중심 y 좌표
                c = float(list[3])
                d = float(list[4])
                a = int(a * 512)
                b = int(b * 512)
                c = int(c * 512 / 2) # bbox width 절반
                d = int(d * 512 / 2) # bbox height 절반
                output = np.zeros((d * 2, c * 2, 3), np.uint8)
                for y in range(output.shape[1]):
                    for x in range(output.shape[0]):
                        xp, yp = x + b - d, y + a - c
                        output[x, y] = img[xp, yp]
                # txt = txt

            elif list[0] == '2':
                alpha = float(list[1])
                if alpha < 0.5:
                    Rtxt = Rtxt + 'Rnor'
                elif alpha > 0.5:
                    Ltxt = Ltxt + 'Lnor'

            elif list[0] == '3':
                alpha = float(list[1])
                if alpha < 0.5:
                    Rtxt = Rtxt + 'Rrhi'
                elif alpha > 0.5:
                    Ltxt = Ltxt + 'Lrhi'

            elif list[0] == '0':
                alpha = float(list[1])
                if alpha < 0.5:
                    Rtxt = Rtxt + 'Rcys'
                elif alpha > 0.5:
                    Ltxt = Ltxt + 'Lcys'
        i = i + 1

        cv2.imwrite('Z:/dataset/crop/train/trc/' + str(Rtxt) +str(Ltxt) + '_' + str(i) +  '.jpg', output)
