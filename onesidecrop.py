import cv2
import numpy as np
import os
from PIL import Image

img_path = 'Z:/dataset/cropped'
img_names = os.listdir(img_path)

k = 0
a=1
b=1
c=1
for k in range(1,len(img_names)-1):
    img = cv2.imread(img_path + '/' + img_names[k])
    print(img_names[k])

    width = img.shape[1]

    if img_names[k][0] == 'R':
        # output = img.crop(0, 0, img.size[0] / 2, img.size[1] / 2)
        output = img[:, : (width//2)]

        if img_names[k][1:4] == 'cys' :
            cv2.imwrite('Z:/dataset/one_side/Cys/cyst_' + str(a) + '.jpg', output)
            a = a + 1
        elif img_names[k][1:4] == 'rhi' :
            cv2.imwrite('Z:/dataset/one_side/Rhi/rhi_' + str(b) + '.jpg', output)
            b = b + 1
        elif img_names[k][1:4] == 'nor' :
            cv2.imwrite('Z:/dataset/one_side/Nor/nor_' + str(c) + '.jpg', output)
            c = c + 1

    if img_names[k][4] == 'L':
        # output = img.crop(img.size[0]/2, img.size[1]/2, img.size[0], img.size[1])
        output = img[:, width//2:]

        if img_names[k][5:8] == 'cys' :
            cv2.imwrite('Z:/dataset/one_side/Cys/cyst_' + str(a) + '.jpg', output)
            a = a + 1
        elif img_names[k][5:8] == 'rhi' :
            cv2.imwrite('Z:/dataset/one_side/Rhi/rhi_' + str(b) + '.jpg', output)
            b = b + 1
        elif img_names[k][5:8] == 'nor' :
            cv2.imwrite('Z:/dataset/one_side/Nor/nor_' + str(c) + '.jpg', output)
            c = c + 1

    else :
        pass


