import cv2
import numpy as np
from PIL import Image, ImageDraw


img = cv2.imread('Z:/dataset/test/1normal010_jpg.rf.b149932998b4e3d397f7eec3b78817b8.jpg')
print(img.shape, img.dtype)

txt = ''

with open ("Z:/dataset/test/1normal010_jpg.rf.b149932998b4e3d397f7eec3b78817b8.txt", "r") as f:
    while True:
        line = f.readline()
        list = line.split()
        if not line:  # 파일 읽기가 종료된 경우
            break

        if list[0] == '1':
            a = float(list[1]) # bbox 중심 x 좌표
            b = float(list[2]) # bbox 중심 y 좌표
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
            txt = ''

        elif list[0] == '2':
            alpha = float(list[1])
            if alpha < 0.5:
                txt = txt + 'Rnor'
            elif alpha > 0.5:
                txt = txt + 'Lnor'

        elif list[0] == '3':
            alpha = float(list[1])
            if alpha < 0.5:
                txt = txt + 'Rrhi'
            elif alpha > 0.5:
                txt = txt + 'Lrhi'

        elif list[0] == '0':
            alpha = float(list[1])
            if alpha < 0.5:
                txt = txt + 'Rcys'
            elif alpha > 0.5:
                txt = txt + 'Lcys'

    cv2.imwrite('Z:/dataset/test/_cropped_' + str(txt) + '.jpg', output)

# im = img.convert('RGB')
# draw = ImageDraw.Draw(im)
# draw.rectangle((a,b,c,d), outline=(0,255,0), width = 5)
draw = cv2.rectangle(img, (a+c,b+d),(a-c,b-d), (255,0,0))
cv2.imwrite('Z:/dataset/test/rgb_' + str(txt) + '.jpg', draw)
