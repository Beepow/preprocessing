import cv2
import numpy as np

img = cv2.imread('Z:/test/1normal005_jpg.rf.4d3d05c22e63dd0b9a7a81b78bb1a4ab.jpg')
print(img.shape, img.dtype)

with open ("Z:/test/1normal005_jpg.rf.4d3d05c22e63dd0b9a7a81b78bb1a4ab.txt", "r") as f:
    while True:
        line = f.readline()
        list = line.split()
        if not line:  # 파일 읽기가 종료된 경우
            break

        if list[0] == '1':
            print(list)
            a = float(list[1])
            b = float(list[2])
            c = float(list[3])
            d = float(list[4])
            a = int(a*512)
            b = int(b*512)
            c = int(c*512 /2)
            d = int(d*512 /2)
            print(a,b,c,d)
            output = np.zeros((d*2 ,c*2  ,3), np.uint8)
            for y in range(output.shape[1]):
                for x in range(output.shape[0]):
                    xp, yp = x + b-d , y+a-c
                    output[x, y] = img[xp, yp]
            txt = ''

        if list[0] == '2':
            txt = txt + 'N'

        if list[0] == '3':
            txt = txt + 'R'

        if list[0] == '0':
            txt = txt + 'C'

    cv2.imwrite('cropped_' + str(txt) + '.jpg', output)



