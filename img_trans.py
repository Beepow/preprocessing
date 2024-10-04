from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = Image.open("001.png")
# image.show()
# img = cv2.imread("001.png")
# img.show()
# img = np.array(img)
# a = img[:,:,0]
# b = img[:,:,1]
# c = img[:,:,2]
# d = img[:,:,3]
# dat = img.load()

# width = img.size[0]
# height = img.size[1]
#
# for x in range(width):
#     for y in range(height):
#         # if dat[x,y]==(255,255,255,255):
#         #     dat[x,y]=(255,255,255,255)
#         # elif dat[x,y]==(0,0,0,0):
#         #     dat[x,y] = (0,0,0,0)
#         if dat[x,y][1] < (150) & dat[x,y][3] > 1:
#             dat[x,y] = (0,0,0,255)
#         else :
#             dat[x,y] = dat[x,y]
# img.save("r.png")

# image = img.convert("CMYK")
# # image.show()
# print(image)
# image = np.array(image)
# a = image[:,:,0]
# b = image[:,:,1]
# c = image[:,:,2]
# d = image[:,:,3]
#
# for x in range(width):
#     for y in range(height):
#         if image[x,y][1] > 0:
#             image[x,y][3] = 255
#
# plt.imshow(image)
# # image.save("rrt.png")

black = 100

width = img.size[0]
height = img.size[1]
b,g,r = img

def cmyk(r,g,b):
    if(r==0)&(g==0)&(b==0):
        return 0,0,0,black
    c = 1- r/255
    m = 1- g/255
    y = 1-b/255
    min_cmy = min(c,m,y)
    c = (c-min_cmy)/(1-min_cmy)
    m = (m - min_cmy) / (1 - min_cmy)
    y = (y - min_cmy) / (1 - min_cmy)

    k = min_cmy

    return int(c*black), int(m*black), int(y*black), int(k*black)

res = cmyk(r,g,b)

for x in range(width):
    for y in range(height):
        if res[3] > 0:
            img[x,y][3] = 255
