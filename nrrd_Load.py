import nrrd
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import sklearn.preprocessing as PP

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

NAME = 'Normal_4'
CT = 1
Anno = 1
if CT:
    CTCA_data, header = nrrd.read(f'C:/Users/최재원/Desktop/ASOCADataAccess/ASOCADataAccess/CompressedVersion/Normal/CTCA/{NAME}.nrrd')
    data = np.moveaxis(CTCA_data, -1, 0)
    CTCA_coord = []
    for zC, img_data in enumerate(CTCA_data):
        for yC, row in enumerate(img_data):
            for xC, value in enumerate(row):
                if value == 1:
                    CTCA_coord.append((xC, yC, zC))
    xC = [coord[0] for coord in CTCA_coord]
    yC = [coord[1] for coord in CTCA_coord]
    zC = [coord[2] for coord in CTCA_coord]
    ax.scatter(xC, yC, zC, c='b', alpha=0.2, s=1)

if Anno:
    data, header = nrrd.read(f'C:/Users/최재원/Desktop/ASOCADataAccess/ASOCADataAccess/CompressedVersion/Normal/Annotations/{NAME}.nrrd')
    data = np.moveaxis(data, -1, 0)
    data_coord = []
    for zD, img_data in enumerate(data):
        for yD, row in enumerate(img_data):
            for xD, value in enumerate(row):
                if value == 1:
                    data_coord.append((xD, yD, zD))
    xD = [coord[1] for coord in data_coord]
    yD = [coord[0] for coord in data_coord]
    zD = [coord[2] for coord in data_coord]
    ax.scatter(xD, yD, zD, c='k',  s=1)

ax.set_ylabel('Y')
ax.set_xlabel('X')
ax.set_zlabel('D')
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])
# plt.legend()
plt.show()


output = f"C:/Users/최재원/Desktop/ASOCADataAccess/array_rep/{NAME}"
if not os.path.exists(output):
    os.makedirs(output)

scaler = PP.MinMaxScaler()
data = (data*255).astype(np.uint8)
for k in range(data.shape[0]):
    # annodata = data[k, :, :]
    # ctdata = (CTCA_data[k, :, :]).astype(np.uint8)
    ctdata = data[k, :, :]

    normalized_image = np.clip(ctdata, -300, 400)

    plt.imshow(normalized_image[:, :], cmap='gray')
    plt.colorbar()
    plt.show()
    # cv2.imwrite(f"./rep/{k+1}.png", normalized_image)


