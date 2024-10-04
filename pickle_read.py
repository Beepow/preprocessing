import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#6f290a', '#87c392', '#48ff71', '#ffff45', '#ff45ff', '#45ffff', '#180483']


#################################################################################

def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)

def nearest_points(coords_list1, coords_list2):
    closest_pairs = []
    for point1 in coords_list1:
        min_distance = float('inf')
        closest_point = None
        for point2 in coords_list2:
            dist = distance(point1, point2)
            if dist < min_distance:
                min_distance = dist
                closest_point = point2
        closest_pairs.append((point1, closest_point))
    return closest_pairs

state = 'Normal'
number = '4'
NAME = f'{state}_{number}'
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
with open(f"C:/Users/최재원/Desktop/ASOCADataAccess/vessel/centerlines_main_3/centerlines_main_3_{NAME}.pkl", "rb") as f:
    groups = pickle.load(f)

listup = []
for i in range((len(groups))):
    g1 = np.array(groups[i])
    # if i == 10:
    #     g1 = g1[10:-int(len(g1)/1.7)]
    #     # g1 = np.concatenate((g1[80:410],g1[75:0:-1],g1[410:-int(len(g1)/1.7)]), axis=0)
    #     ax.scatter(g1[:, 0], g1[:, 1], g1[:, 2], c=colors[i], s=1, label=i)
    #     ax.scatter(g1[0, 0], g1[0, 1], g1[0, 2], c='k', s=10)
    #     listup.append(g1)
    # elif i== 1:
    #     # g1 = g1[-23:-1]
    #     g1 = np.concatenate((g1[0:70], g1[130:-23], g1[120:70:-1], g1[-23:]), axis=0)
    #     ax.scatter(g1[:, 0], g1[:, 1], g1[:, 2], c=colors[i], s=1, label=i)
    #     ax.scatter(g1[0, 0], g1[0, 1], g1[0, 2], c='k', s=10)
    #     listup.append(g1)
    # elif i == 12:
    #     g1 = g1[10:-int(len(g1)/1.7)]
    #     # g1 = np.concatenate((g1[80:170],g1[75:0:-1],g1[170:-int(len(g1)/2)]), axis=0)
    #     ax.scatter(g1[:, 0], g1[:, 1], g1[:, 2], c=colors[i], s=1, label=i)
    #     ax.scatter(g1[0, 0], g1[0, 1], g1[0, 2], c='k', s=10)
    #     listup.append(g1)
    # elif i == 3:
    #     g1 = g1[100:-int(len(g1)/1.7)]
    #     # g1 = np.concatenate((g1[3:200], groups[6][63:230],groups[6][60:1:-1], groups[7][1:]), axis=0)
    #     ax.scatter(g1[:, 0], g1[:, 1], g1[:, 2], c=colors[i], s=1, label=i)
    #     ax.scatter(g1[0, 0], g1[0, 1], g1[0, 2], c='k', s=10)
    #     listup.append(g1)
    # elif i==10  or i==12 or i==3:
    #     g1 = None
    #     pass
    # else:
    #     ax.scatter(g1[:, 0], g1[:, 1], g1[:, 2], c=colors[i], s=1, label=i)
    #     ax.scatter(g1[0, 0], g1[0, 1], g1[0, 2], c='k', s=10)
    #     listup.append(g1)
    ax.scatter(g1[:, 0], g1[:, 1], g1[:, 2], c=colors[i], s=1, label=i)
    ax.scatter(g1[0, 0], g1[0, 1], g1[0, 2], c='k', s=10)
    listup.append(g1)

print(len(listup))
# ax.set_ylabel('Y')
ax.set_xlabel('X')
ax.set_zlabel('D')
ax.legend()
plt.savefig(f"C:/Users/최재원/Desktop/ASOCADataAccess/vessel/centerlines_main_3_fig/{NAME}.png")
plt.show()
print("saved")

with open(f"C:/Users/최재원/Desktop/ASOCADataAccess/vessel/centerlines_main_3/centerlines_main_3_{NAME}.pkl", "wb") as f:
    pickle.dump(listup, f, protocol=pickle.HIGHEST_PROTOCOL)
print("saved")