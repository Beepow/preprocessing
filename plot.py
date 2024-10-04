# # import matplotlib.pyplot as plt
# # from mpl_toolkits.mplot3d import Axes3D
# # import aedat
# # import numpy as np
# #
# # data = f"C:/Users/최재원/Desktop/test/cifar10_dog_16.aedat4"
# #
# # decoder = aedat.Decoder(data)
# #
# # index = 0
# # h = decoder.id_to_stream()[0].get('height')
# # w = decoder.id_to_stream()[0].get('width')
# #
# # xy_points = []
# # colors = []
# # cnt = 1
# # al_time = []
# # al_polarity = []
# # al_xy = []
# # idx_list = []
# # dataarray = []
# #
# # for packet in decoder:
# #     if "events" in packet:
# #
# #         timestamp = packet["events"]["t"]
# #         x = packet["events"]["x"]
# #         y = packet["events"]["y"]
# #         polarity = (packet["events"]["on"]).astype(int)
# #         xy_points = list(zip(x,y))
# #
# #         al_time.extend(timestamp)
# #         al_polarity.extend(polarity)
# #         al_xy.extend(xy_points)
# # xy1 = []
# # xy2 = []
# # al_time = al_time[5000:10000]
# # for t in range(len(al_time)):
# #     if al_polarity[t] == 1:
# #         xy1.append(al_xy[t])
# #         xy2.append([0, 0])
# #     elif al_polarity[t] == 0:
# #         xy2.append(al_xy[t])
# #         xy1.append([0,0])
# #
# # x1 = [sublist[0] for sublist in xy1]
# # y1 = [sublist[1] for sublist in xy1]
# #
# # x2 = [sublist[0] for sublist in xy2]
# # y2 = [sublist[1] for sublist in xy2]
# #
# # # 3차원 이벤트 플롯
# # fig = plt.figure(figsize=(16, 9))
# # ax = fig.add_subplot(111, projection='3d')
# #
# # ax.set_box_aspect([1, 3, 1])
# # # 이벤트를 3차원 공간에 플롯
# # # ax.scatter(X, al_time, Y, c='r', marker='.')
# # ax.scatter(y1, al_time, x1, c='red', marker='.', label='Red Group')
# #
# # # 파란색 데이터 그룹 플롯
# # ax.scatter(y2, al_time, x2, c='blue', marker='.', label='Blue Group')
# #
# # # ax.set_xticks(np.arange(min(al_time), max(al_time)+0.5, 0.1))
# # # 축 레이블
# # ax.set_xlabel('axis_X')
# # ax.set_ylabel('Time (us)')
# # ax.set_zlabel('axis_Y')
# # # ax.set_yticks([])
# # # 그래프 제목
# # ax.xaxis.labelpad = 10  # x 축 라벨 간격 조절
# # ax.yaxis.labelpad = 25  # y 축 라벨 간격 조절
# # ax.zaxis.labelpad = 15  # z 축 라벨 간격 조절
# #
# # ax.tick_params(axis='x', pad=5)  # x 축 눈금 간격 조절
# # ax.tick_params(axis='y', pad=5)  # y 축 눈금 간격 조절
# # ax.tick_params(axis='z', pad=10)  # z 축 눈금 간격 조절
# #
# # plt.title('3D Events Over Time')
# #
# # # 플롯 보여주기
# # plt.show()
#
#
#
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import aedat
# import numpy as np
#
# data = f"C:/Users/최재원/Desktop/test/cifar10_dog_16.aedat4"
#
# decoder = aedat.Decoder(data)
#
# h = decoder.id_to_stream()[0].get('height')
# w = decoder.id_to_stream()[0].get('width')
#
# data = []
# for packet in decoder:
#     if "events" in packet:
#
#         timestamp = packet["events"]["t"]
#         x = packet["events"]["x"]
#         y = packet["events"]["y"]
#         polarity = (packet["events"]["on"]).astype(int)
#         data_array = (np.stack((timestamp,x,y,polarity), -1)).astype(int)
#         data.append(data_array)
#
# # data = (np.vstack(data))[15000:515000]
# data = np.vstack(data)
# # 데이터를 polarity에 따라 분리
# positive_data = [(t, x, y) for t, x, y, p in data if p > 0]
# negative_data = [(t, x, y) for t, x, y, p in data if p == 0]
#
# # 시각화
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111, projection='3d')
# ax.set_box_aspect([10,50,10])
# # 양수 데이터 파란색으로 플로팅
# if positive_data:
#     times, xs, ys = zip(*positive_data)
#     # ax.scatter(ys, times, xs, c='blue',  s=1)
#
# # 음수 데이터 빨간색으로 플로팅
# if negative_data:
#     times, xs, ys = zip(*negative_data)
#     ax.scatter(ys, times, xs, c='red',  s=1)
#
# plt.xlabel(None)
# plt.ylabel(None)
# # # 축 라벨링
# # ax.set_ylabel('Time')
# # ax.set_xlabel('X-axis')
# # ax.set_zlabel('Y-axis')
# # # ax.set_yticks([])
# #
# # ax.xaxis.labelpad = 10  # x 축 라벨 간격 조절
# # ax.yaxis.labelpad = 25  # y 축 라벨 간격 조절
# # ax.zaxis.labelpad = 15  # z 축 라벨 간격 조절
# # #
# # ax.tick_params(axis='x', pad=5)  # x 축 눈금 간격 조절
# # ax.tick_params(axis='y', pad=5)  # y 축 눈금 간격 조절
# # ax.tick_params(axis='z', pad=10)  # z 축 눈금 간격 조절
# # #
# # # 범례 추가
#
# plt.yticks([])
#
# plt.legend()
#
# # 그래프 표시
# plt.show()

# import matplotlib.pyplot as plt
#
# # 예시 데이터 (모델 이름과 정확도)
# models = ['Hop1', 'Hop2', 'Hop3', 'Hop4']
# accuracies = [0.71, 0.74, 0.77, 0.79]
#
# # 그래프 그리기
# plt.plot(models, accuracies, marker='o', linestyle='-', color='blue')
# plt.xlabel('Number of Event-VoxelHop units')
# plt.ylabel('Acc')
# plt.title('Hops')
#
# plt.ylim(0.5, 1.0)
# # 정확도 값 표시
# for i, accuracy in enumerate(accuracies):
#     plt.text(i, accuracy + 0.01, f'{accuracy:.2f}', ha='center', va='bottom')
#
# # 그래프 표시
# plt.show()
#
import matplotlib.pyplot as plt
import numpy as np

# x축을 AC component로, y축을 Log of energy로 하는 데이터 생성 예시
x_values = [0.01, 0.008, 0.005, 0.001, 0.0008, 0.00065, 0.0005]  # 0부터 30까지의 정수형 값
y_values = [0.9125, 0.8775, 0.8975, 0.885, 0.855, 0.91, 0.86]  # 랜덤한 로그 값으로 예시

# 그래프 그리기
plt.plot(x_values, y_values, marker='o', linestyle='--')

# 그래프에 레이블 추가
plt.xlabel('Energy Threshold')
plt.ylabel('Accuracy')

# 그래프 제목 추가
plt.title('DVS-CIFAR10 (Dog vs truck)')
plt.xscale('log')

# 그리드 표시
plt.grid(True)

# 그래프 표시
plt.show()
