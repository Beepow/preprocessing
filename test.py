# # import numpy as np
# #
# # # # 예시 배열 생성 (임의의 데이터로 예시)
# # # recon = np.random.rand(30, 128, 128, 5, 9)
# # # trans_W = np.random.rand(5, 8, 9)
# # #
# # # print(trans_W[0].shape)
# # # print(trans_W.shape)
# # # print(recon[: ,: ,: , 0,:].shape)
# # # S = recon.shape
# # # transformed = []
# # # for i in range(S[3]):
# # #     transformed_feature = np.matmul(recon[: ,: ,: , i,:], np.transpose(trans_W[i]))
# # #     print(transformed_feature.shape)
# # #     transformed.append(transformed_feature)
# # #
# # # transformed = np.stack(transformed, axis=3)
# # # print(transformed.shape)
# # #
# # # x = np.random.rand(30, 128, 128, 8)
# # # y = np.random.rand(30, 128, 128, 7)
# # #
# # # r = np.concatenate((x,y), axis=3)
# # # print(r.shape)
# #
# # arr1 = np.arange(1, 46)
# # arr2 = np.reshape(np.arange(1, 361), (5,8,9))
# # arr4 = np.arange(1,1801).reshape((45,40))
# #
# # print(arr1.shape)
# # print(arr2.shape)
# #
# # # x = arr1@arr2
# # # print(x)
# # arr3 = np.reshape(arr1, (5,9))
# # print(arr3.shape)
# #
# # y1 = arr3@np.transpose(arr2[0])
# # print(y1.shape)
# # print(arr2[0].shape)
# # y2 = arr3@np.transpose(arr2[1])
# # print(y2)
# # y3 = arr3@np.transpose(arr2[2])
# # print(y3)
# # y4 = arr3@np.transpose(arr2[3])
# # print(y4)
# # y5 = arr3@np.transpose(arr2[4])
# # print(y5)
# #
# # z = np.concatenate((y1,y2,y3,y4,y5), axis=1)
# # print(z.shape)
# # re = np.sum(z, axis=0)
# # print(re)
# #
# # k = arr1@arr4
# # print(k.shape)
# # print(k)
#
# import numpy as np
# arr1 = np.arange(1, 6)
# print(arr1)
# arr2 = np.arange(1,4)
# print(arr2)
# x = np.concatenate((arr1, arr2))

# import numpy as np
# A = np.random.randn(50, 128, 128, 9)
# D = A.reshape(-1, A.shape[-1])
# B = np.mean(A)
# print(B)
# C = np.mean(A, axis=3)
# print(C.shape)
# print(np.mean(D, axis=0))
# print(A[0:5, :,:,:].shape)
# Y = np.mean(A[0:10, :,:,:])
# print(Y)
# Z = np.mean(A[10:20, :,:,:])
# print(Z)
# T = np.mean(A[20:30, :,:,:])
# print(T)
# P = np.mean(A[30:40, :,:,:])
# print(P)
# K = np.mean(A[40:50, :,:,:])
# print(K)
# print((Y+Z+T+K+P)/5)


R, C = map(int, input().split(' '))
N = int(input())
l = [[] for _ in range(N)]

answer = [1, 1]
for _ in range(N):
    a, v, h = map(int, input().split(' '))
    l[a-1].append([v,h])

l = [j for j in l if len(j) > 1]
print(l)
for i, k in enumerate(l):
    if len(k) != 0:
        x = max([kk[0] for kk in k]) - min([kk[0] for kk in k]) + 1
        y = max([kk[1] for kk in k]) - min([kk[1] for kk in k]) + 1
        print(x,y)
        if answer[1] < x*y:
            answer[1] = x*y
print(answer[0], answer[1])