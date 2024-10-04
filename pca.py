# import numpy as np
# from sklearn.decomposition import PCA
#
# # 전체 데이터 (20000, 270)
# data = np.random.rand(20000, 270)
#
# # 데이터를 두 개의 하위 집합으로 나눔
# data1, data2 = np.split(data, [10000])
#
# # 각각의 하위 집합에 대해 PCA 수행
# pca1 = PCA(n_components=270)
# pca2 = PCA(n_components=270)
# pca1.fit(data1)
# pca2.fit(data2)
#
# print((pca1.components_).shape)
# print((pca2.components_).shape)
#
# pca3 = PCA(n_components=270)
# pca3.fit(data)
#
# # 두 개의 주성분과 설명된 분산을 합쳐 공분산 행렬 구성
# combined_components = np.vstack((pca1.components_, pca2.components_))
# print(combined_components.shape)
# combined_explained_variance = np.hstack((pca1.explained_variance_, pca2.explained_variance_))
# print((pca1.explained_variance_).shape)
# print((pca2.explained_variance_).shape)
# print(combined_explained_variance.shape)
#
#
# # 합친 공분산 행렬로 PCA 수행
# covariance_matrix_combined = (combined_components.T @ np.diag(combined_explained_variance)) @ combined_components
# print(covariance_matrix_combined.shape)
#
# # PCA 객체 생성 후 공분산 행렬로 fitting
# pca_combined = PCA(n_components=270)
# pca_combined.fit(covariance_matrix_combined)
#
# # 원본 데이터에 대해 PCA 수행
# pca_full = PCA(n_components=270)
# pca_full.fit(data)
#
# # 주성분과 explained variance 비교
# print(np.allclose(pca_combined.components_, pca_full.components_))
# print(np.allclose(pca_combined.explained_variance_, pca_full.explained_variance_))
#
#
#
#
#
#
# # import numpy as np
# # from numpy import linalg as LA
# #
# # A = np.array([[1.1, 224.464,  3.246,    4.357,    34,  256, 3,   5,   3,   2,   5],
# #               [1,   6.4,      334.468,  9.8,      3,   4,   6,   6,   8,   3,   1],
# #               [5,   2.7,      8,        4.794,    6,   8,   9,   34,  56,  71,  1],
# #               [1.31,468.486,  3.467,    4.678,    3,   5,   6,   7,   8,   9,   21]])
# # print(A.shape)
# #
# # C = A.transpose()@A
# # print(C)
# # U, Sigma, Vt = np.linalg.svd(C)
# # # print(U)
# # # print(Sigma)
# # print(Vt)
# # # print("1")
# # # eva, eve = LA.eigh(C)
# # # print(eva)
# # # print(eve)
# # # t = eve@(np.diag(eva))@LA.inv(eve)
# # # print(t)
# # # print('----------2')
# # A1 = A[:2]
# # C1 = A1.transpose()@A1
# # # print(C1)
# # U1, Sigma1, Vt1 = np.linalg.svd(C1)
# # # print(U1)
# # # print(Sigma1)
# # print(Vt1)
# #
# # # eva1, eve1 = LA.eigh(C1)
# # # print(eva1)
# # # print(eve1)
# # # t1 = eve1@(np.diag(eva1))@LA.inv(eve1)
# # # print(t1)
# #
# # print('----------3')
# # A2 = A[2:4]
# # # print(A2)
# # C2 = A2.transpose()@A2
# # # print(C2)
# # U2, Sigma2, Vt2 = np.linalg.svd(C2)
# # # print(U2)
# # print(Sigma2)
# # print(Vt2)
# # explained_variance_ = (Sigma2 ** 2) / (11 - 1)
# # total_var = explained_variance_.sum()
# # print(total_var)
# # explained_variance_ratio_ = explained_variance_ / total_var
# # print(explained_variance_ratio_)
# #
# # VV = Vt1+Vt2
# # print(Vt - VV)
# #


# import numpy as np
#
# # 각 변수 X와 Y에 대한 데이터 생성
# X = np.random.rand(1000,40)
# print(X.shape)
# TT = np.transpose(X)@X
# print(TT.shape)
#
# # data1, data2 = np.split(data, [10000])
# X_batches = np.split(X, [500])
#
# # 각 배치에 대한 분산과 공분산 계산
# var_X_batches = [np.var(batch) for batch in X_batches]
# print(var_X_batches)
# cov = [np.cov(batch) for batch in X_batches]
# print(cov)
# total_cov_X = np.cov(X, ddof=0)[0, 0]
#
# # 각 배치의 통계량을 결합하여 전체 데이터에 대한 분산과 공분산 계산
# total_var_X = np.average(var_X_batches, weights=[len(batch) for batch in X_batches])
#
# cov_X_batches = [np.cov(batch_X, ddof=0)[0, 0] for batch_X in X_batches]
#
#
# print("전체 데이터에 대한 분산 (Var(X)):", total_var_X)
#
# for i, cov_X_batch in enumerate(cov_X_batches):
#     print(f"배치 {i+1}에 대한 공분산 (Cov(X)):", cov_X_batch)
#
# print("\n전체 데이터에 대한 공분산 (Cov(X)):", total_cov_X)


# from sklearn.decomposition import PCA
# import numpy as np
#
# # 가상의 대용량 데이터셋 생성 (X는 특성 행렬)
# # X = ...
# X = np.random.rand(30000,40)
#
# # 데이터를 두 부분으로 나누기
# split_index = 10000
# X1, X2, X3 = X[:split_index, :], X[split_index:2*split_index, :], X[2*split_index:3*split_index, :]
#
# # 각 부분에 대해 PCA 수행
# pca1 = PCA(n_components=30)
# X1_pca = pca1.fit_transform(X1)
#
# pca2 = PCA(n_components=30)
# X2_pca = pca2.fit_transform(X2)
#
# pca3 = PCA(n_components=30)
# X3_pca = pca3.fit_transform(X3)
#
# # PCA 결과 합치기
# X_combined_pca = np.vstack((X1_pca, X2_pca, X3_pca))
# print(X_combined_pca.shape)
#
# pca = PCA(n_components=30)
# X_res = pca.fit_transform(X)
# print(X_res.shape)
#
# # 추가적인 연산을 통해 필요한 결과 도출 가능
# # 예: 각 샘플에 대한 분산 계산
# var_samples_combined = np.var(X_combined_pca, axis=1)
#
# # 주성분 간 상관관계 비교
# correlation_matrix_combined = np.corrcoef(X_combined_pca.T)
# correlation_matrix_res = np.corrcoef(X_res.T)
#
# # 두 상관행렬 비교
# print(np.allclose(correlation_matrix_combined, correlation_matrix_res))
#
# difference = np.abs(X_combined_pca - X_res)
# max_difference = np.max(difference)
# mean_difference = np.mean(difference)
#
# print("Max Difference:", max_difference)
# print("Mean Difference:", mean_difference)
#
# correlation_matrix_combined = np.corrcoef(X_combined_pca.T)
# correlation_matrix_res = np.corrcoef(X_res.T)
#
# # 상관관계 행렬 출력
# print("Correlation Matrix (Combined PCA):")
# print(correlation_matrix_combined)
#
# print("\nCorrelation Matrix (Single PCA):")
# print(correlation_matrix_res)
#
# # 두 상관행렬 간의 차이 출력
# correlation_difference = np.abs(correlation_matrix_combined - correlation_matrix_res)
# max_correlation_difference = np.max(correlation_difference)
# mean_correlation_difference = np.mean(correlation_difference)
#
# print("\nMax Correlation Difference:", max_correlation_difference)
# print("Mean Correlation Difference:", mean_correlation_difference)


import numpy as np
from sklearn.decomposition import PCA

# 가상의 이미지 데이터 생성 (실제 데이터로 대체해야 함)
class_1_data = np.random.rand(100, 32, 32, 3)
class_2_data = np.random.rand(100, 32, 32, 3)

# 채널방향으로 복사
expanded_class_1_data = np.repeat(class_1_data, 9, axis=-1)
expanded_class_2_data = np.repeat(class_2_data, 9, axis=-1)

# 이미지 데이터를 (102400, 27) 형태로 평탄화
flattened_class_1_data = expanded_class_1_data.reshape((100, -1))
flattened_class_2_data = expanded_class_2_data.reshape((100, -1))

# PCA 모델 생성
n_components = 27  # 주성분 개수 설정

# 클래스 1에 대한 PCA 수행
pca_class_1 = PCA(n_components=n_components)
pca_class_1.fit(flattened_class_1_data)

# 클래스 2에 대한 PCA 수행
pca_class_2 = PCA(n_components=n_components)
pca_class_2.fit(flattened_class_2_data)

# 클래스 1의 주성분과 클래스 2의 주성분 비교
for i in range(n_components):
    correlation = np.corrcoef(pca_class_1.components_[i], pca_class_2.components_[i])[0, 1]
    print(f"Correlation between component {i+1} of class 1 and class 2: {correlation}")
