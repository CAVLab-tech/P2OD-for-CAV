import numpy as np
import time


# 返回一个池化区域内最大值分量（映射至原始图像）
def MaxPool(X):  # X代表待池化图像，池化区域为2×2
    ch, h, w = X.shape
    X = np.pad(X, ((0, 0), (0, h % 2), (0, w % 2)), 'minimum')

    # 将池化区域拆分为四个子区域，然后两两进行三轮比较，分别输出最大的分量
    X = X.transpose(1, 2, 0)  # 必须转置，因为按奇偶下标拆分子矩阵是按维度顺序
    X_0_0 = X[::2, ::2]  # ::2取奇数位，1::2取偶数位
    X_0_1 = X[::2, 1::2]
    X_1_0 = X[1::2, ::2]
    X_1_1 = X[1::2, 1::2]

    x = np.where(X_0_0 - X_0_1 > 0, X_0_0, X_0_1)
    x = np.where(x - X_1_0 > 0, x, X_1_0)
    x = np.where(x - X_1_1 > 0, x, X_1_1)
    x = x.transpose(2, 0, 1)
    return x


# 返回一个池化区域内最大值分量（映射至原始图像）
def SecMaxPool(X,Y):  # X代表子图像1，Y代表子图像2，池化区域为2×2
    ch, h, w = X.shape
    X = np.pad(X, ((0, 0), (0, h % 2), (0, w % 2)), 'minimum')
    Y = np.pad(Y, ((0, 0), (0, h % 2), (0, w % 2)), 'minimum')

    # 将池化区域拆分为四个子区域，然后两两进行三轮比较，分别输出最大的分量
    X = X.transpose(1, 2, 0)  # 必须转置，因为按奇偶下标拆分子矩阵是按维度顺序
    Y = Y.transpose(1, 2, 0)
    X_0_0 = X[::2, ::2]   # ::2取奇数位，1::2取偶数位
    X_0_1 = X[::2, 1::2]
    X_1_0 = X[1::2, ::2]
    X_1_1 = X[1::2, 1::2]

    Y_0_0 = Y[::2, ::2]   
    Y_0_1 = Y[::2, 1::2]
    Y_1_0 = Y[1::2, ::2]
    Y_1_1 = Y[1::2, 1::2]
    
    (x, y) = np.where((X_0_0-X_0_1)+(Y_0_0-Y_0_1) > 0, (X_0_0, Y_0_0), (X_0_1, Y_0_1))
    (x, y) = np.where((x-X_1_0)+(y-Y_1_0) > 0, (x, y), (X_1_0, Y_1_0))
    (x, y) = np.where((x-X_1_1)+(y-Y_1_1) > 0, (x, y), (X_1_1, Y_1_1))
    x = x.transpose(2, 0, 1)
    y = y.transpose(2, 0, 1)
    return x, y


# l=8
# X = np.random.randint(0,2**(l-1), (1, 500, 200, 200))
# Y = np.random.randint(0,2**(l-1), (1, 500, 200, 200))
# sum = X + Y
# x, y = SecMaxPool(X,Y)
# output = x + y

# U_1=np.random.uniform(-2**8,2**8,(1,1,100,100))
# U_2=np.random.uniform(-2**8,2**8,(1,1,100,100))
# t1=time.time()*100
# f_1,f_2=SecMaxPool(U_1, U_2)
# t2=time.time()*100
# print('pool',t2-t1)

# X = np.random.randint(0,2**8, (500, 199, 200))
# Y = MaxPool(X)