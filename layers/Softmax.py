import numpy as np
import math
import time
from Secure_Protocols import SExp


# 明文意义上的softmax
def Softmax(a):    #每行---概率
    max_row = np.amax(a, axis=1)     #每行的最大值
    a_re = a-max_row[:, np.newaxis]  #每行的差值(<=0)
    a_exp = np.exp(a_re)             #指数结果,区间在(0, 1]内
    a_exp_sum = np.sum(a_exp, axis=1)#按行求差值指数结果的和
    a_softmax = a_exp/a_exp_sum[:, np.newaxis]  #概率
    return a_softmax


# 密文意义上的softmax（clss_axis = 1）
def SSoftmax(x_1, x_2):             # 每行---概率
    y_1, y_2 = SExp(x_1, x_2)       # (rois_num, cls_num)

    y_sum_1 = np.sum(y_1, axis=1)   # 求每行的和值，(rois_num, )
    y_sum_2 = np.sum(y_2, axis=1)
    y_sum = y_sum_1 + y_sum_2

    out_1 = y_1 / y_sum[:, np.newaxis]
    out_2 = y_2 / y_sum[:, np.newaxis]

    k = 1
    _out_1 = np.random.uniform(-k, k, out_1.shape)  # S1
    s_1 = out_1 - _out_1
    _out_2 = out_2 + s_1  # S2

    return _out_1, _out_2


# sh = (1, 2, 10, 100)
# sh = (100, 20)
# range =10
# x = np.random.uniform(0, range, sh)
# x1 = np.random.uniform(0, range, sh)
# x2 = x - x1
#
# t1=time.time()
# y_ori = Softmax(x)
# t2=time.time()
# y1, y2 = SSoftmax(x1, x2)
# t3=time.time()
# y_new = y1 + y2
# error = y_new - y_ori
#
# print('plain', (t2-t1)*1000)
# print('cipher', (t3-t2)*1000)

# sh = (1, 2, 3)
# x = np.random.uniform(0, 10, sh)
# y = np.sum(x, axis=1)
# z = y[:, np.newaxis]
# a = x/z
# print(x, '\n', y, '\n', z, '\n', a)
# # out_1 = y_1 / y_sum[:, np.newaxis]

