import time
import numpy as np
import random
import math


def STMA(u1, u2):   # 乘法 -> 加法转换
    sh = u1.shape
    a1 = np.random.uniform(0, 2 ** 8 - 1, sh)    # T
    a2 = np.random.uniform(0, 2 ** 8 - 1, sh)
    c = a1 * a2
    c1 = np.random.uniform(0, 2 ** 8 - 1, sh)
    c2 = c - c1
    p1 = u1 - a1    # S1
    p2 =u2 - a2     # S2
    f1 = c1 + a1*p2         # S1
    f2 = c2 + a2*p1 +p1*p2  # S2
    return f1, f2


def STAM(u1, u2):  # 加法 -> 乘法转换
    sh = u1.shape
    a1 = np.random.randint(1, 10 ** 3, sh)    # T
    a2 = np.random.randint(1, 10 ** 3, sh)
    c = a1 * a2
    c1 = np.random.randint(1, 10 ** 3, sh)
    c2 = c - c1
    p1 = u1 - a1    # S1
    f2 = random.randint(1, 10 ** 3)   # non-zero   # S2
    lamda1 = (p1 + u2)/f2 -c2
    lamda2 = 1/f2 + a2
    f1 = lamda1 + a1*lamda2 - c1         # S1
    return f1, f2


def SComp(x1, x2, y1, y2):  # 比较函数
    g1 = x1 - y1           # S1
    g2 = x2 - y2           # S2
    a1, a2 = STAM(g1, g2)  # S1 & S2
    b1 = np.where(a1 >= 0, 0, 1)  # S1
    b2 = np.where(a2 >= 0, 0, 1)  # S2  # 符号的算术乘法共享等同于异或共享
    # bit = b_1 ^ b_2 = np.mod((b_1 + b_2), 2)     # S1 & S2
    #     = (1 - b_1) * b_2 + b_1 * (1 - b_2) = b_1 + b_2 - 2 * b_1 *b_2
    c1, c2 = STMA(-2 * b1, b2)  # @S1 & S2
    z1 = b1 + c1  # @S1
    z2 = b2 + c2  # @S2    # 符号的算术加法共享
    return z1, z2     # z = z1 + z2


def SMax(x_1, x_2, y_1, y_2):   # max((x_1+x_2), (y_1+y_2))
    bit_1, bit_2 = SComp(x_1, x_2, y_1, y_2)             # @S1 & S2
    a_1, a_2 = SMul(y_1 - x_1, y_2 - x_2, bit_1, bit_2)  # @S1 & S2
    f_1 = x_1 + a_1      # @S1
    f_2 = x_2 + a_2      # @S2
    return f_1, f_2


def SMin(x_1, x_2, y_1, y_2):   # min((x_1+x_2), (y_1+y_2))
    bit_1, bit_2 = SComp(x_1, x_2, y_1, y_2)  # @S1 & S2
    b_1, b_2 = SMul(x_1 - y_1, x_2 - y_2, bit_1, bit_2)  # @S1 & S2
    f_1 = y_1 + b_1  # @S1
    f_2 = y_2 + b_2  # @S2
    return f_1, f_2


def SMul(x_1, x_2, y_1, y_2):  #安全十进制乘法函数, f_1+f_2=(x_1+x_2).* (y_1+y_2)=x.*y
    sh=x_1.shape
    a_1 = np.random.randint(0, 10 ** 3, sh)    # T
    a_2 = np.random.randint(0, 10 ** 3, sh)
    b_1 = np.random.randint(0, 10 ** 3, sh)
    b_2 = np.random.randint(0, 10 ** 3, sh)
    c = (a_1 + a_2) * (b_1 + b_2)
    c_1 = np.random.randint(0, 10 ** 3, sh)
    c_2 = c - c_1

    alpha = (x_1 - a_1) + (x_2 - a_2)      # S1 & S2
    beta = (y_1 - b_1) + (y_2 - b_2)
    f_1 = c_1 + b_1 * alpha + a_1 * beta   # S1
    f_2 = c_2 + b_2 * alpha + a_2 * beta + alpha * beta   # S2
    return f_1, f_2


# def SExp_gcd(x1, x2):  # 指数函数
#     r = 5
#     (b1, b2) = np.where((x1 < -r) & (x2 > r), (x1 + np.rint(x2 / r) * r,
#                         x2 - np.rint(x2 / r) * r), (x1, x2))     # S2传递倍数给S1
#     (f1, f2) = np.where((b1 > r) & (b2 < -r), (b1 - np.rint(b1 / r) * r,
#                         b2 + np.rint(b1 / r) * r), (b1, b2))     # S1传递倍数给S2
#     z1, z2 = STMA(np.exp(f1), np.exp(f2))    # S1 & S2
#     return z1, z2

def SExp(x1, x2):  # 指数函数
    k = 5
    _x1 = np.random.uniform(-k, k, x1.shape)   # S1  if x1 <= -k & x1 >= k:
    s1 = x1 - _x1
    _x2 = x2 + s1     # S2

    z1, z2 = STMA(np.exp(_x1), np.exp(_x2))    # S1 & S2
    return z1, z2


# sh = 10**4
# U_1 = np.random.uniform(-2**8, 2**8, sh)
# U_2 = np.random.uniform(-2**8, 2**8, sh)
# V_1 = np.random.uniform(-2**8, 2**8, sh)
# V_2 = np.random.uniform(-2**8, 2**8, sh)
#
# f_1, f_2 = SComp(U_1, U_2, V_1, V_2)
# f = np.where((U_1 + U_2) < (V_1 + V_2), 1, 0)
# error_bit = (f_1 + f_2) - f
#
# g_1, g_2 = SMax(U_1, U_2, V_1, V_2)
# g = np.where((U_1 + U_2) < (V_1 + V_2), (V_1 + V_2), (U_1 + U_2))
# error_max = (g_1 + g_2) - g

# sh = 10**4
# range =10
# x = np.random.uniform(0, range, sh)
# x1 = np.random.uniform(0, range, sh)
# x2 = x -x1
# t1=time.time()
# y_ori = np.exp(x)
# t2=time.time()
# y1, y2 = SExp_gcd(x1, x2)
# t3=time.time()
# y_new = y1 + y2
# error = y_new - y_ori
#
# print('plain', (t2-t1)*1000)
# print('cipher', (t3-t2)*1000)
