import time
import numpy as np
from Secure_Protocols import SComp, SMul


def ReLU(input):    #明文ReLU
    bit = np.where(input >= 0, 1, 0)
    output = input * bit
    return output


def SReLU(input_1, input_2):     #密文ReLU
    sign_1, sign_2 = SComp(input_1, input_2, 0, 0)  # @S1 & S2
    output_1, output_2 = SMul(input_1, input_2, 1 - sign_1, -sign_2)  # @S1 & S2
    return output_1, output_2


# ran = 4
# sh =10**5
# U_1=np.random.uniform(-10**ran, 10**ran, sh)
# U_2=np.random.uniform(-10**ran, 10**ran, sh)
# t0 = time.time()
# f_ori = ReLU(U_1 + U_2)
# t1 = time.time()
# f_1,f_2=SReLU(U_1, U_2)
# t2=time.time()
#
# g_ori = np.where(U_1 + U_2 >= 0, 0, 1)
# t3=time.time()
# g_1, g_2 = SComp(U_1, U_2, 0, 0)
# t4=time.time()
#
# print('relu', (t1-t0)*1000)
# print('srelu', (t2-t1)*1000)
# print('comp', (t3-t2)*1000)
# print('scomp', (t4-t3)*1000)
#
# f = f_1 + f_2
# error_relu = f - f_ori
# g = g_1 + g_2
# error_comp = g - g_ori


