import numpy as np
import time
from Secure_Protocols import SComp



# 函数作用：过滤掉proposals中边框长宽太小的框
def Filter_box(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0] #保留宽和高均大于阈值的proposals
    return keep


#安全地过滤掉边框长宽太小的框
def SFilter(boxes_1, boxes_2, min_size):
    ws_1 = boxes_1[:, 2] - boxes_1[:, 0] + 1
    ws_2 = boxes_2[:, 2] - boxes_2[:, 0]
    hs_1 = boxes_1[:, 3] - boxes_1[:, 1] + 1
    hs_2 = boxes_2[:, 3] - boxes_2[:, 1] 
    a_1, a_2 = SComp(ws_1, ws_2, min_size, 0)
    b_1, b_2 = SComp(hs_1, hs_2, min_size, 0)

    # keep = np.where((a_1 + a_2 - 0.5 < 0) & (b_1 + b_2 - 0.5 < 0))[0]
    keep = np.where((a_1 + a_2 == 0) & (b_1 + b_2 == 0))[0]    # 保留宽和高均大于阈值的proposals
    return keep


# sh = (17100, 4)
# U_1=np.random.uniform(-2**8,2**8, sh)
# U_2=np.random.uniform(-2**8,2**8, sh)
# t0 = time.time()*100
# f_ori = Filter_box(U_1 + U_2, 16)
# t1 = time.time()*100
# f = SFilter(U_1, U_2, 16)
# t2 = time.time()*100
# error = f - f_ori

# print('filter',t1-t0)
# print('secfilter',t2-t1)
# print('SAF',t3-t2)

