import numpy as np
import time
from Secure_Protocols import SMax, SMin



#函数作用：处理超过图像边界的bbox，使得pred_boxes位于图片内
def Clip_box(pred_boxes, im_shape):   #im_shape：原图（h，w，3），即（375，500，3）
    # x1 >= 0
    #np.minimum(pred_boxes[:, 0::4], im_shape[1] - 1)=min(x，宽度方向的最大坐标499)
    pred_boxes[:, 0::4] = np.maximum(np.minimum(pred_boxes[:, 0::4], im_shape[1] - 1), 0)  
    # y1 >= 0
    pred_boxes[:, 1::4] = np.maximum(np.minimum(pred_boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1],x1<x2
    pred_boxes[:, 2::4] = np.maximum(np.minimum(pred_boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0],y1<y2
    pred_boxes[:, 3::4] = np.maximum(np.minimum(pred_boxes[:, 3::4], im_shape[0] - 1), 0)
    return pred_boxes


def SClip(pred_boxes_1, pred_boxes_2, im_shape):   #im_shape：原图（h，w，3），即（375，500，3）
    # x1 >= 0
    #np.minimum(pred_boxes[:, 0::4], im_shape[1] - 1)=min(x，宽度方向的最大坐标499)
    #需要使用不定长参数传递的功能，否则嵌套函数调用会报错（传递参数数量不匹配）
    pred_boxes_1[:, 0::4], pred_boxes_2[:, 0::4] = SMax(*(SMin(pred_boxes_1[:, 0::4], pred_boxes_2[:, 0::4], im_shape[1] - 1, 0)), 0, 0)
    # y1 >= 0
    pred_boxes_1[:, 1::4], pred_boxes_2[:, 1::4] = SMax(*(SMin(pred_boxes_1[:, 1::4], pred_boxes_2[:, 1::4], im_shape[0] - 1, 0)), 0, 0)
    # x2 < im_shape[1],x1<x2
    pred_boxes_1[:, 2::4], pred_boxes_2[:, 2::4] = SMax(*(SMin(pred_boxes_1[:, 2::4], pred_boxes_2[:, 2::4], im_shape[1] - 1, 0)), 0, 0)
    # y2 < im_shape[0],y1<y2
    pred_boxes_1[:, 3::4], pred_boxes_2[:, 3::4] = SMax(*(SMin(pred_boxes_1[:, 3::4], pred_boxes_2[:, 3::4], im_shape[0] - 1, 0)), 0, 0)
    return pred_boxes_1, pred_boxes_2


# im_shape = (375, 500, 3)
# range = 10**2
# shape = 10**4
#
# boxes=np.random.uniform(0,range,(shape, 4))
# boxes_1=np.random.uniform(0,range,(shape, 4))
# boxes_2=boxes - boxes_1
#
# t1=time.time()
# y_ori = Clip_box(boxes, im_shape)
# t2=time.time()
# y1, y2 = SecClip(boxes_1, boxes_2, im_shape)
# t3=time.time()
# # y_new = y1 + y2
# z1, z2 = SClip(boxes_1, boxes_2, im_shape)
# t4=time.time()
# # z_new = z1 + z2
# # err1 = y_new - y_ori
# # err2 = z_new - y_ori
#
# print('plain', (t2-t1)*100)
# print('sec', (t3-t2)*100)
# print('s', (t4-t3)*100)












