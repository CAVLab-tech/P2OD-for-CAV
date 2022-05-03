import numpy as np
import time
from Secure_Protocols import STMA


# 函数作用:得到改善后的anchor的信息(xmin, ymin, xmax, ymax)，根据anchor和偏移量计算预测proposals
def Transform_box(boxes, deltas):  #boxes:anchor(300,4)，box_deltas:偏移量(300,84)
    boxes = boxes.astype(deltas.dtype, copy=False)  #数据类型转换，float32,(xmin, ymin, xmax, ymax)
    #将anchor还原为（x_center,y_center,w,h）的格式，各个分量的维度为(300,)
    widths = boxes[:, 2] - boxes[:, 0] + 1.0   #xmax-xmin+1, 所有boxes的宽度
    heights = boxes[:, 3] - boxes[:, 1] + 1.0  # ymax - ymin + 1, 所有boxes的高度
    ctr_x = boxes[:, 0] + 0.5 * widths         #宽的中心点,x_center
    ctr_y = boxes[:, 1] + 0.5 * heights        #高的中心点,y_center
    
    # 得到（x,y,w,h）方向上的偏移量,每个类别的偏移量不同，基于同一个anchor所产生的pridict bbox也就不同
    dx = deltas[:, 0::4]  #维度为(300, 21),按类别分离bbox，取每一行中0, 4, 8, ..., 位置的数据
    dy = deltas[:, 1::4]  # 取每一行中1, 5, 9, ..., 位置的数据
    dw = deltas[:, 2::4]  # 取每一行中2, 6, 10, ..., 位置的数据
    dh = deltas[:, 3::4]  # 取每一行中3, 7, 11, ..., 位置的数据
    
    #得到预测proposals的（x_center,y_center,w,h）
    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]  #x方向平移变换,(300,21)*(300,1)+(300,1)=(300,21)
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis] #y方向平移变换
    pred_w = np.exp(dw) * widths[:, np.newaxis]   #w的缩放,P4
    pred_h = np.exp(dh) * heights[:, np.newaxis]  #h的缩放

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)  #(300,84)
    #得到预测proposals的（x1,y1,x2,y2）
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w  #x1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h  #y1
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w  #x2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h  #y2
    return pred_boxes


# 安全的anchor变换（根据偏移deltas）
def STrans(boxes, deltas_1, deltas_2):   # boxes:anchor(n,4)，box_deltas:偏移量(n,4*K)
    # 将anchor还原为（x_center,y_center,w,h）的格式，各个分量的维度为(n,)
    boxes = boxes.astype(deltas_1.dtype, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + 1.0         # xmax-xmin+1, 所有boxes的宽度
    heights = boxes[:, 3] - boxes[:, 1] + 1.0        # ymax - ymin + 1, 所有boxes的高度
    ctr_x = boxes[:, 0] + 0.5 * widths               # 宽的中心点,x_center
    ctr_y = boxes[:, 1] + 0.5 * heights              # 高的中心点,y_center

    # 得到（x,y,w,h）方向上的偏移量,每个类别的偏移量不同，基于同一个anchor所产生的pridict bbox也就不同
    dx_1 = deltas_1[:, 0::4]  # 维度为(n, K),按类别分离bbox，取每一行中0, 4, 8, ..., 位置的数据
    dx_2 = deltas_2[:, 0::4]
    dy_1 = deltas_1[:, 1::4]  # 取每一行中1, 5, 9, ..., 位置的数据
    dy_2 = deltas_2[:, 1::4]
    dw_1 = deltas_1[:, 2::4]  # 取每一行中2, 6, 10, ..., 位置的数据
    dw_2 = deltas_2[:, 2::4]
    dh_1 = deltas_1[:, 3::4]  # 取每一行中3, 7, 11, ..., 位置的数据    
    dh_2 = deltas_2[:, 3::4]

    # 得到预测proposals的（x_center,y_center,w,h）
    pred_ctr_x_1 = dx_1 * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_x_2 = dx_2 * widths[:, np.newaxis]    # x方向平移变换,(300,21)*(300,1)+(300,1)=(300,21)

    pred_ctr_y_1 = dy_1 * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_ctr_y_2 = dy_2 * heights[:, np.newaxis]   # y方向平移变换

    # print(np.max(np.abs(out_1)), np.max(np.abs(out_2)), np.max(np.abs(out_1 + out_2)))
    # print(np.max(np.abs(_out_1)), np.max(np.abs(_out_2)), np.max(np.abs(_out_1 + _out_2)))

    pred_w_1, pred_w_2 = STMA(widths[:, np.newaxis] * np.exp(dw_1), np.exp(dw_2))   # w的尺度变换
    pred_h_1, pred_h_2 = STMA(heights[:, np.newaxis] * np.exp(dh_1), np.exp(dh_2))  # h的尺度变换

    # 得到预测proposals的（x1,y1,x2,y2）
    pred_boxes_1 = np.zeros(deltas_1.shape, dtype=deltas_1.dtype)  # (n,4*K)
    pred_boxes_2 = np.zeros(deltas_1.shape, dtype=deltas_1.dtype)
    pred_boxes_1[:, 0::4] = pred_ctr_x_1 - 0.5 * pred_w_1  # x1
    pred_boxes_2[:, 0::4] = pred_ctr_x_2 - 0.5 * pred_w_2
    pred_boxes_1[:, 1::4] = pred_ctr_y_1 - 0.5 * pred_h_1  # y1
    pred_boxes_2[:, 1::4] = pred_ctr_y_2 - 0.5 * pred_h_2
    pred_boxes_1[:, 2::4] = pred_ctr_x_1 + 0.5 * pred_w_1  # x2
    pred_boxes_2[:, 2::4] = pred_ctr_x_2 + 0.5 * pred_w_2
    pred_boxes_1[:, 3::4] = pred_ctr_y_1 + 0.5 * pred_h_1  # y2
    pred_boxes_2[:, 3::4] = pred_ctr_y_2 + 0.5 * pred_h_2
    return pred_boxes_1, pred_boxes_2


# shape = 10**3
# ran = 10**3
# boxes=np.random.uniform(-ran,ran,(shape, 4))
# deltas=np.random.uniform(-10**0,10**0,(shape, 84))
# deltas_1=np.random.uniform(-10**0,10**0,(shape, 84))
# deltas_2 = deltas - deltas_1
#
# t0=time.time()
# pred_boxes = Transform_box(boxes, deltas)
# t1=time.time()
# pred_boxes_1, pred_boxes_2 = STrans(boxes, deltas_1, deltas_2)
# t2=time.time()
#
# pred_boxes_new = pred_boxes_1 + pred_boxes_2
# error_sat = pred_boxes_new - pred_boxes
#
# print('at', (t1-t0)*1000)
# print('sat', (t2-t1)*1000)

