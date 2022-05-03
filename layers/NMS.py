import random
import numpy as np
import time
from Secure_Protocols import SMax, SMin, SComp


'''原始的非极大值抑制函数'''
def NMS(dets, thresh):  #dets=[x1,y1,x2,y2,scores],thresh=NMS_THRESH = 0.3,generally equals 0.3~0.5;
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]   #(x1,y1,x2,y2)检测框左下角和右上角坐标
    scores = dets[:, 4]  #属于该检测框的概率分数

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  #检测框的面积
    order = scores.argsort()[::-1]  #将检测框按照score分数降序排序，输出的是bbox的索引;默认升序
    keep = []   #NMS后,保留的bbox的索引
    while order.size > 0:
        i = order[0]   #最大score的索引
        keep.append(i) #将最大score的索引添加入keep中,重复过程，直至所有bbox被选择或丢弃
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])  #左下角坐标，取较右的x，较上的y
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])  #右上角坐标，取较左的x，较下的y
        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)   #max(0,相交部分)
        inter = w * h   #（重叠）相交部分面积

        ovr = inter / (areas[i] + areas[order[1:]] - inter)  #计算IoU：重叠面积/（面积1+面积2-重叠面积）
        #inds保留的是ovr的索引，即order的索引-1；其中元素是bbox的索引
        inds = np.where(ovr <= thresh)[0]  #保留IoU小于阈值thresh=NMS_THRESH的bbox
        order = order[inds + 1]  #降序的索引；+1的原因是order[0]不在丢弃范围内，ovr数组的长度比order数组少一位
    return keep


'''安全的非极大值抑制函数'''
def SNMS(dets_1, dets_2, thresh):  # dets=[x1,y1,x2,y2,scores],thresh=NMS_THRESH = 0.3,generally equals 0.3~0.5;
    x1_1 = dets_1[:, 0]
    y1_1 = dets_1[:, 1]
    x2_1 = dets_1[:, 2]
    y2_1 = dets_1[:, 3]   # (x1,y1,x2,y2)检测框左下角和右上角坐标
    x1_2 = dets_2[:, 0]
    y1_2 = dets_2[:, 1]
    x2_2 = dets_2[:, 2]
    y2_2 = dets_2[:, 3]
    areas = ((x2_1 - x1_1) + (x2_2 - x1_2) + 1) * ((y2_1 - y1_1) + (y2_2 - y1_2) + 1)  #检测框的面积

    m_1 = random.uniform(0, 1)
    m_2 = random.uniform(0, 1)
    scores_1 = dets_1[:, 4] - m_1  # 属于该检测框的概率分数
    scores_2 = dets_2[:, 4] - m_2
    scores_fic = scores_1 + scores_2  # score - m
    order = scores_fic.argsort()[::-1]   # 将检测框按照score分数降序排序，输出的是bbox的索引

    keep = []   # NMS后,保留的bbox的索引
    while order.size > 0:
        i = order[0]    # 最大score的索引
        keep.append(i)  # 将最大score的索引添加入keep中,重复过程，直至所有bbox被选择或丢弃

        # 相交矩形部分的左下角和右上角坐标
        xx1_1, xx1_2 = SMax(x1_1[i], x1_2[i], x1_1[order[1:]], x1_2[order[1:]])
        yy1_1, yy1_2 = SMax(y1_1[i], y1_2[i], y1_1[order[1:]], y1_2[order[1:]])
        xx2_1, xx2_2 = SMin(x2_1[i], x2_2[i], x2_1[order[1:]], x2_2[order[1:]])
        yy2_1, yy2_2 = SMin(y2_1[i], y2_2[i], y2_1[order[1:]], y2_2[order[1:]])

        w = np.maximum(0.0, (xx2_1-xx1_1)+(xx2_2-xx1_2)+1)
        h = np.maximum(0.0, (yy2_1-yy1_1)+(yy2_2-yy1_2)+1)  # max(0,相交部分)
        inter = w * h   #（重叠）相交部分面积
        ovr = inter / (areas[i] + areas[order[1:]] - inter)  # 计算IoU：重叠面积/（面积1+面积2-重叠面积）
        # inds保留的是ovr的索引，即order的索引-1；其中元素是bbox的索引
        inds = np.where(ovr <= thresh)[0]  # 保留IoU小于阈值thresh=NMS_THRESH的bbox
        order = order[inds + 1]  # 降序的索引；+1的原因是order[0]不在丢弃范围内，ovr数组的长度比order数组少一位
    return keep


# range = 10**2
# sh = 10**3
# dets = np.random.uniform(0, range,(sh, 5))
# dets_1 = np.random.uniform(0, range,(sh,5))
# dets_2 = dets - dets_1
#
# thresh=0.7
# t1 = time.time()
# keep_ori = NMS(dets, thresh)
# t2 = time.time()
# keep_new = SNMS(dets_1, dets_2, thresh)
# t3 = time.time()
#
# error_snms = np.array(keep_new) - np.array(keep_ori)
# print('nms', (t2-t1)*1000)
# print('snms', (t3-t2)*1000)
