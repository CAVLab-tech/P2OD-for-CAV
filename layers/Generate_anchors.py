import numpy as np
import time




    #array([[ -83.,  -39.,  100.,   56.],   # 输出9个锚(x1, y1, x2, y2)
    #       [-175.,  -87.,  192.,  104.],
    #       [-359., -183.,  376.,  200.],
    #       [ -55.,  -55.,   72.,   72.],
    #       [-119., -119.,  136.,  136.],
    #       [-247., -247.,  264.,  264.],
    #       [ -35.,  -79.,   52.,   96.],
    #       [ -79., -167.,   96.,  184.],
    #       [-167., -343.,  184.,  360.]])
    # base_size：映射倍数，与池化层数量有关；
    # rations：长宽比[0.5, 1, 2]；
    # scales：面积尺度[8, 16, 32]；
def generate_anchors(base_size=16, ratios=[0.5, 1, 2], scales=2**np.arange(3, 6)):
    base_anchor = np.array([1, 1, base_size, base_size]) - 1    # 基准坐标(0, 0, 15, 15)
    
    
    ratio_anchors = _ratio_enum(base_anchor, ratios)    #(3, 4)
    # [[-3.5  2. 18.5 13.]
    #  [0.    0. 15.  15.]
    #  [2.5 - 3. 12.5 18.]]
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in range(ratio_anchors.shape[0])])   # i=1, 2, 3
    # 将不同纵横比的anchor，进行不同尺度变换，并将结果沿竖直(按行顺序)方法把数组给堆叠起来
    return anchors

def _whctrs(anchor):    # Return width, height, x center, and y center for an anchor (window).
    #anchor表示为(x1,y1,x2,y2)
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr

def _mkanchors(ws, hs, x_ctr, y_ctr):    #返回anchors窗口的集合，
    ws = ws[:, np.newaxis]   #(3, 1)
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors

def _ratio_enum(anchor, ratios):
    # Enumerate a set of anchors for each aspect ratio wrt an anchor.
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def _scale_enum(anchor, scales):   # Enumerate a set of anchors for each scale wrt an anchor.
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors



# ttt = time.time()
# anchors = generate_anchors()  #生成anchors，（9,4）
# tt = time.time()
# print('gen',(tt - ttt)*100)