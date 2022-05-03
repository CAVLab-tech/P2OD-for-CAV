
import numpy as np
import random
import math
import time
from Secure_Protocols import SMax, SMin, SComp, SMul


def SRA(u1, u2):
    sh = u1.shape
    r1 = np.random.randint(0, 224, sh)  # @S1
    t1 = u1 - r1
    r2 = np.random.randint(0, 224, sh)  # @S2
    t2 = u2 - r2

    t = t1 + t2    # S1@S2
    d = np.rint(t) - np.floor(t1) - np.floor(t2)     # np.rint

    f1 = np.floor(u1) + d
    f2 = np.floor(u2)
    return f1, f2


def roipooling_vector(feature_data, rois):
    batch, channels, height, width = feature_data.shape    # (1, 512, 19, 63)
    num_rois = rois.shape[0]   # anchors数量300
    pool_height = 7  # pool后的高和宽（pool5）
    pool_width = 7
    spatial_scale = 0.0625  # 尺度系数
    
    roi_start_w = np.rint(rois[:, 1] * spatial_scale)  # x1  #定位出共享卷积层输出的特征图上的映射位置开始坐标
    roi_start_h = np.rint(rois[:, 2] * spatial_scale)  # y1  #四舍五入
    roi_end_w   = np.rint(rois[:, 3] * spatial_scale)  # x2
    roi_end_h   = np.rint(rois[:, 4] * spatial_scale)  # y2
    roi_height  = np.maximum(roi_end_h[:, np.newaxis] - roi_start_h[:, np.newaxis] + 1, 1)  # 特征图上面的高 #(num_rois,)
    roi_width   = np.maximum(roi_end_w[:, np.newaxis] - roi_start_w[:, np.newaxis] + 1, 1)

    bin_size_h = roi_height / pool_height   # 高/7,每个section在高方向上的大小
    bin_size_w = roi_width / pool_width
    ph = np.arange(0, 7)
    pw = np.arange(0, 7)
        
    hstart_grid = np.floor(bin_size_h * ph[np.newaxis, :])       # 向下取整  #(num_rois,) * (1, 7) --> (num_rois,7)
    wstart_grid = np.floor(bin_size_w * pw[np.newaxis, :])
    hend_grid   = np.ceil(bin_size_h * (ph[np.newaxis, :] + 1))  # 向上取整
    wend_grid   = np.ceil(bin_size_w * (pw[np.newaxis, :] + 1))

    hstart = np.minimum(np.maximum(hstart_grid + roi_start_h[:,np.newaxis], 0), height-1)  # 坐标约束，限制其位于[0, 14]
    wstart = np.minimum(np.maximum(wstart_grid + roi_start_w[:,np.newaxis], 0), width-1)   # (num_rois,7)
    hend   = np.minimum(np.maximum(hend_grid + roi_start_h[:,np.newaxis], 0), height)
    wend   = np.minimum(np.maximum(wend_grid + roi_start_w[:,np.newaxis], 0), width)

    hstart=hstart.astype(int)
    wstart=wstart.astype(int)
    hend=hend.astype(int)
    wend=wend.astype(int)

    region = np.zeros((num_rois, pool_height * pool_width, 4))   # 存储每个边界框(锚)的边界坐标表示
    for i in range(pool_height):
        region[:, pool_width *i: pool_width *(i+1), 0] = hstart[:, i][:, np.newaxis]
        region[:, pool_width * i: pool_width * (i + 1), 1] = wstart
        region[:, pool_width *i: pool_width *(i+1), 2] = hend[:, i][:, np.newaxis]
        region[:, pool_width * i: pool_width * (i + 1), 3] = wend
    region_reshape = region.reshape(num_rois * pool_height * pool_width, 4).astype(int)

    out = np.zeros((channels, num_rois * pool_height * pool_width))
    for j in range(num_rois * pool_height * pool_width):
        out[:, j] = np.max((feature_data[0, :, region_reshape[j, 0]:region_reshape[j, 2],
                           region_reshape[j, 1]:region_reshape[j, 3]].reshape((channels, -1))), axis = 1)
    output = out.reshape(channels, num_rois, pool_height, pool_width).transpose(1, 0, 2, 3)
    return output


def SROIP_T(feature_data_1, feature_data_2, rois_1, rois_2):
    batch, channels, height, width = feature_data_1.shape  # (1, 512, 19, 63)
    num_rois = rois_1.shape[0]   #anchors数量300
    pool_height = 7  #pool后的高和宽（pool5）
    pool_width = 7
    spatial_scale = 0.0625  #尺度系数1/16

    roi_start_w_1, roi_start_w_2 = SRA(rois_1[:, 1] * spatial_scale, rois_2[:, 1] * spatial_scale)  # x1  #定位出共享卷积层输出的特征图上的映射位置开始坐标
    roi_start_h_1, roi_start_h_2 = SRA(rois_1[:, 2] * spatial_scale, rois_2[:, 2] * spatial_scale)  # y1
    roi_end_w_1, roi_end_w_2     = SRA(rois_1[:, 3] * spatial_scale, rois_2[:, 3] * spatial_scale)  # x2
    roi_end_h_1, roi_end_h_2     = SRA(rois_1[:, 4] * spatial_scale, rois_2[:, 4] * spatial_scale)  # y2

    roi_height = np.maximum((roi_end_h_1[:,np.newaxis] - roi_start_h_1[:,np.newaxis] + 1) +
                            (roi_end_h_2[:,np.newaxis] - roi_start_h_2[:,np.newaxis]), 1)       # public 特征图上面的高 #(num_rois,)
    roi_width  = np.maximum((roi_end_w_1[:,np.newaxis] - roi_start_w_1[:,np.newaxis] + 1) +
                            (roi_end_w_2[:,np.newaxis] - roi_start_w_2[:,np.newaxis]), 1)

    bin_size_h = roi_height / pool_height   # 高/7,每个section在高方向上的大小(num_rois,)
    bin_size_w = roi_width / pool_width
    ph = np.arange(0,7)    # (7,)
    pw = np.arange(0,7)

    hstart_grid = np.floor(bin_size_h * ph[np.newaxis, :])       # 向下取整  #(num_rois,) * (1, 7) --> (num_rois,7)
    wstart_grid = np.floor(bin_size_w * pw[np.newaxis, :])
    hend_grid   = np.ceil(bin_size_h * (ph[np.newaxis, :] + 1))  # 向上取整
    wend_grid   = np.ceil(bin_size_w * (pw[np.newaxis, :] + 1))

    # 坐标约束，限制其位于[0, 14]
    hstart_one_1, hstart_one_2 = SMax(hstart_grid+roi_start_h_1[:,np.newaxis],    roi_start_h_2[:,np.newaxis], 0, 0)
    wstart_one_1, wstart_one_2 = SMax(wstart_grid + roi_start_w_1[:, np.newaxis], roi_start_w_2[:, np.newaxis], 0, 0)
    hend_one_1, hend_one_2     = SMax(hend_grid + roi_start_h_1[:, np.newaxis],   roi_start_h_2[:, np.newaxis], 0, 0)
    wend_one_1, wend_one_2     = SMax(wend_grid + roi_start_w_1[:, np.newaxis],   roi_start_w_2[:, np.newaxis], 0, 0)

    hstart_1, hstart_2 = SMin(hstart_one_1, hstart_one_2, height-1, 0)    #(num_rois,7)
    wstart_1, wstart_2 = SMin(wstart_one_1, wstart_one_2, width-1, 0)
    hend_1, hend_2     = SMin(hend_one_1,   hend_one_2, height, 0)
    wend_1, wend_2     = SMin(wend_one_1,   wend_one_2, width, 0)

    hstart_1 = hstart_1.astype(int)
    hstart_2 = hstart_2.astype(int)
    wstart_1 = wstart_1.astype(int)
    wstart_2 = wstart_2.astype(int)
    hend_1   = hend_1.astype(int)
    hend_2   = hend_2.astype(int)
    wend_1   = wend_1.astype(int)
    wend_2   = wend_2.astype(int)

    region_1 = np.zeros((num_rois, pool_height * pool_width, 4))   # 存储每个边界框(锚)的边界坐标表示
    region_2 = np.zeros((num_rois, pool_height * pool_width, 4))
    for i in range(pool_height):
        region_1[:, pool_width * i: pool_width * (i + 1), 0] = hstart_1[:, i][:, np.newaxis]
        region_2[:, pool_width * i: pool_width * (i + 1), 0] = hstart_2[:, i][:, np.newaxis]
        region_1[:, pool_width * i: pool_width * (i + 1), 1] = wstart_1
        region_2[:, pool_width * i: pool_width * (i + 1), 1] = wstart_2
        region_1[:, pool_width * i: pool_width * (i + 1), 2] = hend_1[:, i][:, np.newaxis]   # row_same
        region_2[:, pool_width * i: pool_width * (i + 1), 2] = hend_2[:, i][:, np.newaxis]
        region_1[:, pool_width * i: pool_width * (i + 1), 3] = wend_1
        region_2[:, pool_width * i: pool_width * (i + 1), 3] = wend_2

    region_reshape_1 = region_1.reshape(num_rois * pool_height * pool_width, 4).astype(int)
    region_reshape_2 = region_2.reshape(num_rois * pool_height * pool_width, 4).astype(int)

    a_1 = random.uniform(0, 10 ** 3)  # S1
    a_2 = random.uniform(0, 10 ** 3)  # S2
    feature_data_fict_1 = feature_data_1 + a_1  # blind_feature_map
    feature_data_fict_2 = feature_data_2 + a_2
    
    feature_data_fict = feature_data_fict_1 + feature_data_fict_2  # T: 由T对保序密态特征图进行目标区域选择。
    region_reshape = region_reshape_1 + region_reshape_2           # T：明文边界框

    # ROI-pooling-core-computing
    roi_feature_fict   = []   # 动态列表(内嵌numpy数组)
    for j in range(num_rois * pool_height * pool_width):    # 参考池化的索引分块？可是每次循环的执行区域大小不是固定的。。。
        feature_fict = feature_data_fict[0, :, region_reshape[j, 0]:region_reshape[j, 2], region_reshape[j, 1]:region_reshape[j, 3]].reshape((channels, -1))
        roi_feature_fict.append(feature_fict)      # T  （num_rois * pool_height * pool_width，channels，roi_h * roi_w）--> S1 & S2

    out = np.zeros((num_rois * pool_height * pool_width, channels)) # S1 & S2
    for j in range(num_rois * pool_height * pool_width):
        if roi_feature_fict[j].size == 0:
            out[j, :] = 0
        else:
            out[j, :] = np.max(roi_feature_fict[j], axis=1)
    output = out.reshape(num_rois, pool_height, pool_width, channels).transpose(0, 3, 1, 2)  # S1 & S2  # x_1 + x_2 + a_1 + a_2

    output_1 = np.random.uniform(0, 10 ** 3, (num_rois, channels, pool_height, pool_width))
    b = output - (a_1 + output_1)    # x_1 + x_2 - output_1 + a_2
    output_2 = b - a_2               # x_1 + x_2 - output_1
    return output_1, output_2


def SROIP(feature_data_1, feature_data_2, rois_1, rois_2):
    batch, channels, height, width = feature_data_1.shape  # (1, 512, 19, 63)
    num_rois = rois_1.shape[0]  # anchors数量300
    pool_height = 7  # pool后的高和宽（pool5）
    pool_width = 7
    spatial_scale = 0.0625  # 尺度系数1/16

    rois_sca_1 = rois_1 * spatial_scale  # 映射
    rois_sca_2 = rois_2 * spatial_scale

    rois_round_1 = np.floor(rois_sca_1)  # 四舍五入取整
    rois_round_2 = np.rint((rois_sca_1 - np.floor(rois_sca_1)) + rois_sca_2)

    # private, 特征图上每个bounding-box的角坐标，尺寸为(num_rois,)
    roi_start_w_1 = rois_round_1[:, 1][:, np.newaxis]  # x1
    roi_start_w_2 = rois_round_2[:, 1][:, np.newaxis]
    roi_start_h_1 = rois_round_1[:, 2][:, np.newaxis]  # y1
    roi_start_h_2 = rois_round_2[:, 2][:, np.newaxis]
    roi_end_w_1 = rois_round_1[:, 3][:, np.newaxis]  # x2
    roi_end_w_2 = rois_round_2[:, 3][:, np.newaxis]
    roi_end_h_1 = rois_round_1[:, 4][:, np.newaxis]  # y2
    roi_end_h_2 = rois_round_2[:, 4][:, np.newaxis]

    # public, 特征图上每个bounding-box的高和宽，尺寸为(num_rois, 1)
    roi_height = np.maximum((roi_end_h_1 - roi_start_h_1 + 1) + (roi_end_h_2 - roi_start_h_2), 1)
    roi_width = np.maximum((roi_end_w_1 - roi_start_w_1 + 1) + (roi_end_w_2 - roi_start_w_2), 1)

    # public, 每个bounding-box的section尺寸,(高/7, 宽/7)，尺寸为(num_rois, 1)
    bin_size_h = roi_height / pool_height
    bin_size_w = roi_width / pool_width

    # 每个bounding-box池化后的特征图尺寸(7, 7)
    ph = np.arange(0, pool_height)  # (pool_height,)
    pw = np.arange(0, pool_width)  # (pool_width,)

    # public, 具体至第(i,j)个section的量化尺寸，(num_rois,) * (1, 7) -> (num_rois,7)
    hstart_grid = np.floor(bin_size_h * ph[np.newaxis])  # 向下取整 (num_rois, pool_height)
    wstart_grid = np.floor(bin_size_w * pw[np.newaxis])
    hend_grid = np.ceil(bin_size_h * (ph[np.newaxis] + 1))  # 向上取整 (num_rois, width)
    wend_grid = np.ceil(bin_size_w * (pw[np.newaxis] + 1))

    # 将bounding-box约束至特征图尺寸[width, height]内
    # private, section的高或宽(num_rois,7)，每个bounding-box被划分为7×7个section (wstart, hstart, wend, hend)
    hstart_one_1, hstart_one_2 = SMax(hstart_grid + roi_start_h_1, roi_start_h_2, 0, 0)
    wstart_one_1, wstart_one_2 = SMax(wstart_grid + roi_start_w_1, roi_start_w_2, 0, 0)
    hend_one_1, hend_one_2 = SMax(hend_grid + roi_start_h_1, roi_start_h_2, 0, 0)
    wend_one_1, wend_one_2 = SMax(wend_grid + roi_start_w_1, roi_start_w_2, 0, 0)
    hstart_1, hstart_2 = SMin(hstart_one_1, hstart_one_2, height - 1, 0)
    wstart_1, wstart_2 = SMin(wstart_one_1, wstart_one_2, width - 1, 0)
    hend_1, hend_2 = SMin(hend_one_1, hend_one_2, height, 0)
    wend_1, wend_2 = SMin(wend_one_1, wend_one_2, width, 0)

    # 为了实现数组逐元素比较，增维代替循环(for h_j/w_j in h/w)，量化取整可以省略
    hstart_1 = hstart_1[:, np.newaxis]  # (num_rois, pool_height, 1)
    hstart_2 = hstart_2[:, np.newaxis]
    wstart_1 = wstart_1[:, np.newaxis]  # (num_rois, 1, pool_width)
    wstart_2 = wstart_2[:, np.newaxis]
    hend_1 = hend_1[:, np.newaxis]
    hend_2 = hend_2[:, np.newaxis]
    wend_1 = wend_1[:, np.newaxis]
    wend_2 = wend_2[:, np.newaxis]

    # 图像特征图尺寸(height, width)：(24, 64)
    h = np.arange(0, height)[np.newaxis, :, np.newaxis]  # (1, height, 1)
    w = np.arange(0, width)[np.newaxis, :, np.newaxis]  # (1, width, 1)

    ah_1, ah_2 = SComp(h, 0, hstart_1 - 0.01, hstart_2)  # (num_rois, height, pool_height)
    aw_1, aw_2 = SComp(w, 0, wstart_1 - 0.01, wstart_2)  # (num_rois, width, pool_width)
    bh_1, bh_2 = SComp(h, 0, hend_1, hend_2)  # (num_rois, height, pool_height)
    bw_1, bw_2 = SComp(w, 0, wend_1, wend_2)  # (num_rois, width, pool_width)

    # 将每个section的列索引与宽边界、行索引与高边界的位置关系进行比较，并将行(列)索引扩展为二维索引(shares)
    # (num_rois, height/1, width/1, pool_height, pool_width)
    ah_grid_1 = ah_1[:, :, np.newaxis, :, np.newaxis]
    ah_grid_2 = ah_2[:, :, np.newaxis, :, np.newaxis]
    aw_grid_1 = aw_1[:, np.newaxis, :, np.newaxis]
    aw_grid_2 = aw_2[:, np.newaxis, :, np.newaxis]
    bh_grid_1 = bh_1[:, :, np.newaxis, :, np.newaxis]
    bh_grid_2 = bh_2[:, :, np.newaxis, :, np.newaxis]
    bw_grid_1 = bw_1[:, np.newaxis, :, np.newaxis]
    bw_grid_2 = bw_2[:, np.newaxis, :, np.newaxis]

    # if ah=0 & aw=0 & bh=1 & bw=1, then feature is in box, i.e., flag=1; otherwise flag=0.
    # (1 - ah_grid) × (1 - aw_grid) × bh_grid × bw_grid = 1 or 0
    # flag的尺寸为(num_rois, height * width, pool_height, pool_width)
    c_1, c_2 = SMul(1 - ah_grid_1, -ah_grid_2, 1 - aw_grid_1, -aw_grid_2)
    cc_1, cc_2 = SMul(bh_grid_1, bh_grid_2, bw_grid_1, bw_grid_2)
    flag_1, flag_2 = SMul(c_1, c_2, cc_1, cc_2)

    # inf表示极小的值，加上inf的特征不会被max-pool选择为输出表示
    # 若特征不在池化区域内，则flag=0，1-flag=1，则feature_fict = feature_data + inf
    # flag_fic的尺寸为(num_rois * pool_height * pool_width, height * width)
    inf = -100
    flag_fic_1 = ((1 - flag_1) * inf).transpose(0, 3, 4, 1, 2).reshape(-1, height * width)
    flag_fic_2 = (-flag_2 * inf).transpose(0, 3, 4, 1, 2).reshape(-1, height * width)

    # feature_data的尺寸为(channels, height * width)
    feature_data_1 = feature_data_1.reshape(channels, height * width)
    feature_data_2 = feature_data_2.reshape(channels, height * width)

    section_feature = np.zeros((num_rois * pool_height * pool_width, channels))
    for j in range(num_rois * pool_height * pool_width):
        feature_fic_1 = feature_data_1 + flag_fic_1[j, :]  # (channels, height * width)
        feature_fic_2 = feature_data_2 + flag_fic_2[j, :]

        # 运行时间太长，安全求每个box的section内的最大特征值过程省略
        # 方法一：若采用安全全局最大池化，类似于SMP协议；
        # 方法二：若采用一个随机掩码掩盖特征，则不改变最大值位置，采用np.argmax或np.max
        feature_fic_both = feature_fic_1 + feature_fic_2
        section_feature[j, :] = np.max(feature_fic_both, axis=1)  # (channels,)

    # (num_rois, channels, pool_height, pool_width)
    section_feature = section_feature.reshape(num_rois, pool_height, pool_width, channels).transpose(0, 3, 1, 2)
    pool_feature_1 = np.random.uniform(0, 256, (num_rois, channels, pool_height, pool_width))
    pool_feature_2 = (section_feature - pool_feature_1)
    return pool_feature_1, pool_feature_2


# range_1 = 1
# range_2 = 5
# sh_anchor = 10**0
# feature_data = np.random.uniform(-10**range_1, 10**range_1, (1, 1, 10, 1))
# feature_data_1 = np.random.uniform(-10**range_1, 10**range_1, (1, 1, 10, 1))
# feature_data_2 = feature_data - feature_data_1
# rois = np.random.uniform(0, 10**range_2, (sh_anchor, 5))
# rois_1 = np.random.uniform(0, 10**range_2, (sh_anchor, 5))
# rois_2 = rois - rois_1
#
# time0=time.time()
# output_2 = roipooling_vector(feature_data, rois)
# time1=time.time()
# f_1, f_2 = SROIP(feature_data_1, feature_data_2, rois_1, rois_2)
# time2=time.time()
#
# print('roipooling_vector', (time1 - time0) * 1000)
# print('SROIP', (time2 - time1) * 1000)
# error_sec = (f_1 + f_2) - output_2
# error = np.max(np.abs(error_sec))   # 10^(-8)
