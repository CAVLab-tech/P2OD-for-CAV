
import numpy as np
import os, cv2
import sys
import time
import pickle as pk

from layers.Conv import mul_weight, add_bias
from layers.ReLU import SReLU
from layers.Max_Pooling import SecMaxPool
from layers.Softmax import SSoftmax
from layers.Proposal_box import SProposal
from layers.ROI_Pooling import SROIP_T, SROIP


model_dir = 'model/'                                   # 模型文件根路径
f_r = open(model_dir + 'faster_rcnn_model.pkl', 'rb')  # 建立文件对象
model = pk.load(f_r)                                   # 加载pkl文件


def faster_rcnn_sec(image_one, image_two, im_scale):
    # t0 = time.time()

    # 卷积模块1
    conv_weight_1_1_one = mul_weight(image_one, model['conv1_1_w'])   # 缺省pad = 1
    conv_1_1_one = add_bias(conv_weight_1_1_one, model['conv1_1_b'])
    conv_1_1_two = mul_weight(image_two, model['conv1_1_w'])
    relu_1_1_one, relu_1_1_two = SReLU(conv_1_1_one, conv_1_1_two)

    conv_weight_1_2_one = mul_weight(relu_1_1_one, model['conv1_2_w'])   # 缺省pad = 1
    conv_1_2_one = add_bias(conv_weight_1_2_one, model['conv1_2_b'])
    conv_1_2_two = mul_weight(relu_1_1_two, model['conv1_2_w'])
    relu_1_2_one, relu_1_2_two = SReLU(conv_1_2_one, conv_1_2_two)

    pool_1_one, pool_1_two = SecMaxPool(relu_1_2_one, relu_1_2_two)

    # 卷积模块2
    conv_weight_2_1_one = mul_weight(pool_1_one, model['conv2_1_w'])
    conv_2_1_one = add_bias(conv_weight_2_1_one, model['conv2_1_b'])
    conv_2_1_two = mul_weight(pool_1_two, model['conv2_1_w'])
    relu_2_1_one, relu_2_1_two = SReLU(conv_2_1_one, conv_2_1_two)

    conv_weight_2_2_one = mul_weight(relu_2_1_one, model['conv2_2_w'])
    conv_2_2_one = add_bias(conv_weight_2_2_one, model['conv2_2_b'])
    conv_2_2_two = mul_weight(relu_2_1_two, model['conv2_2_w'])
    relu_2_2_one, relu_2_2_two = SReLU(conv_2_2_one, conv_2_2_two)

    pool_2_one, pool_2_two = SecMaxPool(relu_2_2_one, relu_2_2_two)

    # 卷积模块3
    conv_weight_3_1_one = mul_weight(pool_2_one, model['conv3_1_w'])
    conv_3_1_one = add_bias(conv_weight_3_1_one, model['conv3_1_b'])
    conv_3_1_two = mul_weight(pool_2_two, model['conv3_1_w'])
    relu_3_1_one, relu_3_1_two = SReLU(conv_3_1_one, conv_3_1_two)

    conv_weight_3_2_one = mul_weight(relu_3_1_one, model['conv3_2_w'])
    conv_3_2_one = add_bias(conv_weight_3_2_one, model['conv3_2_b'])
    conv_3_2_two = mul_weight(relu_3_1_two, model['conv3_2_w'])
    relu_3_2_one, relu_3_2_two = SReLU(conv_3_2_one, conv_3_2_two)

    conv_weight_3_3_one = mul_weight(relu_3_2_one, model['conv3_3_w'])
    conv_3_3_one = add_bias(conv_weight_3_3_one, model['conv3_3_b'])
    conv_3_3_two = mul_weight(relu_3_2_two, model['conv3_3_w'])
    relu_3_3_one, relu_3_3_two = SReLU(conv_3_3_one, conv_3_3_two)

    pool_3_one, pool_3_two = SecMaxPool(relu_3_3_one, relu_3_3_two)

    # 卷积模块4
    conv_weight_4_1_one = mul_weight(pool_3_one, model['conv4_1_w'])
    conv_4_1_one = add_bias(conv_weight_4_1_one, model['conv4_1_b'])
    conv_4_1_two = mul_weight(pool_3_two, model['conv4_1_w'])
    relu_4_1_one, relu_4_1_two = SReLU(conv_4_1_one, conv_4_1_two)

    conv_weight_4_2_one = mul_weight(relu_4_1_one, model['conv4_2_w'])
    conv_4_2_one = add_bias(conv_weight_4_2_one, model['conv4_2_b'])
    conv_4_2_two = mul_weight(relu_4_1_two, model['conv4_2_w'])
    relu_4_2_one, relu_4_2_two = SReLU(conv_4_2_one, conv_4_2_two)

    conv_weight_4_3_one = mul_weight(relu_4_2_one, model['conv4_3_w'])
    conv_4_3_one = add_bias(conv_weight_4_3_one, model['conv4_3_b'])
    conv_4_3_two = mul_weight(relu_4_2_two, model['conv4_3_w'])
    relu_4_3_one, relu_4_3_two = SReLU(conv_4_3_one, conv_4_3_two)

    pool_4_one, pool_4_two = SecMaxPool(relu_4_3_one, relu_4_3_two)

    # 卷积模块5
    conv_weight_5_1_one = mul_weight(pool_4_one, model['conv5_1_w'])
    conv_5_1_one = add_bias(conv_weight_5_1_one, model['conv5_1_b'])
    conv_5_1_two = mul_weight(pool_4_two, model['conv5_1_w'])
    relu_5_1_one, relu_5_1_two = SReLU(conv_5_1_one, conv_5_1_two)

    conv_weight_5_2_one = mul_weight(relu_5_1_one, model['conv5_2_w'])
    conv_5_2_one = add_bias(conv_weight_5_2_one, model['conv5_2_b'])
    conv_5_2_two = mul_weight(relu_5_1_two, model['conv5_2_w'])
    relu_5_2_one, relu_5_2_two = SReLU(conv_5_2_one, conv_5_2_two)

    conv_weight_5_3_one = mul_weight(relu_5_2_one, model['conv5_3_w'])
    conv_5_3_one = add_bias(conv_weight_5_3_one, model['conv5_3_b'])
    conv_5_3_two = mul_weight(relu_5_2_two, model['conv5_3_w'])
    relu_5_3_one, relu_5_3_two = SReLU(conv_5_3_one, conv_5_3_two)

    # t1 = time.time()

    # RPN网络
    rpn_conv_weight_one = mul_weight(relu_5_3_one, model['rpn_conv_w'])
    rpn_conv_one = add_bias(rpn_conv_weight_one, model['rpn_conv_b'])
    rpn_conv_two = mul_weight(relu_5_3_two, model['rpn_conv_w'])
    rpn_relu_one, rpn_relu_two = SReLU(rpn_conv_one, rpn_conv_two)

    rpn_bbox_weight_one = mul_weight(rpn_relu_one, model['rpn_bbox_w'])
    rpn_bbox_one = add_bias(rpn_bbox_weight_one, model['rpn_bbox_b'])
    rpn_bbox_two = mul_weight(rpn_relu_two, model['rpn_bbox_w'])
    rpn_bbox_one = rpn_bbox_one[np.newaxis, :]    # (channels，height, width) -->(batch, channels，height, width)
    rpn_bbox_two = rpn_bbox_two[np.newaxis, :]

    rpn_score_weight_one = mul_weight(rpn_relu_one, model['rpn_score_w'])
    rpn_score_one = add_bias(rpn_score_weight_one, model['rpn_score_b'])
    rpn_score_two = mul_weight(rpn_relu_two, model['rpn_score_w'])
    rpn_score_one = rpn_score_one[np.newaxis, :]    # (channels，height, width) -->(batch, channels，height, width)
    rpn_score_two = rpn_score_two[np.newaxis, :]

    rpn_score_one_reshape = rpn_score_one.reshape(rpn_score_one.shape[0], 2, 9 * rpn_score_one.shape[2], rpn_score_one.shape[3])
    rpn_score_two_reshape = rpn_score_two.reshape(rpn_score_two.shape[0], 2, 9 * rpn_score_two.shape[2], rpn_score_two.shape[3])

    rpn_prob_one_reshape, rpn_prob_two_reshape = SSoftmax(rpn_score_one_reshape, rpn_score_two_reshape)
    rpn_prob_one = rpn_prob_one_reshape.reshape(rpn_score_one.shape)
    rpn_prob_two = rpn_prob_two_reshape.reshape(rpn_score_two.shape)

    rois_one, rois_two = SProposal(rpn_prob_one, rpn_prob_two, rpn_bbox_one, rpn_bbox_two,
                                   im_info=np.array([image_one.shape[1], image_one.shape[2], im_scale]))

    # t2 = time.time()

    pool_5_one, pool_5_two = SROIP_T(relu_5_3_one[np.newaxis, :], relu_5_3_two[np.newaxis, :], rois_one, rois_two)    # （300，512，7，7）

    # t3 = time.time()

    # 全连接模块1
    pool5_one_straight = pool_5_one.reshape(pool_5_one.shape[0], pool_5_one.shape[1] * pool_5_one.shape[2] * pool_5_one.shape[3])  # （300，25088）
    pool5_two_straight = pool_5_two.reshape(pool_5_two.shape[0], pool_5_two.shape[1] * pool_5_two.shape[2] * pool_5_two.shape[3])

    fc_6_one = np.dot(pool5_one_straight, model['fc6_w'].transpose()) + model['fc6_b'].transpose()  # （300，4096）
    fc_6_two = np.dot(pool5_two_straight, model['fc6_w'].transpose())
    relu_6_one, relu_6_two = SReLU(fc_6_one, fc_6_two)

    # 全连接模块2
    fc_7_one = np.dot(relu_6_one, model['fc7_w'].transpose()) + model['fc7_b'].transpose()   #（300，4096）
    fc_7_two = np.dot(relu_6_two, model['fc7_w'].transpose())
    relu_7_one, relu_7_two = SReLU(fc_7_one, fc_7_two)

    # 网络分类和回归输出
    boxes_deltas_one = np.dot(relu_7_one, model['bbox_w'].transpose()) + model['bbox_b'].transpose()     # （300，class_num * 4）; class_num = 9
    boxes_deltas_two = np.dot(relu_7_two, model['bbox_w'].transpose())

    score_one = np.dot(relu_7_one, model['score_w'].transpose()) + model['score_b'].transpose()  # （300，class_num）
    score_two = np.dot(relu_7_two, model['score_w'].transpose())

    prob_one, prob_two = SSoftmax(score_one, score_two)

    # t4 = time.time()
    # print('1', t1-t0)
    # print('2', t2 - t1)
    # print('3', t3 - t2)
    # print('4', t4 - t3)

    return boxes_deltas_one, boxes_deltas_two, prob_one, prob_two, rois_one, rois_two
