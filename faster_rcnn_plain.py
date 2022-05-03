
import numpy as np
import os, cv2
import sys
import time
import pickle as pk

from layers.Conv import mul_weight, add_bias
from layers.ReLU import ReLU
from layers.Max_Pooling import MaxPool
from layers.Softmax import Softmax
from layers.Proposal_box import Proposal
from layers.ROI_Pooling import roipooling_vector


model_dir = 'model/'                       # 模型文件根路径
f_r = open(model_dir + 'faster_rcnn_model.pkl', 'rb')  # 建立文件对象
model = pk.load(f_r)                       # 加载pkl文件


def faster_rcnn_plain(image, im_scale):
    # 卷积模块1
    # t0 = time.time()
    conv_weight_1_1 = mul_weight(image, model['conv1_1_w'])   # 缺省pad = 1
    conv_1_1 = add_bias(conv_weight_1_1, model['conv1_1_b'])
    relu_1_1 = ReLU(conv_1_1)
    conv_weight_1_2 = mul_weight(relu_1_1, model['conv1_2_w'])
    conv_1_2 = add_bias(conv_weight_1_2, model['conv1_2_b'])
    relu_1_2 = ReLU(conv_1_2)
    pool_1 = MaxPool(relu_1_2)

    # 卷积模块2
    conv_weight_2_1 = mul_weight(pool_1, model['conv2_1_w'])
    conv_2_1 = add_bias(conv_weight_2_1, model['conv2_1_b'])
    relu_2_1 = ReLU(conv_2_1)
    conv_weight_2_2 = mul_weight(relu_2_1, model['conv2_2_w'])
    conv_2_2 = add_bias(conv_weight_2_2, model['conv2_2_b'])
    relu_2_2 = ReLU(conv_2_2)
    pool_2 = MaxPool(relu_2_2)

    # 卷积模块3
    conv_weight_3_1 = mul_weight(pool_2, model['conv3_1_w'])
    conv_3_1 = add_bias(conv_weight_3_1, model['conv3_1_b'])
    relu_3_1 = ReLU(conv_3_1)
    conv_weight_3_2 = mul_weight(relu_3_1, model['conv3_2_w'])
    conv_3_2 = add_bias(conv_weight_3_2, model['conv3_2_b'])
    relu_3_2 = ReLU(conv_3_2)
    conv_weight_3_3 = mul_weight(relu_3_2, model['conv3_3_w'])
    conv_3_3 = add_bias(conv_weight_3_3, model['conv3_3_b'])
    relu_3_3 = ReLU(conv_3_3)
    pool_3 = MaxPool(relu_3_3)

    # 卷积模块4
    conv_weight_4_1 = mul_weight(pool_3, model['conv4_1_w'])
    conv_4_1 = add_bias(conv_weight_4_1, model['conv4_1_b'])
    relu_4_1 = ReLU(conv_4_1)
    conv_weight_4_2 = mul_weight(relu_4_1, model['conv4_2_w'])
    conv_4_2 = add_bias(conv_weight_4_2, model['conv4_2_b'])
    relu_4_2 = ReLU(conv_4_2)
    conv_weight_4_3 = mul_weight(relu_4_2, model['conv4_3_w'])
    conv_4_3 = add_bias(conv_weight_4_3, model['conv4_3_b'])
    relu_4_3 = ReLU(conv_4_3)
    pool_4 = MaxPool(relu_4_3)

    # 卷积模块5
    conv_weight_5_1 = mul_weight(pool_4, model['conv5_1_w'])
    conv_5_1 = add_bias(conv_weight_5_1, model['conv5_1_b'])
    relu_5_1 = ReLU(conv_5_1)
    conv_weight_5_2 = mul_weight(relu_5_1, model['conv5_2_w'])
    conv_5_2 = add_bias(conv_weight_5_2, model['conv5_2_b'])
    relu_5_2 = ReLU(conv_5_2)
    conv_weight_5_3 = mul_weight(relu_5_2, model['conv5_3_w'])
    conv_5_3 = add_bias(conv_weight_5_3, model['conv5_3_b'])
    relu_5_3 = ReLU(conv_5_3)

    # t1 = time.time()

    # RPN网络
    rpn_conv_weight = mul_weight(relu_5_3, model['rpn_conv_w'])
    rpn_conv = add_bias(rpn_conv_weight, model['rpn_conv_b'])
    rpn_relu = ReLU(rpn_conv)

    rpn_bbox_weight = mul_weight(rpn_relu, model['rpn_bbox_w'])
    rpn_bbox = add_bias(rpn_bbox_weight, model['rpn_bbox_b'])
    rpn_bbox = rpn_bbox[np.newaxis, :]        # (channels，height, width) -->(batch, channels，height, width)

    rpn_score_weight = mul_weight(rpn_relu, model['rpn_score_w'])
    rpn_score = add_bias(rpn_score_weight, model['rpn_score_b'])
    rpn_score = rpn_score[np.newaxis, :]     # (channels，height, width) -->(batch, channels，height, width)
    rpn_score_reshape = rpn_score.reshape(rpn_score.shape[0], 2, 9*rpn_score.shape[2], rpn_score.shape[3])   # Softmax
    rpn_prob_reshape = Softmax(rpn_score_reshape)
    rpn_prob = rpn_prob_reshape.reshape(rpn_score.shape)

    rois = Proposal(rpn_prob, rpn_bbox, im_info=np.array([image.shape[1], image.shape[2], im_scale]))

    # t2 = time.time()

    pool_5 = roipooling_vector(relu_5_3[np.newaxis, :], rois)   # （300，512，7，7）

    # t3 = time.time()

    # 全连接模块1
    pool5_straight = pool_5.reshape(pool_5.shape[0], pool_5.shape[1] * pool_5.shape[2] * pool_5.shape[3])  # （300，25088）
    fc_6 = np.dot(pool5_straight, model['fc6_w'].transpose()) + model['fc6_b'].transpose()         #（300，4096）
    relu_6 = ReLU(fc_6)

    # 全连接模块2
    fc_7 = np.dot(relu_6, model['fc7_w'].transpose()) + model['fc7_b'].transpose()   #（300，4096）
    relu_7 = ReLU(fc_7)

    # 网络分类和回归输出
    boxes_deltas = np.dot(relu_7, model['bbox_w'].transpose()) + model['bbox_b'].transpose()     # （300，class_num * 4）; class_num = 9
    score = np.dot(relu_7, model['score_w'].transpose()) + model['score_b'].transpose()          # （300，class_num）
    prob = Softmax(score)

    # t4 = time.time()
    # print('1', t1-t0)
    # print('2', t2 - t1)
    # print('3', t3 - t2)
    # print('4', t4 - t3)

    return boxes_deltas, prob, rois
