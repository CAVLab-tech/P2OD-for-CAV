
import numpy as np
import os, cv2
import sys
import pickle as pk


# generate randomly the content of model parameter
conv1_1_w = np.load('conv1_1_w.npy')
conv1_1_b = np.load('conv1_1_b.npy')
conv1_2_w = np.load('conv1_2_w.npy')
conv1_2_b = np.load('conv1_2_b.npy')

conv2_1_w = np.load('conv2_1_w.npy')
conv2_1_b = np.load('conv2_1_b.npy')
conv2_2_w = np.load('conv2_2_w.npy')
conv2_2_b = np.load('conv2_2_b.npy')

conv3_1_w = np.load('conv3_1_w.npy')
conv3_1_b = np.load('conv3_1_b.npy')
conv3_2_w = np.load('conv3_2_w.npy')
conv3_2_b = np.load('conv3_2_b.npy')
conv3_3_w = np.load('conv3_3_w.npy')
conv3_3_b = np.load('conv3_3_b.npy')

conv4_1_w = np.load('conv4_1_w.npy')
conv4_1_b = np.load('conv4_1_b.npy')
conv4_2_w = np.load('conv4_2_w.npy')
conv4_2_b = np.load('conv4_2_b.npy')
conv4_3_w = np.load('conv4_3_w.npy')
conv4_3_b = np.load('conv4_3_b.npy')

conv5_1_w = np.load('conv5_1_w.npy')
conv5_1_b = np.load('conv5_1_b.npy')
conv5_2_w = np.load('conv5_2_w.npy')
conv5_2_b = np.load('conv5_2_b.npy')
conv5_3_w = np.load('conv5_3_w.npy')
conv5_3_b = np.load('conv5_3_b.npy')

rpn_conv_w = np.load('rpn_conv_w.npy')
rpn_conv_b = np.load('rpn_conv_b.npy')
rpn_score_w = np.load('rpn_score_w.npy')
rpn_score_b = np.load('rpn_score_b.npy')
rpn_bbox_w = np.load('rpn_bbox_w.npy')
rpn_bbox_b = np.load('rpn_bbox_b.npy')

fc6_w = np.load('fc6_w.npy')
fc6_b = np.load('fc6_b.npy')
fc7_w = np.load('fc7_w.npy')
fc7_b = np.load('fc7_b.npy')

score_w = np.load('score_w.npy')
score_b = np.load('score_b.npy')
bbox_w = np.load('bbox_w.npy')
bbox_b = np.load('bbox_b.npy')


# save the model parameter as dict data type by pickle library
keys = ['conv1_1_w', 'conv1_1_b', 'conv1_2_w', 'conv1_2_b',
        'conv2_1_w', 'conv2_1_b', 'conv2_2_w', 'conv2_2_b',
        'conv3_1_w', 'conv3_1_b', 'conv3_2_w', 'conv3_2_b', 'conv3_3_w', 'conv3_3_b',
        'conv4_1_w', 'conv4_1_b', 'conv4_2_w', 'conv4_2_b', 'conv4_3_w', 'conv4_3_b',
        'conv5_1_w', 'conv5_1_b', 'conv5_2_w', 'conv5_2_b', 'conv5_3_w', 'conv5_3_b',
        'rpn_conv_w', 'rpn_conv_b', 'rpn_score_w', 'rpn_score_b', 'rpn_bbox_w', 'rpn_bbox_b',
        'fc6_w', 'fc6_b', 'fc7_w', 'fc7_b', 'score_w', 'score_b', 'bbox_w', 'bbox_b']

values = [conv1_1_w, conv1_1_b, conv1_2_w, conv1_2_b,
          conv2_1_w, conv2_1_b, conv2_2_w, conv2_2_b,
          conv3_1_w, conv3_1_b, conv3_2_w, conv3_2_b, conv3_3_w, conv3_3_b,
          conv4_1_w, conv4_1_b, conv4_2_w, conv4_2_b, conv4_3_w, conv4_3_b,
          conv5_1_w, conv5_1_b, conv5_2_w, conv5_2_b, conv5_3_w, conv5_3_b,
          rpn_conv_w, rpn_conv_b, rpn_score_w, rpn_score_b, rpn_bbox_w, rpn_bbox_b,
          fc6_w, fc6_b, fc7_w, fc7_b, score_w, score_b, bbox_w, bbox_b]

model_dict = dict(zip(keys, values))


# save dict data as pkl file
f_w = open('model.pkl', 'wb')  # 建立文件对象
pk.dump(model_dict, f_w)       # 将model字典保存为pkl文件
f_w.close()                    # 记得关闭文件！！！

