import torch
import numpy as np
import time


def add_bias(conv_weight, conv_bias):  # conv_weight.shape = (core_num, h, w); conv_bias.shape = (core_num, 1)
    conv_out = conv_weight + conv_bias.reshape(conv_bias.shape[0], 1, 1)
    return conv_out


def mul_weight(img, conv_filter, zero_num = 1):  # img：(ch, h, w)；conv_filter：(num, ch, h, w)
    img_padding = np.pad(img, ((0, 0), (zero_num, zero_num), (zero_num, zero_num)), 'constant')  # 缺省填充0
    img_ch, img_h, img_w = img_padding.shape
    filter_num, img_ch, filter_h, filter_w = conv_filter.shape
    feature_h = img_h - filter_h + 1
    feature_w = img_w - filter_w + 1

    # 将输入图片张量转换成矩阵形式 (3D-->2D)
    img_new = img_padding.transpose(1, 2, 0).reshape(img_h, img_w * img_ch)
    shape = (feature_h, feature_w, filter_h, filter_w * img_ch)
    strides = img_new.itemsize * np.array((img_w * img_ch, img_ch, img_w * img_ch, 1))
    x_stride = np.lib.stride_tricks.as_strided(img_new, shape=shape, strides=strides)
    img_matrix = x_stride.reshape(feature_h * feature_w, filter_h * filter_w * img_ch).astype(np.float64)

    # 将卷积核张量转换成矩阵形式  (num, ch, h, w) --> (ch*h*w, num)
    filter_matrix = conv_filter.transpose(0, 2, 3, 1).reshape(filter_num, filter_w * filter_h * img_ch).transpose()
    feature_matrix = np.dot(img_matrix, filter_matrix)                                  # conv (h*w, num)
    conv_weight = feature_matrix.transpose().reshape(filter_num, feature_h, feature_w)  # (h*w, num)-->(num, h, w)
    return conv_weight


# channel = 256
# outnum = 512
# sh_image = (channel, 100, 100)     #(channel, h, w)
# sh_filter = (outnum, channel, 1, 1)  #(num, channel, h, w)
# sh_bias = (outnum, 1)     # #(num, 1)
# image = np.random.randint(-10, 10, sh_image)   # 卷积输入
# filter = np.random.randint(-10, 10, sh_filter)   # 卷积核_权重
# bias = np.random.randint(-10, 10, sh_bias)   # 卷积核_偏置
#
# t1 = time.time()
# conv_output_1 = vector_conv(image, filter, 1)
# output_1 = add_bias(conv_output_1, bias)
# t2 = time.time()
# conv_output_2 = mul_weight(image, filter, 0)

# output_2 = add_bias(conv_output_2, bias)
# t3 = time.time()
#
# error = output_2 - output_1
# ee= np.max(np.abs(error))
#
# error_1 = conv_output_2 - conv_output_1
# print('conv_im2col', (t2 - t1)*1000)
# print('conv_paper', (t3 - t2)*1000)

# 逐点卷积
# b = image[np.newaxis] * filter    # (outnum, channel, 100, 100)
# c = np.sum(b, axis=1)             # (outnum, 100, 100)
# error = conv_output_2 - c
# error_max = np.max(np.abs(error))
