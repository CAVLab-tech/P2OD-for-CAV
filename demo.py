import torch
import numpy as np
import os, cv2
import sys
import time
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
# print(os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, r'D:\+++learning\+++code\+++our\Secure_Faster_RCNN\P2OD_numpy\layers')

from faster_rcnn_plain import faster_rcnn_plain
from faster_rcnn_sec import faster_rcnn_sec
from layers.Transform_box import Transform_box
from layers.Clip_box import Clip_box
from layers.NMS import NMS
from vis_detections import vis_detection   # plot box


CLASSES = ('__background__', 'truck', 'pedestrian', 'van', 'car', 'tram')         # 5 classes + 1 background
CONF_THRESH = 0.8         # 分类score阈值
NMS_THRESH  = 0.3         # NMS的IoU阈值
target_size = 600         # 一般而言，缩放后的较短边
max_size    = 1000        # 缩放后的较长边

### 读取图像列表
dir = 'KITTI/testing'     # 测试数据集路径
filelist  = []
filenames = os.listdir(dir)
for fn in filenames:
	fullfilename = os.path.join(dir, fn)  # 补全图像路径
	filelist.append(fullfilename)         # 读取testing文件里的图像，添加至图像名列表

### 图像特征处理
for i in range(1):       # 顺序地读取并处理若干张图像
	print('开始测试第{}张图片'.format(i + 1))

	im_file = filelist[i]
	im_in = cv2.imread(im_file)                                # 读取图片像素矩阵
	im = im_in - np.array([[[102.9801, 115.9465, 122.7717]]])  # 将三个通道的像素值分别减去三个通道的平均值(1,1,3)
	im_shape = im.shape                  # 图像维度(375, 1242, 3)
	im_size_min = np.min(im_shape[0:2])  # 高
	im_size_max = np.max(im_shape[0:2])  # 宽

	im_scale = target_size / im_size_min              # 缩放比例(默认w轴和h轴比例相同)
	if np.round(im_scale * im_size_max) > max_size:   # 要求缩放后长度不超过1000
		im_scale = max_size / im_size_max
	image = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)  # 缩放为(302, 1000, 3)

	image = image.transpose(2, 0, 1)                    # (3, 302, 1000)
	image_1 = np.random.uniform(0, 2**8, image.shape)   # S1
	image_2 = image - image_1                           # S2

	t1 = time.time()
	# boxes_deltas, prob, rois = faster_rcnn_plain(image, im_scale)         # 目标检测（明文）
	t2 = time.time()
	print('*************')
	boxes_deltas_1, boxes_deltas_2, prob_1, prob_2, rois_1, rois_2 = faster_rcnn_sec(image_1, image_2, im_scale)    # 目标检测（密文）
	t3 = time.time()

	# err_rois = rois - (rois_1 + rois_2)
	# err_boxes_deltas = boxes_deltas - (boxes_deltas_1 + boxes_deltas_2)
	# err_prob = prob - (prob_1 + prob_2)

	boxes_deltas = boxes_deltas_1 + boxes_deltas_2
	prob = prob_1 + prob_2
	rois = rois_1 + rois_2

	### 图像目标测试
	boxes = rois[:, 1:5] / im_scale
	pred_boxes_trans = Transform_box(boxes, boxes_deltas)   # 预测边界框的转换
	pred_boxes_clip = Clip_box(pred_boxes_trans, im_shape)  # 预测边界框的裁剪

	im_rgb = im_in[:, :, (2, 1, 0)]               # 处理第三维像素通道，其中 读取图像按照BGR 存储，而画图时需要按RGB格式。
	fig, ax = plt.subplots(figsize=(12, 12))      # fig代表绘图窗口(Figure)；ax代表这个绘图窗口上的坐标系(axis)对象，画板的宽和高均为12
	ax.imshow(im_rgb, aspect='equal')             # 显示明文图像

	for cls_ind, cls in enumerate(CLASSES[1:]):  # cls_ind:类别索引0-8；cls：9种类别; 逐类地进行NMS和画框;
		cls_ind += 1                                                   # 跳过背景类，1-8(8个fg)
		cls_boxes = pred_boxes_clip[:, 4 * cls_ind:4 * (cls_ind + 1)]  # 8个fg的bbox坐标,(300,4)
		cls_scores = prob[:, cls_ind]                                  # 8个fg的bbox对应的score,(300,1)
		dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis]))       # 合并边界框和分数表示，300个bbox，(300,5)
		keep = NMS(dets, NMS_THRESH)                                   # 非极大值抑制，保留bbox的索引
		dets = dets[keep, :]

		vis_detection(cls, dets, ax, thresh=CONF_THRESH)  # im:图像像素矩阵；cls：类名;dets：bbox的坐标和score

	plt.axis('off')     # 不显示坐标框
	plt.tight_layout()  # 画图布局
	plt.draw()          # 将图形画在画板上
	plt.show()          # 显示画的图示

	print('结束测试第{}张图片'.format(i + 1))
	print('明文目标检测网络', t2 - t1)
	print('密文目标检测网络', t3 - t2)
