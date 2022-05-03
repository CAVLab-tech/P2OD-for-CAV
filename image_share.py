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

image = cv2.imread('KITTI/testing/000004.png')  #读取图片像素值

image1 = np.random.randint(2**8, size=image.shape, dtype=np.uint8)    #子图像1
cv2.imwrite('image1.png',image1)

image2 = image - image1         ##子图像2
cv2.imwrite('image2.png',image2)

# image_recover = image1 + image2
# error = image_recover - image

cv2.namedWindow("Image")   #创建窗口
cv2.namedWindow("Image1")   #创建窗口
cv2.namedWindow("Image2")   #创建窗口
cv2.imshow("Image", image)   #显示图片
cv2.imshow("Image1", image1)   #显示图片
cv2.imshow("Image2", image2)   #显示图片
cv2.waitKey (0)    #等待用户操作
cv2.destroyAllWindows()   #释放窗口
