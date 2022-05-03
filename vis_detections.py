import matplotlib.pyplot as plt
import numpy as np

# 增加ax参数显示一张图像中所有预测目标信息
def vis_detection(class_name, dets, ax, thresh=0.5):  #im:图像像素矩阵；class_name：类名;dets：bbox的坐标和score
                                                       # #缺省thresh=0.5，实际传参是CONF_THRESH=0.8;
    inds = np.where(dets[:, -1] >= thresh)[0] #保留score大于0.8的bbox,[0]？ 属于单类的多bbox
    print(dets.shape, inds.shape)
    if len(inds) == 0:  #如果分类score没有大于等于0.8的行（bbox），那么直接返回
        return

    for i in inds:
        bbox = dets[i, :4]  #1-4列
        score = dets[i, -1] #5列（最后一列）
        
        ax.add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], fill=False,
                    edgecolor='red', linewidth=3.5))                                  #根据起始点坐标以及w,h 画出矩形框
        ax.text(bbox[0], bbox[1] - 2, '{:s} {:.3f}'.format(class_name, score),       #添加文本
                bbox=dict(facecolor='blue', alpha=0.5), fontsize=14, color='white')  #蓝底白字

    # ax.set_title(('{} detections with p({} | box) >= {:.1f}').format(class_name, class_name, thresh), fontsize=14)

