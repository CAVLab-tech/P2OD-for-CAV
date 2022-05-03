
import numpy as np
from Generate_anchors import generate_anchors
from Transform_box import Transform_box, STrans
from Clip_box import Clip_box, SClip
from Filter_box import Filter_box, SFilter
from NMS import NMS, SNMS
import time
import random



def Proposal(rpn_prob, rpn_bbox, im_info):   # 明文意义下的生成bbox, im_info = np.array([600., 800., 1.6], dtype=np.float32)  #公共参数
    feat_stride = 16                # 映射倍数
    anchors = generate_anchors()    # 生成anchors，（9,4）
    num_anchors = anchors.shape[0]  # anchors的数量9

    pre_nms_topN  = 1000 # Number of top scoring boxes to keep before apply NMS.
    post_nms_topN = 300  # Number of top scoring boxes to keep after applying NMS.
    nms_thresh    = 0.7  # NMS threshold used on RPN proposals.
    min_size      = 16   # Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)

    scores = rpn_prob[:, num_anchors:, :, :]  #(1,9,14,14)   #两个分量,rpn_cls_prob_reshape(1,2*9,14,14),前9个是背景bg，所以取后9个fg。
    bbox_deltas = rpn_bbox                    #rpn_bbox_pred(1,4*9,14,14),每个anchor都是4元组坐标变换信息   #两个分量

    height, width = scores.shape[-2:]
    shift_x = np.arange(0, width) * feat_stride  #(w,)  # Enumerate all shifts
    shift_y = np.arange(0, height) * feat_stride #(h,)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)    # (w,h)大小的网格
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose() 
    #ravel函数负责降维，先行后列,transpose转置函数, (w*h,4)大小,w*h个网格坐标，(x1,y1,x2,y2),x1=x2,y1=y2

    A = num_anchors  #9
    K = shifts.shape[0] #h*w
    anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    anchors = anchors.reshape((K * A, 4))  #每个坐标点都有9个anchors，总共A*K个

    bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4)) # bbox_deltas(1, 4 * A, H, W)->(1, H, W, 4 * A)->(1 * H * W * A, 4)
    scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))  # scores(1, A, H, W)->(1, H, W, A)->(1 * H * W * A, 1)

    proposals = Transform_box(anchors, bbox_deltas)   # 根据anchor和偏移量bbox_deltas计算预测proposals
    proposals = Clip_box(proposals, im_info[:2])  # 处理超过图像边界的bbox，使得pred_boxes位于图片内
    keep = Filter_box(proposals, min_size * im_info[2])   # 删掉特别小的proposals
    proposals = proposals[keep, :]
    scores = scores[keep]

    order = scores.ravel().argsort()[::-1]  # 降序
    order = order[:pre_nms_topN]            # take top pre_nms_topN (e.g. 6000)

    proposals = proposals[order, :]
    scores = scores[order]

    keep = NMS(np.hstack((proposals, scores)), nms_thresh)   # apply nms (e.g. threshold = 0.7)
    keep = keep[:post_nms_topN]   # take after_nms_topN (e.g. 300)
    proposals = proposals[keep, :]
    #
    batch_inds = np.zeros((proposals.shape[0], 1))         # only supports a single input image, so all batch inds are 0
    rois = np.hstack((batch_inds, proposals))  # rois blob: (n = 0, x1, y1, x2, y2)
    return rois


def SProposal(rpn_prob_1, rpn_prob_2, rpn_bbox_1, rpn_bbox_2, im_info):   # 密文意义下的生成bbox
    feat_stride = 16                #映射倍数
    anchors = generate_anchors()    #生成anchors，（9,4）
    num_anchors = anchors.shape[0]  #anchors的数量9

    pre_nms_topN  = 1000 # Number of top scoring boxes to keep before apply NMS.
    post_nms_topN = 300  # Number of top scoring boxes to keep after applying NMS.
    nms_thresh    = 0.7  # NMS threshold used on RPN proposals.
    min_size      = 16  # Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
      
    scores_1 = rpn_prob_1[:, num_anchors:, :, :]  # (1,9,14,14)   #两个分量
    scores_2 = rpn_prob_2[:, num_anchors:, :, :]
    k = 2
    _rpn_bbox_1 = np.random.uniform(-k, k, rpn_bbox_1.shape)  # S1  if x1 <= -k & x1 >= k:
    s_1 = rpn_bbox_1 - _rpn_bbox_1
    _rpn_bbox_2 = rpn_bbox_2 + s_1  # S2
    bbox_deltas_1 = _rpn_bbox_1   # rpn_bbox_pred(1,4*9,14,14),每个anchor都是4元组坐标变换信息   #两个分量
    bbox_deltas_2 = _rpn_bbox_2

    height, width = scores_1.shape[-2:]
    shift_x = np.arange(0, width) * feat_stride  # (w,)
    shift_y = np.arange(0, height) * feat_stride # (h,)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)   # (w,h)大小的网格
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()

    A = num_anchors     # 9
    K = shifts.shape[0] # h*w
    anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    anchors = anchors.reshape((K * A, 4))  # 每个坐标点都有9个anchors，总共A*K个

    bbox_deltas_1 = bbox_deltas_1.transpose((0, 2, 3, 1)).reshape((-1, 4))
    bbox_deltas_2 = bbox_deltas_2.transpose((0, 2, 3, 1)).reshape((-1, 4))

    scores_1 = scores_1.transpose((0, 2, 3, 1)).reshape((-1, 1))
    scores_2 = scores_2.transpose((0, 2, 3, 1)).reshape((-1, 1))

    # print(np.max(np.abs(bbox_deltas_1)), np.max(np.abs(bbox_deltas_2)), np.max(np.abs(bbox_deltas_1 + bbox_deltas_2)))

    proposals_1, proposals_2 = STrans(anchors, bbox_deltas_1, bbox_deltas_2)
    proposals_1, proposals_2 = SClip(proposals_1, proposals_2, im_info[:2])

    keep = SFilter(proposals_1, proposals_2, min_size * im_info[2])
    proposals_1 = proposals_1[keep, :]
    proposals_2 = proposals_2[keep, :]
    scores_1 = scores_1[keep]
    scores_2 = scores_2[keep]

    m_1 = random.randint(0, 10 ** 3)
    m_2 = random.randint(0, 10 ** 3)
    scores_1_fic = scores_1 + m_1       # 属于该检测框的概率分数
    scores_2_fic = scores_2 + m_2
    scores_fic = (scores_1_fic).ravel() + (scores_2_fic).ravel()  # score - m, 混淆数组，但不改变排序结果
    order = scores_fic.argsort()[::-1]  # 降序
    order = order[:pre_nms_topN]

    proposals_1 = proposals_1[order, :]
    proposals_2 = proposals_2[order, :]
    scores_1 = scores_1[order]
    scores_2 = scores_2[order]

    keep = SNMS(np.hstack((proposals_1, scores_1)), np.hstack((proposals_2, scores_2)), nms_thresh)

    keep = keep[:post_nms_topN]
    proposals_1 = proposals_1[keep, :]
    proposals_2 = proposals_2[keep, :]
    #
    batch_inds = np.zeros((proposals_1.shape[0], 1))
    rois_1 = np.hstack((batch_inds, proposals_1))  # rois blob: (n, x1, y1, x2, y2)
    rois_2 = np.hstack((batch_inds, proposals_2))
    return rois_1, rois_2


# import sys
# sys.path.insert(0, r'C:\Users\ASUS\Desktop\P2OD_numpy\layers')
# sys.path.insert(0, r'C:\Users\ASUS\Desktop\P2OD_numpy')
#
# image = np.load('image.npy')
# im_scale = np.load('im_scale.npy')
#
# rpn_prob_1 = np.load('rpn_prob_one.npy')
# rpn_prob_2 = np.load('rpn_prob_two.npy')
#
# rpn_bbox_1 = np.load('rpn_bbox_one.npy')
# rpn_bbox_2 = np.load('rpn_bbox_two.npy')
#
# rpn_prob = np.load('rpn_prob.npy')
# rpn_bbox = np.load('rpn_bbox.npy')
#
# rpn_prob_2 = rpn_prob - rpn_prob_1
#
# rois = Proposal(rpn_prob, rpn_bbox, im_info = np.array([image.shape[1], image.shape[2], im_scale]))
# rois_1, rois_2 = SProposal(rpn_prob_1, rpn_prob_2, rpn_bbox_1, rpn_bbox_2, im_info = np.array([image.shape[1], image.shape[2], im_scale]))
#
# err_rois = rois - (rois_1 + rois_2)

# proposals, scores, keep_1, order_1 = Proposal(rpn_prob, rpn_bbox, im_info = np.array([image.shape[1], image.shape[2], im_scale]))
# proposals_1, proposals_2, scores_1, score_2, keep_2, order_2 = SProposal(rpn_prob_1, rpn_prob_2, rpn_bbox_1, rpn_bbox_2, im_info = np.array([image.shape[1], image.shape[2], im_scale]))

# err_keep = keep_1 - keep_2
# err_order = order_1 - order_2
# err_score = scores - (scores_1 + score_2)
# err_proposals = proposals - (proposals_1 + proposals_2)


# err_rois = rois_1 + rois_2 - rois
# err_keep = np.array(keep_new) - np.array(keep_ori)











# image = np.load('image.npy')
# im_scale = np.load('im_scale.npy')
# rpn_prob = np.load('rpn_prob.npy')       #np.random.uniform(0, 10**2,(1, 18, 38, 50))
# rpn_bbox = np.load('rpn_bbox.npy')       #np.random.uniform(0, 10**2,(1, 36, 38, 50))
# rpn_prob_1 = np.load('rpn_prob_one.npy')
# rpn_bbox_1 = np.load('rpn_bbox_one.npy')
# rpn_prob_2 = np.load('rpn_prob_two.npy')
# rpn_bbox_2 = np.load('rpn_bbox_two.npy')
#
# err_rpn_bbox = rpn_bbox_1 + rpn_bbox_2 - rpn_bbox
# err_rpn_prob = rpn_prob_1 + rpn_prob_2 - rpn_prob

# t1 = time.time()
# rois = Proposal(rpn_prob, rpn_bbox, im_info = np.array([image.shape[1], image.shape[2], im_scale]))
# t2 = time.time()
# rois_1, rois_2 = SProposal(rpn_prob_1, rpn_prob_2, rpn_bbox_1, rpn_bbox_2, im_info = np.array([image.shape[1], image.shape[2], im_scale]))
# t3 = time.time()

# err_s = scores_1 + scores_2 - scores
# err_p = proposals_1 + proposals_2 - proposals
# err_o = order_new - order
# err_rois = rois_1 + rois_2 - rois

# print('plain', (t2-t1)*1000)
# print('cipher', (t3-t2)*1000)
# error = rois_1 + rois_2 - rois

# bbox_deltas_sec = np.load('bbox_deltas_sec.npy')
# proposals_clip_sec = np.load('proposals_clip_sec.npy')
# keep_sec = np.load('keep_sec.npy')
# bbox_deltas = np.load('bbox_deltas.npy')
# proposals_clip = np.load('proposals_clip.npy')
# keep = np.load('keep.npy')
# error_b = bbox_deltas_sec - bbox_deltas
# error_clip = proposals_clip_sec - proposals_clip
# error_k = keep_sec - keep

# scores_sec = np.load('scores_sec.npy')
# proposals_sec = np.load('proposals_sec.npy')
# scores = np.load('scores.npy')
# proposals = np.load('proposals.npy')
# error_s = scores_sec - scores
# error_p = proposals_sec - proposals
