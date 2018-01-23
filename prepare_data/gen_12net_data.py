#coding:utf-8
"""
- for every face in the picture, there will produce about 
pos: 6, part: 14, neg: 55
"""
import sys
import numpy as np
import cv2
import os
import numpy.random as npr

from os.path import abspath, join, dirname
sys.path.append(abspath(dirname(__file__)))
#print(sys.path)

import my_utils
#from utils import IoU

anno_file = "wider_face_train.txt"
im_dir = "WIDER_train/images"
pos_save_dir = "12/positive"
part_save_dir = "12/part"
neg_save_dir = '12/negative'
save_dir = "./12"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
if not os.path.exists(pos_save_dir):
    os.mkdir(pos_save_dir)
if not os.path.exists(part_save_dir):
    os.mkdir(part_save_dir)
if not os.path.exists(neg_save_dir):
    os.mkdir(neg_save_dir)

f1 = open(os.path.join(save_dir, 'pos_12.txt'), 'w')
f2 = open(os.path.join(save_dir, 'neg_12.txt'), 'w')
f3 = open(os.path.join(save_dir, 'part_12.txt'), 'w')
with open(anno_file, 'r') as f:
    annotations = f.readlines()
num = len(annotations)
print("%d pics in total" % num)
p_idx = 0 # positive
n_idx = 0 # negative
d_idx = 0 # dont care
idx = 0
box_idx = 0
for annotation in annotations:
    annotation = annotation.strip().split(' ')
    #image path
    im_path = annotation[0]
    #boxed change to float type
    # Before Python3, map() used to return a list,
    # With Python 3, map() returns an iterator, so list() must be called. 
    bbox = list(map(float, annotation[1:]))
    #gt
    #From 1-d to 2-d,because there may be many faces in the picture.
    #Every line of boxes is [nx, ny, nx + size, ny + size], 
    #that is [left-top-X, left-top-Y, right-bottom-X, right-bottom-Y]
    boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
    #load image
    img = cv2.imread(os.path.join(im_dir, im_path + '.jpg'))
    idx += 1
    if idx % 100 == 0:
        #print idx, "images done"
        print("%d images done" % (idx))
        
    height, width, channel = img.shape

    neg_num = 0
    #1---->50, generate 50 negative examples
    while neg_num < 50:
        #neg_num's size [40,min(width, height) / 2],min_size:40
        #计算滑动窗口的size，滑动窗口是正方形
        size = npr.randint(12, min(width, height) / 2)
        #在整个图片范围内，计算滑动窗口的top_left坐标
        nx = npr.randint(0, width - size)
        ny = npr.randint(0, height - size)
        #random crop，得到滑动窗口的boundary box
        crop_box = np.array([nx, ny, nx + size, ny + size])
        #计算 iou，返回numpy array
        Iou = my_utils.IoU(crop_box, boxes)
        
        cropped_im = img[ny : ny + size, nx : nx + size]
        resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)
        #if the IoUs of the cropped patch and every face boxes are all less then 0.3
        #np.max()返回array中的最大值
        #滑动窗口和图片中所有face boudary region的IoU都不大于0.3
        if np.max(Iou) < 0.3:
            # Iou with all gts must below 0.3
            save_file = os.path.join(neg_save_dir, "%s.jpg"%n_idx)
            f2.write("12/negative/%s.jpg"%n_idx + ' 0\n')
            cv2.imwrite(save_file, resized_im)
            n_idx += 1
            neg_num += 1
    #as for 正 part样本
    for box in boxes:
        # box (x_left, y_top, x_right, y_bottom)
        x1, y1, x2, y2 = box
        #gt's width
        w = x2 - x1 + 1
        #gt's height
        h = y2 - y1 + 1

        # ignore small faces
        # in case the ground truth boxes of small faces are not accurate
        if max(w, h) < 40 or x1 < 0 or y1 < 0:
            continue
        #在每个face boundary region内还要生成5个negative examples
        for i in range(5):
            #return a single int value between 12 and min(width, height) / 2
            size = npr.randint(12, min(width, height) / 2)
            # delta_x and delta_y are offsets of (x1, y1)
            delta_x = npr.randint(max(-size, -x1), w)
            delta_y = npr.randint(max(-size, -y1), h)
            #计算滑动窗口top-left坐标(nx1, ny1)
            nx1 = int(max(0, x1 + delta_x))
            ny1 = int(max(0, y1 + delta_y))
            if nx1 + size > width or ny1 + size > height:
                continue
            crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
            Iou = my_utils.IoU(crop_box, boxes)
    
            cropped_im = img[ny1: ny1 + size, nx1: nx1 + size, :]
            resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)
    
            if np.max(Iou) < 0.3:
                # Iou with all gts must below 0.3
                save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                f2.write("12/negative/%s.jpg" % n_idx + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1        
	      # generate 6 positive examples and 14 part faces
        for i in range(20):
            # pos and part face size between [minsize*0.8,maxsize*1.25]
            # 生成pos和part examples的滑动窗口的size的计算方法明显不同于生成neg examples的slide window的
            size = npr.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))

            # delta here is the offset of box center
            delta_x = npr.randint(-w * 0.2, w * 0.2)
            delta_y = npr.randint(-h * 0.2, h * 0.2)
            #在face bondary region周边计算滑动窗口的top-left坐标
            #show this way: nx1 = max(x1+w/2-size/2+delta_x)
            nx1 = max(x1 + w / 2 + delta_x - size / 2, 0)
            #show this way: ny1 = max(y1+h/2-size/2+delta_y)
            ny1 = max(y1 + h / 2 + delta_y - size / 2, 0)
            nx2 = nx1 + size
            ny2 = ny1 + size

            if nx2 > width or ny2 > height:
                continue 
            crop_box = np.array([nx1, ny1, nx2, ny2])
            #yu gt(ground truth) de offset
            #计算归一化的slide window和face boundary region的坐标偏移值
            offset_x1 = (x1 - nx1) / float(size)
            offset_y1 = (y1 - ny1) / float(size)
            offset_x2 = (x2 - nx2) / float(size)
            offset_y2 = (y2 - ny2) / float(size)
            #crop
            cropped_im = img[int(ny1) : int(ny2), int(nx1) : int(nx2)]
            #resize
            resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

            box_ = box.reshape(1, -1)
            if my_utils.IoU(crop_box, box_) >= 0.65:
                save_file = os.path.join(pos_save_dir, "%s.jpg"%p_idx)
                #I do not know why to write the offset into the file?
                f1.write("12/positive/%s.jpg"%p_idx + ' 1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                p_idx += 1
            elif my_utils.IoU(crop_box, box_) >= 0.4:
                save_file = os.path.join(part_save_dir, "%s.jpg"%d_idx)
                f3.write("12/part/%s.jpg"%d_idx + ' -1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                d_idx += 1
        box_idx += 1
    print("%s images done, pos: %s part: %s neg: %s"%(idx, p_idx, d_idx, n_idx))
    #break
f1.close()
f2.close()
f3.close()
