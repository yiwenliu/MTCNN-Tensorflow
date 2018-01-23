# coding: utf-8
import os
import time
import math
from os.path import join, exists
import cv2
import numpy as np
from BBox_utils import getDataFromTxt,processImage,shuffle_in_unison_scary,BBox
from Landmark_utils import show_landmark,rotate,flip
import random
import tensorflow as tf
import sys
import numpy.random as npr
dstdir = "12/train_PNet_landmark_aug"
OUTPUT = '12'
if not exists(OUTPUT): os.mkdir(OUTPUT)
if not exists(dstdir): os.mkdir(dstdir)
assert(exists(dstdir) and exists(OUTPUT))

def IoU(box, boxes):
    """Compute IoU between detect box and gt boxes

    Parameters:
    ----------
    box: numpy array , shape (5, ): x1, y1, x2, y2, score
        input box
    boxes: numpy array, shape (n, 4): x1, y1, x2, y2
        input ground truth boxes

    Returns:
    -------
    ovr: numpy.array, shape (n, )
        IoU
    """
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
     # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)
    inter = w * h
    ovr = inter*1.0 / (box_area + area - inter)
    return ovr

def GenerateData(ftxt, output, net, augmentation=False):
    """Write a landmark annotation txt & Crop the corresponding image patches
       ftxt: "FacePoint_train/trainImageList.txt"
       output: 12
    """
    if net == "PNet":
        size = 12
    elif net == "RNet":
        size = 24
    elif net == "ONet":
        size = 48
    else:
        print('Net type error')
        return
    image_id = 0
    f = open(join(output,"landmark_%s_aug.txt" %(size)),'w')
    #dstdir = "train_landmark_few"
    
    #data: [(img_path, BBox_object, landmark_array_5*2)], a list of tuple
    data = getDataFromTxt(ftxt)
    idx = 0
    #image_path bbox(BBox_object) landmarkGt(array of 5*2)
    for (imgPath, bbox, landmarkGt) in data:
        #print imgPath
        F_imgs = [] #the list of face bondary region pixes
        F_landmarks = [] #the list of landmark array
        img = cv2.imread(imgPath)
        assert(img is not None)
        img_h,img_w,img_c = img.shape
        gt_box = np.array([bbox.left,bbox.top,bbox.right,bbox.bottom])
        f_face = img[bbox.top:bbox.bottom+1,bbox.left:bbox.right+1]
        #@f_face: face bondary region pixes
        #@landmark: relative position of the five landmark
        f_face = cv2.resize(f_face,(size,size))
        landmark = np.zeros((5, 2))
        #@rv:人脸的面部轮廓关键点不采用绝对坐标,而采用相对于Bounding Box top-left点的相对坐标
        #@one: iterate each row in the landmarkGt array
        for index, one in enumerate(landmarkGt):
            rv = ((one[0]-gt_box[0])/(gt_box[2]-gt_box[0]), (one[1]-gt_box[1])/(gt_box[3]-gt_box[1]))
            landmark[index] = rv
        
        F_imgs.append(f_face)
        F_landmarks.append(landmark.reshape(10))
        #print(landmark.reshape(10))
        landmark = np.zeros((5, 2)) #landmark array should be zero after every time of landmark annotation        
        #如果需要强化，就在gt face bbx内，用滑动窗口选取postive face bbox，
        #计算gt landmark相对于这个positive face bbox的归一化偏移值
        if augmentation:
            idx = idx + 1
            if idx % 100 == 0:
                print(" %d images done"%(idx))
            x1, y1, x2, y2 = gt_box
            #gt's width
            gt_w = x2 - x1 + 1
            #gt's height
            gt_h = y2 - y1 + 1        
            if max(gt_w, gt_h) < 40 or x1 < 0 or y1 < 0:
                continue
            #random shift, 
            #ten augmentative images for each original image
            for i in range(10):
                #bbox_size是滑动窗口的尺寸
                #和gen_12net_data.py中生成pos和part时的size相等
                bbox_size = npr.randint(int(min(gt_w, gt_h) * 0.8), np.ceil(1.25 * max(gt_w, gt_h)))
                delta_x = npr.randint(-gt_w * 0.2, gt_w * 0.2)
                delta_y = npr.randint(-gt_h * 0.2, gt_h * 0.2)
                nx1 = int(max(x1+gt_w/2-bbox_size/2+delta_x,0))
                ny1 = int(max(y1+gt_h/2-bbox_size/2+delta_y,0))
                
                nx2 = int(nx1 + bbox_size)
                ny2 = int(ny1 + bbox_size)
                if nx2 > img_w or ny2 > img_h:
                    continue
                crop_box = np.array([nx1,ny1,nx2,ny2])
                cropped_im = img[ny1:ny2+1,nx1:nx2+1]
                resized_im = cv2.resize(cropped_im, (size, size))
                #cal iou
                iou = IoU(crop_box, np.expand_dims(gt_box,0))
                #取positive face部分进行强化
                if iou > 0.65:
                    F_imgs.append(resized_im)
                    #normalize
                    for index, one in enumerate(landmarkGt):
                        rv = ((one[0]-nx1)/bbox_size, (one[1]-ny1)/bbox_size)
                        landmark[index] = rv
                    F_landmarks.append(landmark.reshape(10))
                    landmark = np.zeros((5, 2))
                    landmark_ = F_landmarks[-1].reshape(-1,2)
                    bbox = BBox([nx1,ny1,nx2,ny2])                    

                    #mirror                    
                    if random.choice([0,1]) > 0:#在0和1之间选择
                        #flip() is defined in Landmark_utils.py
                        #仅对从gt bbx抠出来的postive face bbx进行镜像
                        face_flipped, landmark_flipped = flip(resized_im, landmark_)
                        face_flipped = cv2.resize(face_flipped, (size, size))
                        #c*h*w
                        F_imgs.append(face_flipped)
                        F_landmarks.append(landmark_flipped.reshape(10))
                    
                    #inverse clockwise rotate，这部分代码应该和"if iou > 0.65:"平级，因为并非用修剪后的图片进行旋转
                    if random.choice([0,1]) > 0:
                        #rotate() is defined in Landmark_utils.py
                        #对整个图片进行逆时针旋转
                        face_rotated_by_alpha, landmark_rotated = rotate(img, bbox, \
                                                                         bbox.reprojectLandmark(landmark_), 5)#逆时针旋转
                        #landmark_offset
                        landmark_rotated = bbox.projectLandmark(landmark_rotated)
                        face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (size, size))
                        F_imgs.append(face_rotated_by_alpha)
                        F_landmarks.append(landmark_rotated.reshape(10))
                        #flip，旋转后再镜像
                        face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                        face_flipped = cv2.resize(face_flipped, (size, size))
                        F_imgs.append(face_flipped)
                        F_landmarks.append(landmark_flipped.reshape(10))                
                    
                    #clockwise rotation（顺时针旋转）
                    if random.choice([0,1]) > 0: 
                        face_rotated_by_alpha, landmark_rotated = rotate(img, bbox, \
                                                                         bbox.reprojectLandmark(landmark_), -5)#顺时针旋转
                        landmark_rotated = bbox.projectLandmark(landmark_rotated)
                        face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (size, size))
                        F_imgs.append(face_rotated_by_alpha)
                        F_landmarks.append(landmark_rotated.reshape(10))
                        #flip
                        face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                        face_flipped = cv2.resize(face_flipped, (size, size))
                        F_imgs.append(face_flipped)
                        F_landmarks.append(landmark_flipped.reshape(10)) 
            #end of for        
        #end of if augmentation
        #array和asarray都可以将结构数据转化为ndarray，但是主要区别就是当数据源是ndarray时，array仍然会copy出一个副本，占用新的内存，但asarray不会。
        F_imgs, F_landmarks = np.asarray(F_imgs), np.asarray(F_landmarks)
        print(F_imgs.shape)
        print(F_landmarks.shape)
        for i in range(len(F_imgs)):
            #print(image_id)
            if np.sum(np.where(F_landmarks[i] <= 0, 1, 0)) > 0:
                continue
            if np.sum(np.where(F_landmarks[i] >= 1, 1, 0)) > 0:
                continue
            cv2.imwrite(join(dstdir,"%d.jpg" %(image_id)), F_imgs[i])
            landmarks = list(map(str,list(F_landmarks[i])))
            f.write(join(dstdir,"%d.jpg" %(image_id))+" -2 "+" ".join(landmarks)+"\n")
            image_id = image_id + 1
        #break
            
    #print F_imgs.shape
    #print F_landmarks.shape
    #F_imgs = processImage(F_imgs)
    #shuffle_in_unison_scary(F_imgs, F_landmarks)
    
    f.close()
    return F_imgs,F_landmarks

if __name__ == '__main__':
    # train data
    net = "PNet"
    #train_txt = "train.txt"
    train_txt = "FacePoint_train/trainImageList.txt"
    imgs,landmarks = GenerateData(train_txt, OUTPUT,net,augmentation=True)
    
   
