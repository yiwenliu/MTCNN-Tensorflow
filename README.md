## Description
This work is used for reproduce MTCNN,a Joint Face Detection and Alignment（对齐） using Multi-task Cascaded Convolutional Networks.

## Prerequisites
1. You need CUDA-compatible GPUs to train the model.
2. You should first download [WIDER Face](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/) and [Celeba](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).**WIDER Face** for face detection and **Celeba** for landmark detection(特征点检测，This is required by original paper.But I found some labels were wrong in Celeba. So I use [this dataset](http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm) for landmark detection).

## Dependencies
* Tensorflow 1.2.1
* TF-Slim
* Python 2.7
* Ubuntu 16.04
* Cuda 8.0

## Prepare For Training Data
1. Download Wider Face Training part only from Official Website , unzip to replace `WIDER_train` and put it into `prepare_data` folder.
2. Download landmark training data from [here](http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm),unzip, rename it as FacePoint_train and put them into `prepare_data` folder, then substitue FacePoint_train/trainImageList.txt with prepare_data/trainImageList.txt using cp command because of the / and \ directory seperator.
3. Run `prepare_data/gen_12net_data.py` to generate training data(Face Detection Part) for **PNet**.
  * The result is "12880 images done, pos: 196960 part: 540330 neg: 809650"
  
  * prepare_data/12下生成三个文件neg_12.txt，part_12.txt和pos_12.txt   
  Each line's format：[path to image] [cls_label] [bbox_label]   
  cls_label: 1 for positive, 0 for negative, -1 for part face.   
  bbox_label are the offset of x1, y1, x2, y2, calculated by (xgt(ygt) - x(y)) / width(height)   
  12/negative/0.jpg 0   
  12/part/0.jpg -1 0.05 0.15 -0.21 0.05    
  12/positive/0.jpg 1 0.12 0.04 0.04 0.18    
  
  * prepare_data/12下生成三个子目录：negative, part, positive，分别存储尺寸为12*12的“非脸部，部分脸部，脸部”训练图片。
4. Run `prepare_data/gen_landmark_aug_12.py` to generate training data(Face Landmark Detection Part) for **PNet**.
  * 在prepare_data/12下生成了子目录train_PNet_landmark_aug，其中存放的都是12x12的positive face
  * 在prepare_data/12下生成landmark_12_aug.txt，每行记录的格式:[path to image] [cls_label] [landmark_label]     
  landmark_label: 五个landmark点相对于bbx top-left的归一化偏移值      
  12/train_PNet_landmark_aug\0.jpg -2 0.288961038961 0.204545454545 0.814935064935 0.262987012987 0.535714285714 0.659090909091 0.275974025974 0.853896103896 0.724025974026 0.905844155844
                                      
5. Run `gen_imglist_pnet.py` to merge two parts of training data.
* 新建目录prepare_data\imglists\PNet
* 在上述目录下，新建train_PNet_landmark.txt，把landmark_12_aug.txt, neg_12.txt，part_12.txt和pos_12.txt中的每一行打乱了顺序，再写入新建annotation文件中
* train_PNet_landmark.txt中的每一行都保持了在原标记文件中的原样,features的数量并没有扩充
6. Run `gen_PNet_tfrecords.py` to generate tfrecord for **PNet**, prepare_data\imglists\PNet\train_PNet_landmark.tfrecord_shuffle.
* 采用protocol buffer的格式，把图片的内容和标记写到了一个文件中。
* 执行时间很长，但是报错——"段错误 (核心已转储)" on my ubuntu with GPU, and the file size is about 859,920,160 bytes.
* 在prepare_data\imglists\PNet下新建train_PNet_landmark.tfrecord_shuffle
7. (1)Run train_models/train_PNet.py to train **PNet**. 
  * 定义PNet：mtcnn_model.py/def P_Net()，包括神经网络结构，cost function    
  * 定义cost function, input pipeline, summary        
  * 模型数据保存在"/data/MTCNN_model/PNet_landmark"    
  在ubuntu GPU上执行这个.py时，出现了如下的问题：    
  * 未安装easydict, $pip install easydict或者$conda install -c auto easydict    
  (2)Then run `gen_hard_example.py --test_mode PNet` to generate training data(Face Detection Part) for **RNet**. 和本文档中第3步生成的内容相同。    
8. Run `gen_landmark_aug_24.py` to generate training data(Face Landmark Detection Part) for **RNet**.
9. Run `gen_imglist_rnet.py` to merge two parts of training data.
10. Run `gen_RNet_tfrecords.py` to generate tfrecords for **RNet**.(**you should run this script four times to generate tfrecords of neg,pos,part and landmark respectively**)
11. After training **RNet**, run `gen_hard_example.py --test_mode RNet` to generate training data(Face Detection Part) for **ONet**.
12. Run `gen_landmark_aug_48.py` to generate training data(Face Landmark Detection Part) for **ONet**.
13. Run `gen_imglist_onet.py` to merge two parts of training data.
14. Run `gen_ONet_tfrecords.py` to generate tfrecords for **ONet**.(**you should run this script four times to generate tfrecords of neg,pos,part and landmark respectively**)

## Some Details
* When training **PNet**,I merge four parts of data(pos,part,landmark,neg) into one tfrecord,since their total number radio is almost 1:1:1:3.But when training **RNet** and **ONet**,I generate four tfrecords,since their total number is not balanced.During training,I read 64 samples from pos,part and landmark tfrecord and read 192 samples from neg tfrecord to construct mini-batch.
* It's important for **PNet** and **RNet** to keep high recall radio.When using well-trained **PNet** to generate training data for **RNet**,I can get 14w+ pos samples.When using well-trained **RNet** to generate training data for **ONet**,I can get 19w+ pos samples.
* Since **MTCNN** is a Multi-task Network,we should pay attention to the format of training data.The format is:
 
  [path to image][cls_label][bbox_label][landmark_label]
  
  For pos sample,cls_label=1,bbox_label(calculate),landmark_label=[0,0,0,0,0,0,0,0,0,0].

  For part sample,cls_label=-1,bbox_label(calculate),landmark_label=[0,0,0,0,0,0,0,0,0,0].
  
  For landmark sample,cls_label=-2,bbox_label=[0,0,0,0],landmark_label(calculate).  
  
  For neg sample,cls_label=0,bbox_label=[0,0,0,0],landmark_label=[0,0,0,0,0,0,0,0,0,0].  

* Since the training data for landmark is less.I use transform,random rotate and random flip to conduct data augment(the result of landmark detection is not that good).

## Result

![result1.png](https://i.loli.net/2017/08/30/59a6b65b3f5e1.png)

![result2.png](https://i.loli.net/2017/08/30/59a6b6b4efcb1.png)

![result3.png](https://i.loli.net/2017/08/30/59a6b6f7c144d.png)

![reult4.png](https://i.loli.net/2017/08/30/59a6b72b38b09.png)

![result5.png](https://i.loli.net/2017/08/30/59a6b76445344.png)

![result6.png](https://i.loli.net/2017/08/30/59a6b79d5b9c7.png)

![result7.png](https://i.loli.net/2017/08/30/59a6b7d82b97c.png)

![result8.png](https://i.loli.net/2017/08/30/59a6b7ffad3e2.png)

![result9.png](https://i.loli.net/2017/08/30/59a6b843db715.png)

**Result on FDDB**
![result10.png](https://i.loli.net/2017/08/30/59a6b875f1792.png)

## License
MIT LICENSE

## References
1. Kaipeng Zhang, Zhanpeng Zhang, Zhifeng Li, Yu Qiao , " Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks," IEEE Signal Processing Letter
2. [MTCNN-MXNET](https://github.com/Seanlinx/mtcnn)
3. [MTCNN-CAFFE](https://github.com/CongWeilin/mtcnn-caffe)
4. [deep-landmark](https://github.com/luoyetx/deep-landmark)
