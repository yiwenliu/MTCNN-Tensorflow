#coding:utf-8
from mtcnn_model import P_Net
from train import train


def train_PNet(base_dir, prefix, end_epoch, display, lr):
    """
    train PNet
    :param base_dir: tfrecord path, '../prepare_data/imglists/PNet'
    :param prefix:'../data/MTCNN_model/PNet_landmark/PNet'
    :param end_epoch: 30
    :param display: 100
    :param lr: 0.01
    :return:
    """
    net_factory = P_Net
    train(net_factory,prefix, end_epoch, base_dir, display=display, base_lr=lr)

if __name__ == '__main__':
    #data path
    base_dir = '../prepare_data/imglists/PNet'
    model_name = 'MTCNN'
    #model_path = '../data/%s_model/PNet/PNet' % model_name
    #with landmark
    model_path = '../data/%s_model/PNet_landmark/PNet' % model_name
            
    prefix = model_path
    #一个epoch的意思就是"迭代次数*batch的数目 == 训练数据的个数"，就是一个epoch。
    end_epoch = 30
    display = 100
    lr = 0.01
    train_PNet(base_dir, prefix, end_epoch, display, lr)
