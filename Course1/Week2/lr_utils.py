import numpy as np
import h5py
    
 # 加载猫分类数据集
def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features  训练集特征
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels 训练集标签

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features 测试集特征
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels 测试集标签

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes 类别名称
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes