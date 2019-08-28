#coding:utf-8
#
# Download dataset from http://www.cs.toronto.edu/~kriz/cifar.html,
# The code is for CIFAR-10 binary version. 
#

import os
import sys
import random
import numpy as np
import PIL.Image as Image
from torch.utils.data import Dataset


def parse_batch_bin(batch_idx, batch_path, save_data_path, file_size):
    fi = open(batch_path)
    raw_data = fi.read()
    size = len(raw_data)

    one_batch_num = 10000
    for j in range(one_batch_num):
        idx = batch_idx * one_batch_num + j
        data = raw_data[j*file_size:j*file_size+file_size]
        data = [data[x] for x in range(len(data))]
        data = np.array(data)
        data.dtype = np.uint8
        data.tofile(os.path.join(save_data_path, "%d.raw" %(idx)))
    fi.close()   
    print("%s processed!" %(batch_path))


def preprocess_cifar10(dataset_dir):
    train_data_path = os.path.join(dataset_dir, "train")
    test_data_path  = os.path.join(dataset_dir, "test")

    if (not os.path.exists(train_data_path)):
        os.mkdir(train_data_path)
    if (not os.path.exists(test_data_path)):
        os.mkdir(test_data_path)

    for i in range(1,6):
        batch_name = "data_batch_%d.bin" %(i)
        batch_path = os.path.join(dataset_dir, batch_name)
        parse_batch_bin(i-1, batch_path, train_data_path, 3073)

    batch_name = "test_batch.bin"
    batch_path = os.path.join(dataset_dir, batch_name)
    parse_batch_bin(0, batch_path, test_data_path, 3073)



class Cifar10(Dataset):

    def __init__(self, dir_path, is_labeled, is_vali, train_ratio, transform_func):
        super(Cifar10, self).__init__()
        
        raw_list = os.listdir(dir_path)
        file_list = []
        for item in raw_list:
            if ('.raw' in item):
                file_list.append(item)

        self.is_labeled_ = is_labeled
        self.is_vali_    = is_vali
        self.transform_  = transform_func

        self.file_list_ = [os.path.join(dir_path, item) for item in file_list]

        # split train/val
        random.seed(0)
        random.shuffle(self.file_list_)
        ratio = train_ratio
        cut = int(ratio * len(self.file_list_))
        if (self.is_vali_):
            self.file_list_ = self.file_list_[cut:]
        else:
            self.file_list_ = self.file_list_[:cut]


    def __getitem__(self, index):
        file = self.file_list_[index]
        data = np.fromfile(file, dtype=np.uint8)
        data.dtype = np.uint8

        if (self.is_labeled_):
            label = int(data[0])
            img   = data[1:]
        else:
            img = data

        img.shape = (3,32,32)
        img = np.transpose(img, (1,2,0))

        # First, convert np.array to PIL.Image
        img = Image.fromarray(img)
        # Then, we can use torchvision.transforms
        img = self.transform_(img)

        if (self.is_labeled_):
            return (img, label)
        else:
            return (img, file)


    def __len__(self):
        return len(self.file_list_)




# preprocess_cifar10('/home/chen/dataset/cifar10/cifar-10-batches-bin/')