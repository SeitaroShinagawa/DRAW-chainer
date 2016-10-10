import os
import numpy as np
import json
from PIL import Image
from collections import defaultdict
from sklearn.datasets import fetch_mldata

#tobe staticmethod
def clip(a):
  return 0 if a<0 else (255 if a>255 else a)

def img_fromarray(im):
  im = im*255
  im = np.vectorize(clip)(im).astype(np.uint8)
  #im=im.transpose(1,2,0)
  img=Image.fromarray(im)
  return img

def Binarize(x):
    return (np.sign(x-0.5)+1)*0.5

class MNIST:
    """
    MNIST handwritten recognition data
    train   60000x784 -> 60000x28x28
    test    60000x784 -> 10000x28x28
    train_label 60000x10
    y_label     10000x10
    """
    def __init__(self,binarize=False):
        mnist = fetch_mldata('MNIST Original')
        x_all = mnist.data.astype(np.float32)/255
        y_all = mnist.target.astype(np.int32)
        x_train, x_test = np.split(x_all, [60000])
        y_train, y_test = np.split(y_all, [60000])
        if binarize==True:
            x_train = Binarize(x_train)
            x_test = Binarize(x_test)
        x_train = np.array(x_train).reshape(60000,28,28)
        x_test = np.array(x_test).reshape(10000,28,28)

        self.train = x_train
        self.test = x_test
        self.train_label = y_train
        self.test_label = y_test
        self.train_size = len(x_train)
        self.test_size = len(x_test)
        self.C=1
        self.width=28
        self.height=28

    def gen_train(self,batchsize,Random=True):
        if Random:
            indexes = np.random.permutation(60000)
        else:
            indexes = np.arange(60000)
        num = 0
        while batchsize*num < len(indexes):
            indexparts = indexes[batchsize*num:batchsize*(num+1)]
            image_batch = np.asarray([[self.train[x]] for x in indexparts],dtype=np.float32)
            label_batch = np.asarray([[self.train_label[x]] for x in indexparts],dtype=np.float32)
            yield image_batch ,label_batch
            num += 1 
                
    def gen_test(self,batchsize):
        """
        Attention! minibatch calculation doesn't generate exact result.
        If you need exact result, make batchsize=1.
        """
        num = 0
        while batchsize*num < 10000:
            image_batch = np.asarray(self.test[batchsize*num:batchsize*(num+1)],dtype=np.float32)
            label_batch = np.asarray(self.test_label[batchsize*num:batchsize*(num+1)],dtype=np.float32)
            yield image_batch,label_batch
            num += 1 
        
        
         
