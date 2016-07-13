#!/usr/bin/env python
# -*- coding:utf-8 -*-
import glob
import sys
import argparse as ap
from chainer import Variable, FunctionSet, optimizers
import chainer.functions as F
from collections import defaultdict
import numpy as np
from skimage import io
import cPickle


def load_gender_dataset(image_dir, is_2d=False):
    u"""return {0: np.ndarray, 1: np.ndarray}"""
    print "load gender images"
    _X = defaultdict(list)
    X = {}
    for label_no, gender in enumerate(["m", "f"]):
        image_paths = glob.glob("{}/{}/*.png".format(image_dir, gender))
        for p in image_paths:
            img = io.imread(p)
            if is_2d:
                # n_sample * channel * height * width
                _X[label_no].append([img])
            else:
                _X[label_no].append(img.reshape(-1))  # vectorized
    for label_no in _X:
        X[label_no] = np.array(_X[label_no], dtype=np.float32) / 255
    return X


def merge_dataset(dataset):
    Xs = []
    Ys = []
    for label, data in dataset.items():
        Xs.append(data)
        Ys.append(np.array([label]*data.shape[0], dtype=np.int32))
    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    return X, Y


class ModelBase(object):
    def __init__(self):
        self.model = None

    def forward(self, x_data, y_data, train=True):
        raise Exception("no implementation exception")


class NN3_Model(ModelBase):
    def __init__(self, input_dim=748, n_units=1000):
        super(NN3_Model, self).__init__()
        self.n_units = n_units
        self.model = FunctionSet(l1=F.Linear(input_dim, n_units),
                                 l2=F.Linear(n_units, n_units),
                                 l3=F.Linear(n_units, 2))

    def forward(self, x_data, y_data, train=True):
        u"""return loss, accuracy"""
        x, t = Variable(x_data), Variable(y_data)
        h1 = F.dropout(F.relu(self.model.l1(x)),  train=train)
        h2 = F.dropout(F.relu(self.model.l2(h1)), train=train)
        y  = self.model.l3(h2)
        # 多クラス分類なので誤差関数としてソフトマックス関数の
        # 交差エントロピー関数を用いて、誤差を導出。最低でもlossは必要
        return {
                "loss": F.softmax_cross_entropy(y, t),
                "accuracy": F.accuracy(y, t)
                }


class CNN3_Model(ModelBase):
    u"""see: http://aidiary.hatenablog.com/entry/20151007/1444223445"""
    def __init__(self, input_size=32):
        super(CNN3_Model, self).__init__()
        # F.Convolution2D(in_channel, out_channel, filter_size)
        self.model = FunctionSet(  # 1*32*32 -(conv)-> 20*28*28 -(pool)-> 20*14*14
                                 conv1=F.Convolution2D(1, 20, 5),
                                   # 20*14*14 -(conv)-> 50*10*10 -(pool)-> 50*5*5=1250
                                 conv2=F.Convolution2D(20, 50, 5),
                                 l1=F.Linear(1250, 300),
                                 l2=F.Linear(300, 2))

    def forward(self, x_data, y_data, train=True):
        u"""return loss, accuracy"""
        x, t = Variable(x_data), Variable(y_data)
        h1 = F.max_pooling_2d(F.relu(self.model.conv1(x)), 2)
        h2 = F.max_pooling_2d(F.relu(self.model.conv2(h1)), 2)
        h3 = F.dropout(F.relu(self.model.l1(h2)), train=train)
        y  = self.model.l2(h3)
        # 多クラス分類なので誤差関数としてソフトマックス関数の
        # 交差エントロピー関数を用いて、誤差を導出。最低でもlossは必要
        return {
                "loss": F.softmax_cross_entropy(y, t),
                "accuracy": F.accuracy(y, t)
                }


class ChainerPredictor(object):
    def __init__(self, n_epoch=20, batch_size=100):
        self.model = None
        self.n_epoch = n_epoch
        self.batch_size = batch_size

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        self.optimizer.setup(self.model.model)

    def train(self, _X, _Y):
        print "train NN model"
        for epoch in range(self.n_epoch):
            print "epoch:", epoch
            # データをシャッフルする
            N = _X.shape[0]
            perm = np.random.permutation(N)
            X = _X[perm]
            Y = _Y[perm]
            # batchで学習
            sum_values = defaultdict(float)
            for i in xrange(0, N, self.batch_size):
                x_batch = X[i:i+self.batch_size]
                y_batch = Y[i:i+self.batch_size]
                # 勾配を初期化
                self.optimizer.zero_grads()
                # 順伝播させて誤差と精度を算出
                result = self.model.forward(x_batch, y_batch)
                result["loss"].backward()
                self.optimizer.update()
                for key, val in result.items():
                    sum_values[key] += (val.data) * self.batch_size
            sys.stdout.write("train: ")
            for key, val in sum_values.items():
                sys.stdout.write("{}:{}, ".format(key, val/ N))
            sys.stdout.write("\n")

    def test(self, X, Y, batch_size=100):
        # 順伝播させて誤差と精度を算出
        N = X.shape[0]
        sum_values = defaultdict(float)
        for i in xrange(0, N, batch_size):
            x_batch = X[i:i+batch_size]
            y_batch = Y[i:i+batch_size]
            result = self.model.forward(x_batch, y_batch, train=False)
            for key, val in result.items():
                sum_values[key] += (val.data) * batch_size
        sys.stdout.write("test: ")
        for key, val in sum_values.items():
            sys.stdout.write("{}:{}, ".format(key, val/ N))
        sys.stdout.write("\n")

    def save(self, dst_path):
        with open(dst_path, "w") as f:
            cPickle.dump(self, f, -1)


def train(image_path, dst_path, batch_size=100, n_epoch=20, n_units=1000,
          CNN_flag=False):
    u"""train NN gender predictor by chainer
    (n_hidden layer is fixed: 3)
    """
    dataset = load_gender_dataset(image_path, is_2d=CNN_flag)
    for key, arr in dataset.items():
        print key, arr.shape
    X, Y = merge_dataset(dataset)
    # Predictorに突っ込む
    predictor = ChainerPredictor(n_epoch=n_epoch, batch_size=batch_size)
    if CNN_flag:
        predictor.model = CNN3_Model(input_size=32)
    else:
        dim = X.shape[1]
        predictor.model = NN3_Model(input_dim=dim, n_units=n_units)
    predictor.set_optimizer(optimizers.Adam())
    # trainする
    predictor.train(X, Y)
    # 保存
    predictor.save(dst_path)
    return predictor


if __name__ == "__main__":
    description = """
    """
    parser = ap.ArgumentParser(description=description,
                               formatter_class=ap.RawDescriptionHelpFormatter)
    parser.add_argument('-s', '--src', required=True,
                        help='input image directory')
    parser.add_argument('-d', '--dst', required=True,
                        help='output pickle path')
    parser.add_argument('-b', '--batch', default=100, type=int,
                        help='batch size')
    parser.add_argument('-e', '--epoch', default=20, type=int,
                        help='num of epoch')
    parser.add_argument('-u', '--unit', default=1000, type=int,
                        help='unit size of hidden layer')
    parser.add_argument('--CNN', default=False, action="store_true",
                        help='use CNN')
    args = parser.parse_args()
    train(args.src, args.dst, args.batch, args.epoch, args.unit, args.CNN)
