#!/usr/bin/env python
# -*- coding:utf-8 -*-
import glob
import argparse as ap
from chainer import Variable, FunctionSet, optimizers
import chainer.functions as F
from collections import defaultdict
import numpy as np
from skimage import io
import cPickle


def load_gender_images(image_dir):
    print "load gender images"
    _X = defaultdict(list)
    X = {}
    for gender in ["m", "f"]:
        image_paths = glob.glob("{}/{}/*.png".format(image_dir, gender))
        for p in image_paths:
            img = io.imread(p)
            _X[gender].append(img.reshape(-1))  # vectorize
    for gender in _X:
        X[gender] = np.array(_X[gender])
        print gender, X[gender].shape
    return X


def train(image_path, dst_path, batch_size=100, n_epoch=20, n_units=1000):
    u"""train NN gender predictor by chainer
    (n_hidden layer is fixed: 3)
    """
    Xdic = load_gender_images(image_path)
    for gender in Xdic:
        # データを[0, 1]に変換
        Xdic[gender] = Xdic[gender].astype(np.float32)
        Xdic[gender] /= 255
    # データを作成
    mX = Xdic["m"]
    mY = np.array([0]*mX.shape[0], dtype=np.int32)
    fX = Xdic["f"]
    fY = np.array([1]*fX.shape[0], dtype=np.int32)
    X = np.concatenate([mX, fX], axis=0)
    Y = np.concatenate([mY, fY], axis=0)
    # modelの定義
    dim = X.shape[1]
    model = FunctionSet(l1=F.Linear(dim, n_units),
            l2=F.Linear(n_units, n_units),
            l3=F.Linear(n_units, 2))
    def forward(x_data, y_data, train=True):
        x, t = Variable(x_data), Variable(y_data)
        h1 = F.dropout(F.relu(model.l1(x)),  train=train)
        h2 = F.dropout(F.relu(model.l2(h1)), train=train)
        y  = model.l3(h2)
        # 多クラス分類なので誤差関数としてソフトマックス関数の
        # 交差エントロピー関数を用いて、誤差を導出
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)
    # optimizereの定義
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    # シャッフルする
    for epoch in range(n_epoch):
        print "epoch:", epoch
        # データをシャッフルする
        N = X.shape[0]
        perm = np.random.permutation(N)
        X = X[perm]
        Y = Y[perm]
        # batchで学習
        sum_loss, sum_accuracy = 0, 0
        for i in xrange(0, N, batch_size):
            x_batch = X[i:i+batch_size]
            y_batch = Y[i:i+batch_size]
            # 勾配を初期化
            optimizer.zero_grads()
            # 順伝播させて誤差と精度を算出
            loss, acc = forward(x_batch, y_batch)
            # 誤差逆伝播で勾配を計算
            loss.backward()
            optimizer.update()

            sum_loss     += float(loss.data) * batch_size
            sum_accuracy += float(acc.data) * batch_size
        print 'train mean loss={}, accuracy={}'.format(sum_loss / N,
                                                       sum_accuracy / N)
    with open(dst_path, "w") as f:
        cPickle.dump(model, f, -1)


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
    args = parser.parse_args()
    train(args.src, args.dst, args.batch, args.epoch, args.unit)
