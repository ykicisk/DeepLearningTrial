#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse as ap
import numpy as np
import tensorflow as tf


def softmax_regression(batch_size=100, n_iter=1000):
    # 最初に、いくつか変数・定数の定義を書く→train!!

    # placeholder: Tensorflowが計算を始めるときの入力が入るもの。固定。
    # Tensorflowでは、pythonでグラフなどを定義して計算自体はpython外でやる。
    x = tf.placeholder(tf.float32, [None, 784])  # None means any dimension
    y_ = tf.placeholder(tf.float32, [None, 10])

    # Variableは計算時可変となる変数。同様に定義する。
    # y = softmax((xW) + b)
    # Wxとしないのは、multiple inputに対応するため、全体的に転置をかけている
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    # 次元が足りないものは拡張されるようです。
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # cross entropy loss (- sum(y_ * log(y))) の定義
    # 演算子 * は行列の要素ごとの積っぽい, reduction_indices == axis
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),
                                                  reduction_indices=[1]))

    # 学習方法の定義(やだ・・・カッコいい・・)
    train_step = tf.train.GradientDescentOptimizer(learning_rate=0.5)\
                         .minimize(cross_entropy)

    # 全部初期化
    init = tf.initialize_all_opiables()

    # 学習するセッションをrunしていく（実際に実行）
    session = tf.InteractiveSession()
    # session = tf.Session()
    session.run(init)
    for i in range(n_iter):
        # numpy array
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        session.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # 1はaxis
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # castはbool列をfloat列にcastしている。
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    acc_val = session.run(accuracy, feed_dict={x: mnist.test.images,
                                               y_: mnist.test.labels})
    print acc_val


def shallow_learning(batch_size=100, n_iter=3000, n_hidden_unit=100):
    u"""三層NNで学習してみる"""
    x = tf.placeholder(tf.float32, [None, 784])  # None means any dimension
    y_ = tf.placeholder(tf.float32, [None, 10])
    # W1 = tf.Variable(tf.zeros([784, n_hidden_unit])) → OUT!!!
    # b1 = tf.Variable(tf.zeros([n_hidden_unit]))
    # W2 = tf.Variable(tf.zeros([n_hidden_unit, 10]))
    # b2 = tf.Variable(tf.zeros([10]))
    W1 = tf.Variable(tf.random_normal([784, n_hidden_unit], mean=0.0, stddev=0.05))
    b1 = tf.Variable(tf.zeros([n_hidden_unit]))
    W2 = tf.Variable(tf.random_normal([n_hidden_unit, 10], mean=0.0, stddev=0.05))
    b2 = tf.Variable(tf.zeros([10]))

    # 隠れ層を追加
    h1 = tf.nn.relu(tf.matmul(x, W1) + b1)
    y = tf.nn.softmax(tf.matmul(h1, W2) + b2)

    # cross entropy loss (- sum(y_ * log(y))) の定義
    # 演算子 * は行列の要素ごとの積っぽい, reduction_indices == axis
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),
                                                  reduction_indices=[1]))
    L2_loss = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2)
    L2_lambda = 0.01
    loss = cross_entropy + L2_lambda * L2_loss

    # 学習方法の定義(やだ・・・カッコいい・・)
    train_step = tf.train.GradientDescentOptimizer(learning_rate=0.05)\
                         .minimize(loss)

    # 全部初期化
    init = tf.initialize_all_opiables()

    # 1はaxis
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # castはbool列をfloat列にcastしている。
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 学習するセッションをrunしていく（実際に実行）
    session = tf.InteractiveSession()
    # session = tf.Session()
    session.run(init)
    for i in range(n_iter):
        # numpy array
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        session.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        if i % 100 == 0:
            feed_dict = {x: mnist.test.images, y_: mnist.test.labels}
            acc_val = session.run(accuracy, feed_dict=feed_dict)
            loss_val = session.run(loss, feed_dict=feed_dict)
            print "iter:", i, ". loss:", loss_val, "accuracy:", acc_val

    acc_val = session.run(accuracy, feed_dict={x: mnist.test.images,
                                               y_: mnist.test.labels})
    print acc_val


class RNNBase(object):
    u"""RNNのベースクラス"""
    def __init__(self, cell, n_src_dim, len_seq, n_dst_dim,
                 pre_inference_func, post_inference_func,
                 loss_func, accuracy_func):
        u"""
        # RNNの入力(sequenceが同じ長さの場合)
        * input: tf variable len_seq * (batch, n_rnn_node)
        * initial state: tf placeholder (batch, n_rnn_state)

        # LSTMの出力
        * output, state: ともに同じ
        (batch, n_node)

        # pre_inference: RNNの前段
        phX(batch, len_seq, input_dim) => len_seq * (batch, n_node)

        # post_inference: RNNの後段
        state(batch, n_node) => y(batch, output_dim)
        """
        self.cell = cell
        self.n_src_dim = n_src_dim
        self.len_seq = len_seq
        self.n_dst_dim = n_dst_dim
        # place holders
        self.src_ph = tf.placeholder(tf.float32, [None, len_seq, n_src_dim],
                                     name="src_ph")
        self.dst_ph = tf.placeholder(tf.float32, [None, n_dst_dim],
                                     name="dst_ph")
        # state_sizeはRNNCellによって違ったりする。
        self.init_state_ph = tf.placeholder(tf.float32,
                                            [None, self.cell.state_size],
                                            name="init_state_ph")

        # inference
        rnn_src_op = pre_inference_func(self.src_ph)
        rnn_dst_op, _ = tf.nn.rnn(self.cell, rnn_src_op,
                                  initial_state=self.init_state_ph,
                                  dtype=tf.float32)
        dst_op = post_inference_func(rnn_dst_op)

        # loss
        self.loss_op = loss_func(dst_op, self.dst_ph)

        # accuracy
        self.accuracy_op = accuracy_func(dst_op, self.dst_ph)

    def train(self, batch_gen, optimizer):
        print "train model"
        optimize_op = optimizer.minimize(self.loss_op)
        init_op = tf.initialize_all_variables()

        self.sess = tf.Session()
        self.sess.run(init_op)
        for X, _y, is_EOE in batch_gen:
            batch_size = X.shape[0]
            init_state_arr = np.zeros((batch_size, self.cell.state_size))
            # why need ":0" ??
            # see: https://github.com/tensorflow/tensorflow/issues/3378
            train_dict = {
                    "src_ph:0": X, "dst_ph:0": _y,
                    "init_state_ph:0": init_state_arr
            }
            self.sess.run(optimize_op, feed_dict=train_dict)
            if is_EOE:
                train_loss = self.sess.run(self.loss_op, feed_dict=train_dict)
                train_accuracy = self.sess.run(self.accuracy_op,
                                               feed_dict=train_dict)
                print "loss:{}\taccuracy:{}".format(train_loss,
                                                    train_accuracy)

    def test(self, X, _y):
        print "test model"
        init_state_arr = np.zeros((X.shape[0], self.cell.state_size))
        test_dict = {
                "src_ph:0": X, "dst_ph:0": _y,
                "init_state_ph:0": init_state_arr
        }
        test_accuracy = self.sess.run(self.accuracy_op,
                                      feed_dict=test_dict)
        print "test accuracy:", test_accuracy

    def save(self, model_path):
        print "save model:", model_path
        if self.sess is None:
            raise Exception("No session exception")
        saver = tf.train.Saver()
        saver.save(self.sess, model_path)

    def load(self, model_path):
        print "load model:", model_path
        saver = tf.train.Saver()
        self.sess = tf.Session()
        saver.restore(self.sess, model_path)


class OneValueLSTM(RNNBase):
    def __init__(self, n_src_dim, len_seq, n_dst_dim,
                 n_hidden_node=80, forget_bias=0.8):
        self.n_hidden_node = n_hidden_node
        # cell = LSTMCell みたいな
        cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_node,
                                            forget_bias=forget_bias)
        super(OneValueLSTM, self).__init__(
            cell=cell,
            n_src_dim=n_src_dim, len_seq=len_seq, n_dst_dim=n_dst_dim,
            pre_inference_func=self.pre_inference_func,
            post_inference_func=self.post_inference_func,
            loss_func=self.loss_func, accuracy_func=self.accuracy_func
            )

    def pre_inference_func(self, src_ph):
        with tf.name_scope("pre_inference") as scope:
            W1_init = tf.truncated_normal([self.n_src_dim, self.n_hidden_node],
                                          stddev=0.1)
            W1 = tf.Variable(W1_init, name="W1")
            b1_init = tf.truncated_normal([self.n_hidden_node], stddev=0.1)
            b1 = tf.Variable(b1_init, name="b1")
            # (batch, len_seq, src_dim) => (len_seq, batch, src_dim)
            in1 = tf.transpose(src_ph, [1, 0, 2])
            # (len_seq, batch, src_dim) => (len_seq * batch, src_dim)
            in2 = tf.reshape(in1, [-1, self.n_src_dim])
            # (len_seq * batch, src_dim) * (src_dim, n_node)
            # => (len_seq * batch, n_node)
            in3 = tf.matmul(in2, W1) + b1
            # (len_seq * batch, n_node) => len_seq * (batch, n_node)
            rnn_src_op = tf.split(0, self.len_seq, in3)
            return rnn_src_op

    def post_inference_func(self, rnn_dst_op):
        with tf.name_scope("pre_inference") as scope:
            W2_init = tf.truncated_normal([self.n_hidden_node, self.n_dst_dim],
                                          stddev=0.1)
            W2 = tf.Variable(W2_init, name="W2")
            b2_init = tf.truncated_normal([self.n_dst_dim], stddev=0.1)
            b2 = tf.Variable(b2_init, name="b2")

            # t-1は予測には使わない
            dst_op = tf.matmul(rnn_dst_op[-1], W2) + b2
            return dst_op

    def loss_func(self, dst_op, dst_ph):
        with tf.name_scope("loss") as scope:
            square_error = tf.reduce_mean(tf.square(dst_op - dst_ph))
            loss_op = square_error
            return loss_op

    def accuracy_func(self, dst_op, dst_ph):
        with tf.name_scope("accuracy") as scope:
            sq_err = tf.square(dst_op - dst_ph)
            tensor005 = tf.constant(0.05, dtype=tf.float32)
            # 1はaxis
            correct_prediction = tf.less(sq_err, tensor005)
            # castはbool列をfloat列にcast。
            accuracy_op = tf.reduce_mean(tf.cast(correct_prediction,
                                                 tf.float32))
            return accuracy_op


def batch_generator(X, _y, n_epoch, batch_size):
    n_sample, n_dim = X.shape[0:2]
    print "=== train data setting ==="
    print "n_sample  :", n_sample
    print "n_dim     :", n_dim
    print "n_epoch   :", n_epoch
    print "batch_size:", batch_size
    print "--------------------------"
    for i_epoch in range(n_epoch):
        print "epoch:", i_epoch
        indices = np.arange(n_sample)
        np.random.shuffle(indices)
        is_EOE = False
        for idxL in range(0, n_sample, batch_size):
            idxR = min(idxL+batch_size, n_sample)
            bX = X[idxL:idxR, :, :]
            b_y = _y[idxL:idxR, :]
            if idxR == n_sample:
                is_EOE = True
            yield bX, b_y, is_EOE
    print "=========================="


def main(mode, model_path, n_sample, len_seq, n_epoch, batch_size):
    # 0 or 1
    X = np.random.randint(0, 2, (n_sample, len_seq, 1)).astype(np.float32)
    _y = np.sum(X, axis=1).reshape((X.shape[0], 1))
    model = OneValueLSTM(n_src_dim=1, len_seq=len_seq, n_dst_dim=1)
    if mode == "train":
        batch_gen = batch_generator(X, _y, n_epoch, batch_size)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        model.train(batch_gen, optimizer)
        model.save(model_path)
    elif mode == "test":
        model.load(model_path)
        model.test(X, _y)


if __name__ == "__main__":
    description = """download haiku data
    from http://www.weblio.jp/category/hobby/gndhh"""
    parser = ap.ArgumentParser(description=description,
                               formatter_class=ap.RawDescriptionHelpFormatter)
    parser.add_argument('mode', choices=["train", "test"],
                        help='train mode')
    parser.add_argument("model", help='model path')
    parser.add_argument('-n', "--n_sample", default=50000, type=int,
                        help='num of data')
    parser.add_argument('-s', "--seq", default=10, type=int,
                        help='length of sequence')
    parser.add_argument('-e', "--epoch", default=200, type=int,
                        help='num of epoch (use in only train mode)')
    parser.add_argument('-b', "--batch", default=5000, type=int,
                        help='batch size (use in only train mode)')
    args = parser.parse_args()
    main(args.mode, args.model, args.n_sample, args.seq,
         args.epoch, args.batch)
