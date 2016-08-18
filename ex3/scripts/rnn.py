#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse as ap
import numpy as np
import tensorflow as tf


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
        self.dst_op = post_inference_func(rnn_dst_op)

        # loss
        self.loss_op = loss_func(self.dst_op, self.dst_ph)

        # accuracy
        self.accuracy_op = accuracy_func(self.dst_op, self.dst_ph)

    def train(self, batch_gen, optimizer, X_all, _y_all):
        print "train model"
        optimize_op = optimizer.minimize(self.loss_op)
        init_op = tf.initialize_all_variables()

        self.sess = tf.Session()
        self.sess.run(init_op)
        init_state_arr_all = np.zeros((X_all.shape[0], self.cell.state_size))
        eval_dict = {
                "src_ph:0": X_all, "dst_ph:0": _y_all,
                "init_state_ph:0": init_state_arr_all
        }
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
                train_loss = self.sess.run(self.loss_op, feed_dict=eval_dict)
                train_accuracy = self.sess.run(self.accuracy_op,
                                               feed_dict=eval_dict)
                print "loss:{}\taccuracy:{}".format(train_loss,
                                                    train_accuracy)

    def test(self, X, _y):
        print "test model"
        init_state_arr = np.zeros((X.shape[0], self.cell.state_size))
        test_dict = {
                "src_ph:0": X, "dst_ph:0": _y,
                "init_state_ph:0": init_state_arr
        }
        test_accuracy = self.sess.run(self.accuracy_op, feed_dict=test_dict)
        print "test accuracy:", test_accuracy
        return self.sess.run(self.dst_op, feed_dict=test_dict)

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


class BasicLSTM(RNNBase):
    def __init__(self, n_src_dim, len_seq, n_dst_dim,
                 n_hidden_node=80, forget_bias=0.8):
        self.n_hidden_node = n_hidden_node
        # cell = LSTMCell みたいな
        cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_node,
                                            forget_bias=forget_bias)
        super(BasicLSTM, self).__init__(
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
    n_sample, len_seq, n_dim = X.shape
    print "=== train data setting ==="
    print "n_sample  :", n_sample
    print "len_seq   :", n_dim
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
    model = BasicLSTM(n_src_dim=1, len_seq=len_seq, n_dst_dim=1)
    if mode == "train":
        batch_gen = batch_generator(X, _y, n_epoch, batch_size)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        model.train(batch_gen, optimizer)
        model.save(model_path)
    elif mode == "test":
        model.load(model_path)
        model.test(X, _y)


if __name__ == "__main__":
    description = """ basic LSTM """
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
