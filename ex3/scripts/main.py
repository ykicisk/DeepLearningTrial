#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse as ap
import numpy as np
import pandas as pd
import tensorflow as tf
import rnn

# parameters
LEARNING_RATE = 0.05
N_HIDDEN_NODE = 50
FORGET_BIAS = 0.8


class N225Predictor(rnn.RNNBase):
    def __init__(self, n_src_dim, len_seq, n_dst_dim, ml_type,
                 n_hidden_node=80, forget_bias=0.8, dropout_rate=0.4,
                 class_weight=[1, 1]):
        self.n_hidden_node = n_hidden_node
        self.ml_type = ml_type
        self.class_weight = class_weight
        # cell = LSTMCell みたいな
        cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_node,
                                            forget_bias=forget_bias)
        super(N225Predictor, self).__init__(
            cell=cell,
            n_src_dim=n_src_dim, len_seq=len_seq, n_dst_dim=n_dst_dim,
            pre_inference_func=self.pre_inference_func,
            post_inference_func=self.post_inference_func,
            loss_func=self.loss_func, accuracy_func=self.accuracy_func,
            dropout_rate=dropout_rate
            )

    def pre_inference_func(self, src_ph):
        with tf.name_scope("pre_inference") as scope:
            W1_init = tf.random_normal([self.n_src_dim, self.n_hidden_node],
                                          stddev=0.1)
            W1 = tf.Variable(W1_init, name="W1")
            b1_init = tf.random_normal([self.n_hidden_node], stddev=0.1)
            b1 = tf.Variable(b1_init, name="b1")
            # (batch, len_seq, src_dim) => (len_seq, batch, src_dim)
            in1 = tf.transpose(src_ph, [1, 0, 2])
            # (len_seq, batch, src_dim) => (len_seq * batch, src_dim)
            in2 = tf.reshape(in1, [-1, self.n_src_dim])
            # (len_seq * batch, src_dim) * (src_dim, n_node)
            # => (len_seq * batch, n_node)
            # in3 = tf.matmul(in2, W1) + b1
            in3 = tf.nn.dropout(tf.nn.relu(tf.matmul(in2, W1) + b1), self.keep_prob)
            # (len_seq * batch, n_node) => len_seq * (batch, n_node)
            rnn_src_op = tf.split(0, self.len_seq, in3)
            return rnn_src_op

    def post_inference_func(self, rnn_dst_op):
        with tf.name_scope("pre_inference") as scope:
            # W2_init = tf.truncated_normal([self.n_hidden_node, self.n_dst_dim],
            #                               stddev=0.1)
            # W2 = tf.Variable(W2_init, name="W2")
            # b2_init = tf.truncated_normal([self.n_dst_dim], stddev=0.1)
            # b2 = tf.Variable(b2_init, name="b2")
            W2_1_init = tf.random_normal([self.n_hidden_node, self.n_hidden_node],
                                         stddev=0.1)
            W2_2_init = tf.random_normal([self.n_hidden_node, self.n_dst_dim],
                                         stddev=0.1)
            W2_1 = tf.Variable(W2_1_init, name="W2_1")
            W2_2 = tf.Variable(W2_2_init, name="W2_2")
            b2_1_init = tf.random_normal([self.n_hidden_node], stddev=0.1)
            b2_2_init = tf.random_normal([self.n_dst_dim], stddev=0.1)
            b2_1 = tf.Variable(b2_1_init, name="b2_1")
            b2_2 = tf.Variable(b2_2_init, name="b2_2")

            # t-1は予測には使わない
            # dst_op = tf.matmul(rnn_dst_op[-1], W2) + b2
            if self.ml_type == "regression":
                h1 = tf.nn.relu(tf.matmul(rnn_dst_op[-1], W2_1) + b2_1)
                dst_op = tf.nn.relu(tf.matmul(h1, W2_2) + b2_2)
                # dst_op = tf.nn.relu(tf.matmul(rnn_dst_op[-1], W2) + b2)
            elif self.ml_type == "classification":
                h1 = tf.nn.dropout(tf.nn.relu(tf.matmul(rnn_dst_op[-1], W2_1) + b2_1), self.keep_prob)
                dst_op = tf.nn.softmax(tf.matmul(h1, W2_2) + b2_2)

                # dst_op = tf.nn.softmax(tf.matmul(rnn_dst_op[-1], W2) + b2)
            return dst_op

    def loss_func(self, dst_op, dst_ph):
        with tf.name_scope("loss") as scope:
            if self.ml_type == "regression":
                square_error = tf.reduce_mean(tf.square(dst_op - dst_ph))
                loss = square_error
            elif self.ml_type == "classification":
                # cross entropy loss (- sum(y_ * log(y))) の定義
                # 演算子 * は行列の要素ごとの積っぽいreduction_indices == axis
                class_weight_c = tf.constant(self.class_weight)
                # class_weight_c = tf.constant(1.0 / np.array(self.class_weight), tf.float32)
                weighted_gt_dst = tf.mul(dst_ph, class_weight_c) # shape [batch_size, 2]
                loss = tf.reduce_mean(-tf.reduce_sum(weighted_gt_dst * tf.log(dst_op),
                                                     reduction_indices=[1]))
            return loss

    def accuracy_func(self, dst_op, dst_ph):
        with tf.name_scope("accuracy") as scope:
            if self.ml_type == "regression":
                sq_err = tf.square(dst_op - dst_ph)
                # tensor005 = tf.constant(2*2, dtype=tf.float32)
                tensor005 = tf.constant(0.1*0.1, dtype=tf.float32)
                # 1はaxis
                correct_prediction = tf.less(sq_err, tensor005)
                # castはbool列をfloat列にcast。
                accuracy_op = tf.reduce_mean(tf.cast(correct_prediction,
                                                     tf.float32))
            elif self.ml_type == "classification":
                correct = tf.equal(tf.argmax(dst_op, 1), tf.argmax(dst_ph, 1))
                # castはbool列をfloat列にcastしている。
                accuracy_op = tf.reduce_mean(tf.cast(correct, tf.float32))
            return accuracy_op


def create_learning_data(df, date_range):
    ex_features = ["DJI", "IXIC", "N100", "EURJPY", "USDJPY"]
    targets = filter(lambda x: x not in ex_features, df.columns)
    _ys = []
    Xs = []
    for tgt in targets:
        _df = df[[tgt] + ex_features].dropna()  # nanは雑に削除
        tgt_idx = 0
        features = _df.as_matrix()
        diff = (features[1:]-features[:-1])/features[:-1] * 100
        _y = diff[date_range:, [tgt_idx]] * 100
        n = diff.shape[0]
        x_list = []
        # print "----"
        for i in range(date_range):
            x = diff[i:(n-date_range+i), :]
            if x.shape[0] > 0:
                x_list.append(x)
                # print x.shape
        if not x_list:
            continue
        X = np.transpose(np.dstack(x_list), axes=(0, 2, 1))
        if _y.shape[0] == 0:
            continue
        _ys.append(_y)
        Xs.append(X)
        # print X.shape, _y.shape
    # print X[-1]
    # print _y[-1]
    X = np.concatenate(Xs, axis=0)
    _y = np.concatenate(_ys, axis=0)
    print X.shape, _y.shape
    return X, _y


def cvt2classification_label(y):
    # (n_data, 1)
    ret = np.concatenate([y >= 0, y < 0], axis=1).astype(np.float32)
    return ret


def main(mode, model_path, ml_type, data_path, date_range,
         boundary_date, n_epoch, batch_size, indivisual_path):
    print "load learning data:", data_path
    df = pd.read_csv(data_path, compression="gzip", index_col=0,
                     parse_dates=True)
    boundary_pdts = pd.Timestamp(boundary_date)
    indivisual_df = None
    if indivisual_path is not None:
        print "load indivisual stock data:", indivisual_path
        indivisual_df = pd.read_csv(indivisual_path, compression="gzip",
                                    index_col=0, parse_dates=True)
        df = df.join(indivisual_df, how="inner")
    trainX, train_y = create_learning_data(df[:boundary_pdts], date_range)
    testX, test_y = create_learning_data(df[boundary_pdts:], date_range)

    class_weight = [1.0, 1.0]
    if ml_type == "classification":
        rasio1 = (train_y >= 0).astype(np.float32).sum() / train_y.shape[0]
        rasio0 = 1.0 - rasio1
        class_weight = [rasio0, rasio1]
        # class_weight = [rasio1, rasio0]
        # class_weight = [rasio0, rasio1]
        print class_weight
        train_y = cvt2classification_label(train_y)
        test_y = cvt2classification_label(test_y)
    print trainX.shape, train_y.shape
    print testX.shape, test_y.shape

    # n_data * seq * dim
    n_src_dim = trainX.shape[2]
    # n_data * dim
    n_dst_dim = train_y.shape[1]
    model = N225Predictor(n_src_dim=n_src_dim, len_seq=date_range,
                          n_dst_dim=n_dst_dim, ml_type=ml_type,
                          n_hidden_node=N_HIDDEN_NODE,
                          forget_bias=FORGET_BIAS,
                          class_weight=class_weight)
    if mode == "train":
        batch_gen = rnn.batch_generator(trainX, train_y, n_epoch, batch_size)
        optimizer = tf.train \
                      .GradientDescentOptimizer(learning_rate=LEARNING_RATE)
        model.train(batch_gen, optimizer, trainX, train_y)
        model.save(model_path)
    elif mode == "test":
        model.load(model_path)
        pred_y = model.test(testX, test_y)
        # print np.concatenate([test_y, pred_y], axis=1)


if __name__ == "__main__":
    description = """ predict Nikkei 225 """
    parser = ap.ArgumentParser(description=description,
                               formatter_class=ap.RawDescriptionHelpFormatter)
    parser.add_argument('mode', choices=["train", "test"],
                        help='train mode')
    parser.add_argument('data', help='learning data')
    parser.add_argument("model", help='model path')
    parser.add_argument("type", choices=["classification", "regression"],
                        help='type of prediction')
    parser.add_argument('-r', "--range", default=20, type=int,
                        help='range of training data')
    parser.add_argument('-d', "--boundary_date", default="20160101",
                        help='boundary date between train and test')
    parser.add_argument('-e', "--epoch", default=10, type=int,
                        help='num of epoch (use in only train mode)')
    parser.add_argument('-b', "--batch", default=300, type=int,
                        help='batch size (use in only train mode)')
    parser.add_argument('-i', "--indivisual", default=None,
                        help='indivisual stock prediction')
    args = parser.parse_args()
    main(args.mode, args.model, args.type, args.data, args.range,
         args.boundary_date, args.epoch, args.batch, args.indivisual)
