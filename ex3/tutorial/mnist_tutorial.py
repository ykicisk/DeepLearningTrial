#!/usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


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
    init = tf.initialize_all_variables()

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
    init = tf.initialize_all_variables()

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


def main():
    # softmax_regression()
    shallow_learning(n_hidden_unit=784)


if __name__ == "__main__":
    main()
