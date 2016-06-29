#!/usr/bin/env python
# -*- coding:utf-8 -*-
import argparse as ap
import numpy as np

class NoopActiovation(object):
    @staticmethod
    def forward(X):
        return X
    @staticmethod
    def backward(X):
        return X

class TanhActiovation(NoopActiovation):
    @staticmethod
    def forward(X):
        return np.tanh(X)
    @staticmethod
    def backward(X):
        return 1 - np.tanh(X)**2

class SoftmaxActiovation(NoopActiovation):
    @staticmethod
    def forward(X):
        exp_scores = np.exp(X)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs
    @staticmethod
    def backward(X):
        raise Exception("No Definition Exception")
        pass

def cross_entropy_loss(y, y_, bias):
    N = y.shape[0]
    return - (y * np.log(y_)).sum() / N + bias

class NNClassifier:
    def __init__(self, activation_funcs, loss_func,
                 n_hidden_layer_node):
        u"""
        """
        assert(len(activation_funcs) == 2)
        self.loss_func = loss_func
        self.activation_funcs = activation_funcs
        self.n_src_dim = 0
        self.n_dst_dim = 0
        self.n_hidden_layer_node = n_hidden_layer_node
        self.W1 = None
        self.W2 = None
        self.b1 = None
        self.b2 = None

    def predict(self, X):
        _, _, y_ = self.forward(X)
        return y_

    def forward(self, X):
        n_sample, n_dim = X.shape
        assert(n_dim == self.n_src_dim)
        z1 = X.dot(self.W1) + self.b1
        a1 = self.activation_funcs[0].forward(z1)
        z2 = a1.dot(self.W2) + self.b2
        a2 = self.activation_funcs[1].forward(z2)
        return a1, z1, a2

    def update(self, X, y, learning_rate, lambda_val):
        a1, z1, y_ = self.forward(X)
        # backpropagation
        delta3 = y_ - y
        delta2 = self.activation_funcs[0].backward(z1) * delta3.dot(self.W2.T)
        delta_W2 = a1.T.dot(delta3)
        delta_b2 = np.sum(delta3, axis=0, keepdims=True)
        delta_W1 = X.T.dot(delta2)
        delta_b1 = np.sum(delta2, axis=0, keepdims=True)
        # reguralization
        delta_W2 += lambda_val * self.W2
        delta_W1 += lambda_val * self.W1
        # update parameters
        self.W2 -= learning_rate * delta_W2
        self.W1 -= learning_rate * delta_W1
        self.b2 -= learning_rate * delta_b2
        self.b1 -= learning_rate * delta_b1
        return y_

    def fit(self, X, y, n_epoch=10000, learning_rate=0.01, lambda_val=0.001):
        n_sample, n_src_dim = X.shape
        n_sample, n_dst_dim = y.shape
        self.n_src_dim = n_src_dim
        self.n_dst_dim = n_dst_dim
        np.random.seed(0)
        print "n_sample:", n_sample
        print "n_src_dim:", self.n_src_dim
        print "n_dst_dim:", self.n_dst_dim
        print "n_hidden_layer_node:", self.n_hidden_layer_node
        self.W1 = np.random.randn(self.n_src_dim, self.n_hidden_layer_node) \
                  / np.sqrt(self.n_src_dim)
        self.W2 = np.random.randn(self.n_hidden_layer_node, n_dst_dim) \
                  / np.sqrt(self.n_hidden_layer_node)
        self.b1 = np.zeros((1, self.n_hidden_layer_node))
        self.b2 = np.zeros((1, self.n_dst_dim))
        for i in range(n_epoch):
            y_ = self.update(X, y, learning_rate, lambda_val)
            # print loss
            sum_regulatization = np.sum(np.square(self.W1)) + np.sum(np.square(self.W2))
            loss_regulatization = lambda_val * sum_regulatization / 2
            if i % 1000 == 0:
                print "--- epoch {} ---".format(i)
                print "loss:", self.loss_func(y, y_, loss_regulatization)


if __name__ == "__main__":
    description = """crawl input files"""
    parser = ap.ArgumentParser(description=description,
                               formatter_class=ap.RawDescriptionHelpFormatter)
    parser.add_argument('-f', '--from_term', required=True,
                        help='target term (from) like "20160601"')
    parser.add_argument('-t', '--to_term', default="",
                        help='target term (to), default: today')
    parser.add_argument('--dst', default="input",
                        help='output directory')
    parser.add_argument('--interval', default=60, type=int,
                        help='time interval for crawl')
    args = parser.parse_args()
    main(args.from_term, args.to_term, args.dst, args.interval)
