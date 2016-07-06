#!/usr/bin/env python
# -*- coding:utf-8 -*-

import NNClassifier as nn

# データの生成
import numpy as np
import sklearn.datasets

np.random.seed(0)
X, y = sklearn.datasets.make_moons(200, noise=0.20)
y = (np.array(y)[:,None]==np.arange(2))+0

classifier = nn.NNClassifier(activation_funcs=[nn.TanhActiovation, nn.SoftmaxActiovation],
                             loss_func=nn.cross_entropy_loss, n_hidden_layer_node=5)
classifier.fit(X, y, n_epoch=100000, learning_rate=0.0001, lambda_val=0.0001)
