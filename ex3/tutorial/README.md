# ex3

## TensorflowでMNIST

### 準備

```sh
$ python
>>> from tensorflow.examples.tutorials.mnist import input_data
>>> mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
```

### 実行

```sh
$ ./mnist_tutorial.py
```

### 参考
 
http://qiita.com/TomokIshii/items/92a266b805d7eee02b1d

## TensorflowでRNN(LSTM)

[こちら](http://qiita.com/yukiB/items/f6314d2861fc8d9b739f)を参考に、
LSTMでsumをとるRNNを実装。

### 学習

```sh
$ ./rnn_tutorial.py train tmp.model
WARNING:tensorflow:<tensorflow.python.ops.rnn_cell.BasicLSTMCell object at 0x111b8b750>: Using a concatenated state is slower and will soon be deprecated.  Use state_is_tuple=True.
train model
=== train data setting ===
n_sample  : 50000
n_dim     : 10
n_epoch   : 200
batch_size: 5000
--------------------------
epoch: 0
loss:3.96867227554      accuracy:0.0
epoch: 1
loss:2.18908286095      accuracy:0.248799994588
epoch: 2
loss:2.13058781624      accuracy:0.248799994588
...
epoch: 99
loss:0.0384677797556    accuracy:0.806999981403
epoch: 100
loss:0.0379077754915    accuracy:0.816799998283
...
epoch: 198
loss:0.0102835884318    accuracy:0.986599981785
epoch: 199
loss:0.0101649425924    accuracy:0.987600028515
==========================
save model: tmp.model
```

### テスト

```sh
$ ./rnn_tutorial.py test tmp.ckpt
WARNING:tensorflow:<tensorflow.python.ops.rnn_cell.BasicLSTMCell object at 0x10dacd810>: Using a concatenated state is slower and will soon be deprecated.  Use state_is_tuple=True.
load model: tmp.ckpt
test model
test accuracy: 0.98706
```

### 参考


