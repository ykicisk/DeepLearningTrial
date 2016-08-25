# ex3

## Tensorflowで日経平均予測

### クラス重みを考慮

* 学習データの (正例:負例) = 54:46くらいだった
* これを考慮したロス関数で再実験

```sh
$ ./scripts/main.py train data/rnn_train.csv.gz clt.ckpt classification -e 300 -r 10
WARNING:tensorflow:<tensorflow.python.ops.rnn_cell.BasicLSTMCell object at 0x110944890>: Using a concatenated state is slower and will soon be deprecated.  Use state_is_tuple=True.
load learning data: data/rnn_train.csv.gz
[0.4614295824486907, 0.5385704175513093]
train model
=== train data setting ===
n_sample  : 1413
len_seq   : 6
n_epoch   : 300
batch_size: 300
--------------------------
epoch: 0
loss:0.344795644283	accuracy:0.461429595947
epoch: 1
loss:0.344662576914	accuracy:0.461429595947
...
epoch: 148
loss:0.340082645416	accuracy:0.627742409706
epoch: 149
loss:0.340045779943	accuracy:0.627034664154
...
epoch: 298
loss:0.322621673346	accuracy:0.658881783485
epoch: 299
loss:0.322450011969	accuracy:0.661004960537
==========================
save model: clt.ckpt
```

**学習がすすんだ！！**

テスト

```sh
$ ./scripts/main.py test data/rnn_train.csv.gz clt.ckpt classification
WARNING:tensorflow:<tensorflow.python.ops.rnn_cell.BasicLSTMCell object at 0x110944890>: Using a concatenated state is slower and will soon be deprecated.  Use state_is_tuple=True.
load learning data: data/rnn_train.csv.gz
[0.4614295824486907, 0.5385704175513093]
load model: clt.ckpt
test model
test accuracy: 0.559702
```

学習データが少ないため、過学習してしまった。
