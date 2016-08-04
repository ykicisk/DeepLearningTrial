# ex3

## (基本) TensorflowでMNIST

### 準備

```sh
$ python
>>> from tensorflow.examples.tutorials.mnist import input_data
>>> mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
```

### 参考
 
http://qiita.com/TomokIshii/items/92a266b805d7eee02b1d

## (応用) TensorflowのRNNで俳句生成

## 俳句データセット

ソース: http://www.weblio.jp/category/hobby/gndhh/aa

## RNNとは

こちらを参考
http://qiita.com/kiminaka/items/87afd4a433dc655d8cfd

## RNN on Tensorflow (BasicRNNCell)

上記URLで記述されていた最もシンプルなRNNは、`BasicRNNCell`クラスで実装されている。
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn_cell.py#L174-L195

* 状態$s_t$と呼ばれるものが抽象ClassのRNNCellで表現されていて、
  RNNCellを帰ることでLSTMなども同じ用に利用できる。
* output = 隠れ状態層なので、その後予測させる(Linearの$V$でwordベクトルにする等)ためにはもう一層必要になる。


### __init__

|引数|内容| 
|:--|:--|
|num_units|ユニット数=隠れ状態層のユニット数|
|activation|アクティベーション関数|

### property

|名前|内容| 
|:--|:--|
|state_size|隠れ状態を現すパラメータ数(BasicRNNCellなら隠れ層のユニット数)|
|output_size|隠れ状態層 == outputなので上と同じ|

### 関数

* __call__
	* 引数
		* input([batchsize * input_size])
		* state([batchsize * input_size])
		* scope (subgraphを作るためのもので、デフォルトはクラス名???）
	* 戻値: output, newstate
		* outputとnewstateは同じ|

```python
# _linearは線形結合を現す。Trueはbias。
output = self._activation(_linear([inputs, state], self._num_units, True))
```

## todo

* [x] 学習データ(俳句)収集
* [ ] 学習データをtokenizeする。
* [ ] RNN学習用バッチ関数を作る
* [ ] RNN構築
* [ ] 学習を回す
* [ ] 生成器を作る
* [ ] 可視化

## 学習データの収集

```sh
$ ./scripts/download_haiku.py -d data/haiku_list.txt
$ wc -l data/haiku_list.txt
38919 data/haiku_list.txt
```

## 学習データのtokenize

## 参考

### RNNについて

* http://qiita.com/kiminaka/items/87afd4a433dc655d8cfd
### RNN on tensorflow

* http://kzky.hatenablog.com/entry/2016/01/06/TensorFlow%3A_Recurrent_Neural_Networks
* http://rnn.classcat.com/2016/03/15/tensorflow-cc-recurrent-neural-networks-and-lstm/
* http://tkengo.github.io/blog/2016/03/14/text-classification-by-cnn/
* http://tensorflow.classcat.com/2016/03/13/tensorflow-cc-recurrent-neural-networks/
