# ex3

## Tensorflowで日経平均予測

### 準備

この辺りから学習データを作成

* http://www.m2j.co.jp/market/historical.php
* http://finance.yahoo.com/world-indices

```python
>>> import pandas as pd
>>> df = pd.read_csv("data/rnn_train.csv.gz", compression="gzip", index_col=0, parse_dates=True)
>>> df
                     DJI         IXIC        N100          N225  EURJPY  \
2010-01-04  10583.959961  2308.419922  696.989990  10654.790039  133.26
2010-01-05  10572.019531  2308.709961  697.570007  10681.830078  133.33
2010-01-06  10573.679688  2301.090088  698.270020  10731.450195  131.74
2010-01-07  10606.860352  2300.050049  697.760010  10681.660156  133.03
2010-01-08  10618.190430  2317.169922  701.380005  10798.320312  133.56
2010-01-12  10627.259766  2282.310059  693.500000  10879.139648  133.66
2010-01-13  10680.769531  2307.899902  694.440002  10735.030273  131.81
2010-01-14  10710.549805  2316.739990  696.320007  10907.679688  132.56
...
2016-08-02  18313.769531  5137.729980  853.809998  16391.449219  114.28
2016-08-03  18355.000000  5159.740234  853.320007  16083.110352  113.21
2016-08-04  18352.050781  5166.250000  859.330017  16254.889648  112.94
2016-08-05  18543.529297  5221.120117  869.609985  16254.450195  112.61
2016-08-08  18529.289062  5213.140137  869.770020  16650.570312  112.99
2016-08-09  18533.050781  5225.479980  878.669983  16764.970703  113.58
2016-08-10  18495.660156  5204.580078  876.229980  16735.119141  113.20
2016-08-12  18576.470703  5232.890137  885.309998  16919.919922  113.50      

    USDJPY
2010-01-04   92.71
2010-01-05   92.54
2010-01-06   91.64
2010-01-07   92.30
2010-01-08   93.29
2010-01-12   92.10
2010-01-13   90.97
2010-01-14   91.35
...
2016-08-02  102.38
2016-08-03  100.86
2016-08-04  101.25
2016-08-05  101.19
2016-08-08  101.89
2016-08-09  102.47
2016-08-10  101.80
2016-08-12  101.87
```

### 実行&テスト

```sh
$ ./scripts/main.py train data/rnn_train.csv.gz clt.ckpt classification -e 500 -r 5
WARNING:tensorflow:<tensorflow.python.ops.rnn_cell.BasicLSTMCell object at 0x110944890>: Using a concatenated state is slower and will soon be deprecated.  Use state_is_tuple=True.
load learning data: data/rnn_train.csv.gz
train model
=== train data setting ===
n_sample  : 1418
len_seq   : 6
n_epoch   : 500
batch_size: 300
--------------------------
epoch: 0
loss:0.692506968975	accuracy:0.53808182478
epoch: 1
loss:0.692492604256	accuracy:0.53808182478
epoch: 2
loss:0.692478775978	accuracy:0.53808182478
...
epoch: 497
loss:0.689675152302	accuracy:0.53808182478
epoch: 498
loss:0.689673185349	accuracy:0.53808182478
epoch: 499
loss:0.689671278	accuracy:0.53808182478
==========================
save model: clt.ckpt
```

```sh
$ ./scripts/main.py test data/rnn_train.csv.gz clt.ckpt classification -e 500 -r 5
load learning data: data/rnn_train.csv.gz
load model: clt.ckpt
test model
test accuracy: 0.496403
[[ 1.          0.          0.54524714  0.45475289]
 [ 0.          1.          0.54150534  0.45849466]
 [ 0.          1.          0.54465872  0.45534128]
 ...
 [ 1.          0.          0.54311776  0.45688227]
 [ 0.          1.          0.54408228  0.45591778]
 [ 1.          0.          0.54542714  0.45457292]]

```

いまいち。

