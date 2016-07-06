# ex2

* chainerで顔画像から性別を判定する。
* CNNはとりあえず使わない

## データセット

### 利用するデータセット

The Images of Groups Dataset
http://chenlab.ece.cornell.edu/people/Andy/ImagesOfGroups.html

### 前処理

`scripts/preprocessing.py`

* 顔画像切り出し
* グレイスケールに変換
* リサイズ

```sh
./preprocessing.py -s data/Group2a -d data/images --resize 64
```

## chainerによる学習




