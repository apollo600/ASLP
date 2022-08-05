### gru 1layer + dnn 3layers

```bash
NET(
  (stft): ConvSTFT()
  (istft): ConviSTFT()
  (gru): GRU(257, 512, batch_first=True)
  (relu): ReLU()
  (fc1): Linear(in_features=512, out_features=1024, bias=True)
  (fc2): Linear(in_features=1024, out_features=1024, bias=True)
  (fc3): Linear(in_features=1024, out_features=257, bias=True)
)

#param: 3.02M
```

- 时间花费正常
- 疑似过拟合，在训练集上损失为-16.5，在验证集上损失为-15.55
- 虽然在训练集和验证集都在继续提升，但是在验证集上提升很小
- 解决方案：
  - 加上dropout，gru多使用几层
  - L2正则化
  - 减少特征数量，特征数量过多容易导致过拟合
- 这个是目前最好的一个结果，SI-SDR结果为18.323

### gru 3layers + dnn 3layers

```bash
NET(
  (stft): ConvSTFT()
  (istft): ConviSTFT()
  (gru): GRU(257, 512, num_layers=3, batch_first=True, dropout=0.1)
  (relu): ReLU()
  (fc1): Linear(in_features=512, out_features=256, bias=True)
  (fc2): Linear(in_features=256, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=257, bias=True)
)

#param: 4.53M

weight_decay: 0.01
```

- 参数数量如果从大小上来看是变多了
- 训练速度明显变慢，这是RNN无法避免的问题
- 收敛的太慢，原因是dropout给的太大了

### gru 3layers + dnn 3layers

```bash
NET(
  (stft): ConvSTFT()
  (istft): ConviSTFT()
  (gru): GRU(257, 128, num_layers=4, batch_first=True, dropout=0.2)
  (relu): ReLU()
  (fc1): Linear(in_features=128, out_features=512, bias=True)
  (fc2): Linear(in_features=512, out_features=512, bias=True)
  (fc3): Linear(in_features=512, out_features=257, bias=True)
)

#param: 0.91M

weight_decay: 0.005
```

- 训练速度慢
- 且收敛速度慢

### gru 3layers + dnn 3layers

```bash
NET(
  (stft): ConvSTFT()
  (istft): ConviSTFT()
  (gru): GRU(257, 512, num_layers=4, batch_first=True, dropout=0.1)
  (relu): ReLU()
  (fc1): Linear(in_features=512, out_features=512, bias=True)
  (fc2): Linear(in_features=512, out_features=1024, bias=True)
  (fc3): Linear(in_features=1024, out_features=257, bias=True)
)

#param: 6.96M

weight_decay: 1e-5
```

### dnn 1layer + gru 8layers + dnn 2layers

```bash
NET(
  (stft): ConvSTFT()
  (istft): ConviSTFT()
  (gru): GRU(128, 128, num_layers=8, batch_first=True, dropout=0.2)
  (relu): ReLU()
  (fc1): Linear(in_features=257, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=1024, bias=True)
  (fc3): Linear(in_features=1024, out_features=257, bias=True)
)

#param: 1.22M
```

- 模型过大，会导致过拟合，考虑只使用三层全连接，尝试减小gru的尺寸
- gru实在训不起来，太慢了，而且效果也不好

### dnn 1layer + gru 2layers + dnn 2layers

```bash
NET(
  (stft): ConvSTFT()
  (istft): ConviSTFT()
  (gru): GRU(257, 257, num_layers=2, batch_first=True, dropout=0.2)
  (relu): ReLU()
  (fc1): Linear(in_features=257, out_features=257, bias=True)
  (fc2): Linear(in_features=257, out_features=1024, bias=True)
  (fc3): Linear(in_features=1024, out_features=257, bias=True)
)

#param: 1.39M
```

- gru层数减小效果稍微好一些，但是总是会卡住
- 倒是下去了，dropout开启之后收敛速度会变慢？
- 3个epoch，下降到-14
- 考虑试一下这一个组合，似乎可以到-15，由于dropout才显得比较慢
- 10个epoch，下降到-14.85，主要是没有过拟合
- 21个epoch，下降到-15.2，开始有点过拟合了
- 36个epoch，下降到-15.39，训练集是-15.72，怎么不算过拟合呢

时域网络

|  ID  |             Settings             | Param |     Loss      | SI-SDR |  PESQ impr|
| :--: | :------------------------------: | :---: | :-----------: | :--: | :--: |
|  0   |      gru-1layer/dnn-3layer       | 3.02M | -16.51/-15.55 |  18.323  | - |
|  1   | gru-3layer/dnn-3layer | 4.53M | too slow |  -   | - |
|  2   | dnn-1layer/gru-2layer/dnn-1layer | 1.39M | -15.72/-15.39 | 18.374 | - |
| 3 | dnn-1layer/gruc-1layer/dnn-1layer/conv | 4.25M | under-fitting | - | - |
| 4 | dnn-3layer/gruc-2layer/dnn-1layer/conv | 4.10M | fast enough | - | - |
| 5 | dnn-3layer/gruc-2layer/dnn-3layer | 5.04M | -16.37/-15.82 | 18.398 | 0.288 |
| 6 | dnn-3layer/gruc-2layer/dnn-3layer | 5.04M | -16.97/-15.86 | 18.473 | 0.289 |
| 7 | dnn-3layer/gruc-2layer/dnn-3layer | 5.04M | -16.29/-15.81 | 18.493 | 0.296 |
| 8 | dnn-3layer/gruc-2layer/dnn-3layer | 5.04M | -16.86/-15.88 | 18.267 | 0.293 |

频域网络

|  ID  |         Settings         | SI-SDR | PESQ  |
| :--: | :----------------------: | :----: | :---: |
|  0   | 257 -> 637 -> 637 -> 257 | 17.585 | 0.097 |
|  1   | 257 -> 512-> 512 -> 257  | 17.618 | 0.101 |
|  2   | 257 -> 512-> 512 -> 257  | 17.663 | 0.202 |
