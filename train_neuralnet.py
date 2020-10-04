# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
from mnist import load_mnist
from two_layer_net import TwoLayerNet

# データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000  # 繰り返しの回数を適宜設定する
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

#1エポックあたりの繰り返し回数
# train_size:全データの個数
iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    # train_size:全データの個数, batch_size:ランダム抽出するデータの個数
    # batch_mask:無作為抽出したデータのインデックスを示すリスト
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 勾配の計算
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)

    # パラメータの更新
    # 各パラメータを1度だけ更新する(傾きが負なら正の方向、正なら負の方向に進めば良い)
    #移動する幅は傾きにlearning_rate=0.1をかけた数とする。
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    #交差エントロピー誤差を損失関数として、その値が返される。
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    #全ての重みに対して正確性を調べると処理が重すぎる→1エポックあたりの回数分だけ調べる。
    if i % iter_per_epoch == 0:
        #入力データ・教師データをx_train(学習に用いたデータ)と
        #t_train(学習データとは異なるテスト用データ)としてその正確性を調べる。
        #抽出されたデータは変わらないが、重みの値の更新を行なった回数が異なる。1セット行う→iter_per_epochセット行う・・・
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

# グラフの描画
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
