# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import pickle
from mnist import load_mnist
from functions import sigmoid, softmax


def get_data():   #mnistファイルにあるload_mnist関数を用いて画像データを読み込む
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    #normalize:入力画像を0.0~1.0に正規化, flatten:1*28*28の三次元行列を要素数784個の一次元配列に変更
    return x_test, t_test


def init_network():   #ファイルに格納されたweight,biasのデータを呼び出し,辞書型で格納
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

#行列の計算(dot演算)を用いているから多次元配列でも同時処理が行える。
def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = init_network()
accuracy_cnt = 0
#各画像のデータを一回一回精査し、for文を用いて全画像データを精査 -->低効率-->バッチ処理(効率○)に変更
"""for i in range(len(x)):
    y = predict(network, x[i])
    p= np.argmax(y) # 最も確率の高い要素のインデックスを取得
    if p == t[i]:   #配列tには各画像の正答が格納されている
        accuracy_cnt += 1"""
#画像データを人まとまりにして読み込み並行して精査を行う(今回の場合は100ごとに読み込む)
batch_size=100 #まとめるデータの個数
for i in range(0,len(x),batch_size):
    x_batch=x[i:i+batch_size]
    y_batch=predict(network,x_batch)
    p=np.argmax(y_batch,axis=1)
    #if i==0:   構造確認用
    #    print(p)
    #    print(t[i:i+batch_size])
    #    print(p==t[i:i+batch_size])
    accuracy_cnt+=np.sum(p==t[i:i+batch_size])
print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
