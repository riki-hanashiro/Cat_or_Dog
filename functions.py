# coding: utf-8
import numpy as np


def identity_function(x):
    return x


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    grad = np.zeros_like(x)
    grad[x>=0] = 1
    return grad


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)   # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


def sum_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)   #教師データと確率値を確実に1行n列にする。
        y = y.reshape(1, y.size)

    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size  #arange(batch_size)は0から(arange_size-1)までの配列を与える。
    #tには正解値のインデックスが格納されている-->[0,3,5,6,2,4,6,,,,,3,5,1,0]
    #よってy[[0,1,2,,,,,,(batch_size-1)],[0,3,,,,,,3,5,1,0]]となり、各データの正解値だけを取り出して計算できる。
    #y[0][0]+y[1][3]+,,,,,,,+y[batch_size-1][0]
def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)
