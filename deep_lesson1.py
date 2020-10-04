import numpy as np
import matplotlib.pylab as plt
#ステップ関数
def step_function(x):
    y=x>0
    return y.astype(np.int)   #yにはbool値が入っていてそれを整数値に変換する。


#シグモイド関数:ステップ関数の改良
def sigmoid(x):
    y= 1/(1+np.exp(-x))
    return y


#出力層との中継ぎをする活性化関数(恒等関数:回帰問題に利用)
def identity_function(x):
    return x


#出力層との中継ぎをする活性化関数(ソフトマックス関数:分類問題に利用)
def softmax(x):
    m=np.max(x)
    exp_a=np.exp(x-m)   #オーバーフロー対策かつ、出力の総和を1にする
    sum_exp=np.sum(exp_a)
    return exp_a/sum_exp


#重み、バイアスの辞書を生成
def init_network():
    network={}  #辞書型
    network['W1']=np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['b1']=np.array([0.1,0.2,0.3])
    network['W2']=np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network['b2']=np.array([0.1,0.2])
    network['W3']=np.array([[0.1,0.3],[0.2,0.4]])
    network['b3']=np.array([0.1,0.2])
    return network


#init_network関数で作成した辞書からデータを取り出し,変数へ格納,演算
def forward(network, x):   #network:重みなどのリスト,x:入力
    W1,W2,W3=network['W1'],network['W2'],network['W3']
    b1,b2,b3=network['b1'],network['b2'],network['b3']

    a1=np.dot(x, W1)+b1
    z1=sigmoid(a1)
    a2=np.(z1, W2)+b2
    z2=sigmoid(a2)
    a3=np.dot(z2, W3)+b3
    y=identity_function(a3)
    return y


#損失関数
def mean_squared_error(y,t): #二乗和誤差
    return 0.5*np.sum((y-t)**2)

def cross_entropy_error(y,t):  #交差エントロピー誤差
    delta=1e-7   #-inf 対策
    return -np.sum(t*np.log(y+delta))
