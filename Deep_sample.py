
from collections import OrderedDict
import numpy as np
import random
from util import im2col
#普通の辞書とOrderedDictの違い
"""od1=OrderedDict()
od2=OrderedDict()

od1["1"]="a"
od1["2"]="b"
od2["2"]="b"
od2["1"]="a"
print(od1)
print(od2)
print(od1==od2)

d1={"1":"a","2":"b"}
d2={"2":"b","1":"a"}
print(d1)
print(d2)
print(d1==d2)

x=np.array([[1,2,3,4,5],[3,4,5,6,7],[3,5,6,7,1],[3,7,4,2,5]])
print(x)
mask=np.random.choice(4,2)
print("mask→ ",mask)
print(x[mask])

a=6*np.random.rand(10) > 5
print(a)
"""

class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W=W
        self.b=b
        self.stride=stride
        self.pad=pad
    def forword(self,x):
        FN, C, FH, FW=self.W.shape
        N, C, H, W=x.shape
        out_h=int(1+(H+2*self.pad-FH)/self.stride)
        out_w=int(1+(W+2*self.pad-FW)/self.stride)

        col=im2col(x, FH, FW, self.stride, self.pad)
        col_W=self.W.reshape(FN, -1).T
        out=np.dot(col, col_W)+self.b

        out=out.reshape(N, out_h, out_w, -1).transpose(0,3,1,2)

        return out

w=np.random.rand(1,3,5,5)
b=np.ones(0)
x=np.random.rand(1,2,2,2)

Conv=Convolution(w, b)
print(Conv.forword(x))
