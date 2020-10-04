
import numpy as np

# affine:順伝播で行う行列の積
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) +self.b

        return out
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx
w=np.array([[1,2,3],
            [4,5,6]])
b=np.array([7,7,7])
x=np.array([[1,2],
            [3,4],
            [5,6]])

aff=Affine(w,b)
dout=aff.forward(x)
print("forward-->\n",aff.forward(x))
print("backward-->\n",aff.backward(dout))
print(aff.dW,aff.db)
