import numpy as np
from dezero import Variable
import dezero.functions as F

np.random.seed(0)
x = np.random.rand(100,1)
y = 5 + 2*x + np.random.rand(100,1)
x, y = Variable(x), Variable(y)

W = Variable(np.zeros((1,1)))   #重み
b = Variable(np.zeros(1))       #バイアス

def predict(x):
    y = F.matmul(x,W) + b
    return y
