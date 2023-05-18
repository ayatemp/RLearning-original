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

def mean_squared_error(x0,x1):
    diff = x0 - x1
    return F.sum(diff**2) / len(diff)

lr = 0.1
iters = 100

for i in range(iters):
    y_pred = predict(x)
    loss = mean_squared_error(y,y_pred) 

    W.cleargrad()
    b.cleargrad()
    loss.backward() #勾配を計算し自動で逆転バイしてくれている

    W.data -= lr*W.grad.data #ここで重みとバイアスを更新している
    b.data -= lr*b.grad.data

    if i % 10 == 0:
        print(loss.data)
    
print('====')
print('W =', W.data)
print('b =', b.data)
