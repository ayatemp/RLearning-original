import numpy as np
from dezero import Variable
import dezero.functions as F


x_np = np.array(5.0)
x = Variable(x_np)

y = 3*x**2
print(y)

a = np.array([1,2,3])
b = np.array([4,5,6])
a,b = Variable(a), Variable(b)
c = F.matmul(a,b)
print(c)

print("-------------------------")

a = np.array([[1,2],[3,4]])
b = np.array([[5,6],[7,8]])
c = F.matmul(a,b)
print(c)


print("-------------------------")

def rosenbrock(x0,x1):
    y = 100 * (x1 - x0 ** 2) ** 2 + (x0 - 1) ** 2
    return y

x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))

y = rosenbrock(x0,x1)
print(y)
y.backward()
print(x0.grad, x1.grad)

print("-------------------------")

def rosenbrock(x0, x1):
    y = 100 * (x1 - x0 ** 2) ** 2 + (x0 - 1) ** 2
    return y

x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))

lr = 0.001      #learning rate
iters = 10000   #iteration count

for i in range(iters):
    #print(x0,x1)
    y = rosenbrock(x0,x1)

    x0.cleargrad() #clear gradient
    x1.cleargrad() #clear gradient
    y.backward()   #自動で偏微分かつ事前に入った変数に勾配を入れる

    x0.data -= lr*x0.grad.data
    x1.data -= lr*x1.grad.data

print(x0,x1)
