import numpy as np
from dezero import Model
from dezero import optimizers
import dezero.layers as L
import dezero.functions as F

# -----------------------------------------------------
linear = L.Linear(10) #出力サイズのみの指定

batch_size,input_size = 100,5
x = np.random.randn(batch_size,input_size)
y = linear(x)

print('y shape:',y.shape)
print('param shape:',linear.W.shape,linear.b.shape)

for param in linear.params():
    print(param.name, param.shape)

# -----------------------------------------------------

# -----------------------------------------------------
#データセットの生成
np.random.seed(0)
x = np.random.rand(100,1)
y = np.sin(2*np.pi*x) + np.random.rand(100,1)

lr = 0.2
iters = 10000

class TwoLayerNet(Model):
    def __init__(self,hidden_size,output_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size) #層1
        self.l2 = L.Linear(output_size) #層2

    def forward(self,x):
        y = F.sigmoid(self.l1(x))
        y = self.l2(y)
        return y
    
model = TwoLayerNet(10,1)
optimizer = optimizers.SGD(lr) #オプティマイザの生成
optimizer.setup(model) #オプティマイザにモデルをセット

for i in range(iters):
    y_pred = model.forward(x)
    loss = F.mean_squared_error(y,y_pred)

    model.cleargrads()
    loss.backward()

    optimizer.update()
    if i % 1000 == 0:
        print(loss)

'''
for param in model.params():
    print(param)

model.cleargrads()
'''