#!/usr/bin/python3
# http://tutorialspoint.com/pytorch

import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

n_in = 10
n_h = 5
n_out = 1
batch_size = 10

x = torch.randn(batch_size, n_in)
x_train = np.array(x, dtype=np.float32)
print(x)
print(x_train)
y = torch.tensor([[1.0], [0.0], [0.0], [1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0]])
print(y)

model = nn.Sequential(nn.Linear(n_in, n_h),
				nn.ReLU(),
				nn.Linear(n_h, n_out),
				nn.Sigmoid())

criterion = torch.nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(50):
	y_pred = model(x)
	loss = criterion(y_pred, y)
	print('epoch:', epoch, 'loss:', loss.item())
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

with torch.no_grad():
	predicted = model(Variable(torch.from_numpy(x_train))).data.numpy()
	print(predicted)

plt.clf()
plt.plot(x, y, 'go', label='True data', alpha=0.5)
plt.plot(x, predicted, '--', label='Predictions', alpha=0.5)
plt.legend(loc='best')
plt.show()
