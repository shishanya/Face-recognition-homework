# 石山，今天也要加油学习鸭！
from utils1 import mytorch as d2l
import torch
from torch import nn
import numpy as np
from time import *



test_fea = np.load('./data/save_test_fea.npy')
test_lab = np.load('./data/save_test_lab.npy')
train_fea = np.load('./data/save_train_fea.npy')
train_lab = np.load('./data/save_train_lab.npy')


#多层感知机
batch_size = 32
train_iter = d2l.load_array((torch.Tensor(train_fea), torch.LongTensor(train_lab)), batch_size)
test_iter =  d2l.load_array((torch.Tensor(test_fea), torch.LongTensor(test_lab)), batch_size)


net = nn.Sequential(nn.Flatten(),
                    nn.Linear(512, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 625))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

lr, num_epochs = 0.1, 50
loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=lr)

begin_time = time()
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
end_time  = time()
print("该程序的运行时间是： ", end_time - begin_time)





