import torch
import torch.nn as nn
import torch.nn.functional as F

n_data = torch.ones(100,2)
x0 = torch.normal(2 * n_data,1)
y0 = torch.zeros(100)
x1 = torch.normal(-1 * n_data,1)
y1 = torch.ones(100)

x = torch.cat((x0,x1),0).type(torch.FloatTensor)
y = torch.cat((y0,y1),0).type(torch.LongTensor)

net = nn.Sequential(
    nn.Linear(2,10),
    nn.ReLU(),
    nn.Linear(10,2)
)

optimizer = torch.optim.Adam(net.parameters())
loss_fn = nn.CrossEntropyLoss()

for i in range(100):
    prd = net(x)
    loss = loss_fn(prd,y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 10 == 0:
        print(loss.item())