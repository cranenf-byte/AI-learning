import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.unsqueeze(torch.linspace(-1,1,100),dim = 1)
y = x.pow(2) + 0.3 * torch.rand(x.size())

class Net(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(Net,self).__init__()
        self.hidden = nn.Linear(input_size,hidden_size)
        self.output = nn.Linear(hidden_size,output_size)

    def forward(self,x):
        x = F.relu(self.hidden(x))
        x = self.output(x)
        return x

net = Net(1,10,1)
optimizer = torch.optim.Adam(net.parameters())
loss_func = nn.MSELoss()

for t in range(100):
    prd = net(x)

    loss = loss_func(prd,y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 10 == 0:
        print(loss.item())

