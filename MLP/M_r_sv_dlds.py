import torch
import torch.nn as nn
import torch.utils.data as Data

Batch_size = 5

n_data = torch.ones(100,2)
x0 = torch.normal(2 * n_data,1)
y0 = torch.zeros(100)
x1 = torch.normal(-1 * n_data,1)
y1 = torch.ones(100)

x = torch.cat((x0,x1),0).type(torch.FloatTensor)
y = torch.cat((y0,y1),0).type(torch.LongTensor)

torch_dataset = Data.TensorDataset(x,y)
loader = Data.DataLoader(
    dataset = torch_dataset,
    batch_size = Batch_size,
    shuffle = True,
    num_workers = 0,
)

net = nn.Sequential(
    nn.Linear(2,10),
    nn.ReLU(),
    nn.Linear(10,2)
)

net.load_state_dict(torch.load('mlp4rg.pkl'))
optimizer = torch.optim.Adam(net.parameters())
loss_fn = nn.CrossEntropyLoss()

for epoch in range(3):
    for step,(b_x, b_y) in enumerate(loader):
        prd = net(b_x)
        loss = loss_fn(prd,b_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(loss.item())

torch.save(net.state_dict(),'mlp4rg.pkl')