import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
         input_size=input_size,
         hidden_size=hidden_size,
         num_layers=1,
         batch_first=True,
        )
        self.output = nn.Linear(64,10)