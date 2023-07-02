import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, block_num = 1, output_class = 10):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.block_num = block_num
        self.forget_gate = nn.Linear(in_features=input_dim + hidden_dim, out_features=hidden_dim)
        self.input_gate = nn.Linear(in_features=input_dim + hidden_dim, out_features=hidden_dim)
        self.cell_update = nn.Linear(in_features=input_dim + hidden_dim, out_features=hidden_dim)
        self.output_gate = nn.Linear(in_features=input_dim + hidden_dim, out_features=hidden_dim)
        self.classify = nn.Linear(in_features=hidden_dim, out_features=output_class)
        self.softmax = nn.LogSoftmax(dim=1)
        # self.device = device

    def forward(self, input):
        input_length = input.size()[0]
        hidden = torch.zeros(1, self.hidden_dim).to(input.device)
        cell = torch.zeros(1, self.hidden_dim).to(input.device)
        output = None
        for i in range(input_length):
            x = input[i]
            state = torch.concat((x, hidden), dim=-1)
            f = torch.sigmoid(self.forget_gate(state))
            i = torch.sigmoid(self.input_gate(state))
            c = torch.tanh(self.cell_update(state))
            cell = f * cell + i * c
            output = torch.sigmoid(self.output_gate(state))
            hidden = output * torch.tanh(cell)
        output = self.softmax(self.classify(output))
        return output