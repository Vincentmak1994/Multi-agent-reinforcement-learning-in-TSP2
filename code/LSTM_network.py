import torch
import torch.nn as nn
import torch.optim as optim
import os
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_action, name='LSTM_default', chkpt_dir='tmp/LSTM'):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_action = n_action
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir+'_LSTM')

        self.LSTM_layer = nn.LSTM(self.input_dim, self.hidden_dim, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.n_action)

    def forward(self, state):
        lstm_out, _ = self.LSTM_layer(state)
        output = self.fc(lstm_out)
        output_score = F.log_softmax(output, dim=1)
        return output_score

if __name__ == '__main__':
    input = [0, 1, 1, 0, 0, 0]
    hidden_dim=6 
    n_action = len(input)-1

    model = LSTM(len(input), hidden_dim, n_action)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # prob = model()
    print(torch.tensor(input, dtype=torch.float32).size())