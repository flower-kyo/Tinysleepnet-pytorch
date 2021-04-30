import torch
import torch.nn as nn
from collections import OrderedDict


class TinySleepNet(nn.Module):
    def __init__(self, config):
        super(TinySleepNet, self).__init__()
        self.padding_edf = {  # same padding in tensorflow
            'conv1': (22, 22),
            'max_pool1': (2, 2),
            'conv2': (3, 4),
            'max_pool2': (0, 1),
        }
        self.config = config
        first_filter_size = int(self.config["sampling_rate"] / 2.0)  # 100/2 = 50, 与以往使用的Resnet相比，这里的卷积核更大
        first_filter_stride = int(self.config["sampling_rate"] / 16.0)  # todo 与论文不同，论文给出的stride是100/4=25
        self.cnn = nn.Sequential(
            nn.ConstantPad1d(self.padding_edf['conv1'], 0),  # conv1
            nn.Sequential(OrderedDict([
                ('conv1', nn.Conv1d(in_channels=1, out_channels=128, kernel_size=first_filter_size, stride=first_filter_stride,
                      bias=False))
            ])),
            # nn.BatchNorm1d(128),
            # nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.99),
            nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.ConstantPad1d(self.padding_edf['max_pool1'], 0),  # max p 1
            nn.MaxPool1d(kernel_size=8, stride=8),
            nn.Dropout(p=0.5),
            nn.ConstantPad1d(self.padding_edf['conv2'], 0),  # conv2
            nn.Sequential(OrderedDict([
                ('conv2',
                 nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, bias=False))
            ])),
            # nn.BatchNorm1d(128),
            # nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.99),
            nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.01),

            nn.ReLU(inplace=True),
            nn.ConstantPad1d(self.padding_edf['conv2'], 0),  # conv3
            nn.Sequential(OrderedDict([
                ('conv3',nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, bias=False))
            ])),
            # nn.BatchNorm1d(128),
            # nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.99),
            nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.01),

            nn.ReLU(inplace=True),
            nn.ConstantPad1d(self.padding_edf['conv2'], 0),  # conv4
            nn.Sequential(OrderedDict([
                ('conv4', nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, bias=False))
            ])),
            # nn.BatchNorm1d(128),
            # nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.99),
            nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.ConstantPad1d(self.padding_edf['max_pool2'], 0),  # max p 2
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Flatten(),
            nn.Dropout(p=0.5),
        )
        # self.rnn = nn.LSTM(input_size=2048, hidden_size=self.config['n_rnn_units'], num_layers=1, dropout=0.5)
        # self.rnn = nn.LSTM(input_size=2048, hidden_size=self.config['n_rnn_units'], num_layers=1)
        self.rnn = nn.LSTM(input_size=2048, hidden_size=self.config['n_rnn_units'], num_layers=1, batch_first=True)
        self.rnn_dropout = nn.Dropout(p=0.5)  # todo 是否需要这个dropout?
        self.fc = nn.Linear(self.config['n_rnn_units'], 5)






    def forward(self, x, state):
        x = self.cnn(x)
        # input of LSTM must be shape(seq_len, batch, input_size)
        # x = x.view(self.config['seq_length'], self.config['batch_size'], -1)
        x = x.view(-1, self.config['seq_length'], 2048)  # batch first == True
        assert x.shape[-1] == 2048
        x, state = self.rnn(x, state)
        # x = x.view(-1, self.config['n_rnn_units'])
        x = x.reshape(-1, self.config['n_rnn_units'])
        # rnn output shape(seq_length, batch_size, hidden_size)
        x = self.rnn_dropout(x)
        x = self.fc(x)

        return x, state



if __name__ == '__main__':
    from torchsummaryX import summary
    from config.sleepedf import train

    model = TinySleepNet(config=train)
    state = (torch.zeros(size=(1, 2, 128)),
             torch.zeros(size=(1, 2, 128)))
    torch
    # state = (state[0].to(self.device), state[1].to(self.device))
    summary(model, torch.randn(size=(2, 1, 3000)), state)



