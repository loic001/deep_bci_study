import torch
from torch.autograd import Variable
import torch.nn as nn

class CNN1D_T2_GRU(nn.Module):
    def __init__(self,
        n_chans,
        pool_mode='max',
        conv_1_out_channels=10,
        conv_1_stride=2,
        conv_1_kernel_size=2,
        drop_1_p=0.5,


        gru_1_num_layers=1,
        gru_1_hidden_size=10,
        gru_1_dropout=0.1,

        linear_1_in=10,
        linear_1_out=2,
                 ):
        super(CNN1D_T2_GRU, self).__init__()
        self.__dict__.update(locals())

        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[self.pool_mode]
        self.features = nn.Sequential()
        #nn.Conv2d is use as a 1-dimension convolution (with filter height equal to n_chans)
        self.features.add_module('conv_1', nn.Conv2d(1, self.conv_1_out_channels, (n_chans, self.conv_1_kernel_size),
                                                     stride=(n_chans, self.conv_1_stride),
                                                     bias=False))
        self.features.add_module('drop_1', nn.Dropout2d(p=self.drop_1_p))
        self.features.add_module('bnorm_1', nn.BatchNorm2d(self.conv_1_out_channels, momentum=0.01, affine=True, eps=1e-3),)
        self.features.add_module('elu_1', nn.ELU())

        self.recurrent = nn.GRU(input_size=self.conv_1_out_channels,
                                                  hidden_size=self.gru_1_hidden_size,
                                                  dropout=self.gru_1_dropout,
                                                  num_layers=self.gru_1_num_layers,
                                                  batch_first=True)

        #classifier
        self.classifier = nn.Sequential()
        self.classifier.add_module('linear_1', nn.Linear(self.linear_1_in, self.linear_1_out, bias=True))
        self.classifier.add_module('softmax', nn.LogSoftmax())

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(1 * self.gru_1_num_layers, batch_size, self.gru_1_hidden_size))

    def forward(self, x, **kwargs):
        batch_size = x.size()[0]
        self.hidden = self.init_hidden(batch_size)

        #x (Batch, n_chans, T) to (Batch, 1, n_chans, T)
        x = x.unsqueeze(1)
        features_out = self.features(x)
        #features_out (Batch, 10, conv_1_out_channels, 1, T//conv_1_stride) to (Batch, 10, conv_1_out_channels, T//conv_1_stride)
        features_out = features_out.squeeze(2)
        features_out = features_out.permute(0, 2, 1)

        #recurrent gru : expect (batch, seq_length, input_size)
        recurrent_out, self.hidden  = self.recurrent(features_out, self.hidden)
        recurrent_out = recurrent_out[:,-1,:]

        #flatten recurrent_out
        recurrent_flattened = recurrent_out.contiguous().view(recurrent_out.size(0), -1)

        #linear classifier and softmax
        out = self.classifier(recurrent_flattened)
        return out

    def init_weights(self):
        pass
