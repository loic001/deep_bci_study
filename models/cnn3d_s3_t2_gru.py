from torch.autograd import Variable
import torch.nn as nn
import torch

class CNN3D_S3_T2_GRU(nn.Module):
    def __init__(self,
        conv_1_n_features_out=5,
        conv_1_kernel_size=(2, 3, 3),
        drop_1_prob=0.5,
        bnorm_1_momentum=0.01,
        pool_1_mode='max',
        pool_1_size=2,
        pool_1_stride=1,

        conv_2_n_features_out=5,
        conv_2_kernel_size=(1, 3, 3),
        drop_2_prob=0.5,
        bnorm_2_momentum=0.01,
        pool_2_mode='max',
        pool_2_size=2,
        pool_2_stride=1,

        conv_3_n_features_out=5,
        conv_3_kernel_size=(1, 3, 3),
        drop_3_prob=0.5,
        bnorm_3_momentum=0.01,
        pool_3_mode='max',
        pool_3_size=2,
        pool_3_stride=1,

        gru_1_num_layers=1,
        gru_1_input_size=45,
        gru_1_hidden_size=10,
        gru_1_dropout=0.1,

        linear_1_in=10,
        linear_1_out=2
                 ):
        super(CNN3D_S3_T2_GRU, self).__init__()
        self.__dict__.update(locals())

        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[self.pool_1_mode]

        #features extraction
        self.features = nn.Sequential()
        self.features.add_module("conv_1", nn.Conv3d(1, self.conv_1_n_features_out, kernel_size=self.conv_1_kernel_size, stride=(2, 2, 2)))
        self.features.add_module('drop_1', nn.Dropout3d(p=self.drop_1_prob))
        self.features.add_module('bnorm_1', nn.BatchNorm3d(self.conv_1_n_features_out, momentum=self.bnorm_1_momentum, affine=True, eps=1e-3))
        self.features.add_module('activation_1', nn.ReLU())

        self.features.add_module("conv_2", nn.Conv3d(self.conv_1_n_features_out,
        self.conv_2_n_features_out, kernel_size=self.conv_2_kernel_size))
        self.features.add_module('drop_2', nn.Dropout3d(p=self.drop_2_prob))
        self.features.add_module('bnorm_2', nn.BatchNorm3d(self.conv_2_n_features_out, momentum=self.bnorm_2_momentum, affine=True, eps=1e-3))
        self.features.add_module('activation_2', nn.ReLU())

        self.features.add_module("conv_3", nn.Conv3d(self.conv_2_n_features_out, self.conv_3_n_features_out, kernel_size=self.conv_3_kernel_size))
        self.features.add_module('drop_3', nn.Dropout3d(p=self.drop_3_prob))
        self.features.add_module('bnorm_3', nn.BatchNorm3d(self.conv_3_n_features_out, momentum=self.bnorm_3_momentum, affine=True, eps=1e-3))
        self.features.add_module('activation_3', nn.ReLU())

        self.recurrent = nn.GRU(input_size=self.gru_1_input_size,
                                                  hidden_size=self.gru_1_hidden_size,
                                                  dropout=self.gru_1_dropout,
                                                  num_layers=self.gru_1_num_layers,
                                                  batch_first=True)
        #classifier
        self.classifier = nn.Sequential()
        self.classifier.add_module("linear_1", nn.Linear(self.linear_1_in, self.linear_1_out))
        self.classifier.add_module("log_softmax", nn.LogSoftmax())

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(1 * self.gru_1_num_layers, batch_size, self.gru_1_hidden_size))

    def forward(self, x, **kwargs):
        batch_size = x.size()[0]
        self.hidden = self.init_hidden(batch_size)

        #x is shape (batch, h, w, time) permuted to (batch, time, h, w)
        x = x.permute(0, 3, 1, 2).float()
        x = x.unsqueeze(1)

        x = self.features(x)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.contiguous().view(x.size(0), x.size(1), -1)

        recurrent_out, self.hidden = self.recurrent(x)
        recurrent_out = recurrent_out[:,-1,:]
        # x = x.view(x.size(0), x.size(1), -1)
        out = self.classifier(recurrent_out)
        return out

    def init_weights(self):
        pass
