from torch.autograd import Variable
import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__(self,
        n_chans,
        drop_1_prob=0.5,
                 ):
        super(MLP, self).__init__()
        self.__dict__.update(locals())
        #classifier
        self.classifier = nn.Sequential()
        self.classifier.add_module("linear_1", nn.Linear(n_chans*61, 2))
        # self.classifier.add_module("linear_2", nn.Linear(3, 3))
        # self.classifier.add_module('activation_2', nn.Sigmoid())
        # self.classifier.add_module("linear_3", nn.Linear(3, 3))
        # self.classifier.add_module('activation_3', nn.Sigmoid())
        # self.classifier.add_module("linear_4", nn.Linear(3, 2))
        # self.classifier.add_module('activation_3', nn.Sigmoid())
        self.classifier.add_module('bnorm_1', nn.BatchNorm1d(2, momentum=0.01, affine=True, eps=1e-3),)
        self.classifier.add_module('elu_1', nn.ELU())
        self.classifier.add_module("log_softmax", nn.LogSoftmax())

    def forward(self, x, **kwargs):
        #x is shape (batch, chan, time)
        features_flattened = x.view(x.size(0), -1)
        out = self.classifier(features_flattened)
        return out
