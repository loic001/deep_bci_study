from torch.autograd import Variable
import torch.nn as nn

class CNN2D_S3(nn.Module):
    def __init__(self,
        conv_1_n_features_in,
        conv_1_n_features_out=5,
        conv_1_kernel_size=3,
        drop_1_prob=0.5,
        bnorm_1_momentum=0.01,
        pool_1_mode='max',
        pool_1_size=2,
        pool_1_stride=1,

        conv_2_n_features_out=5,
        conv_2_kernel_size=3,
        drop_2_prob=0.5,
        bnorm_2_momentum=0.01,
        pool_2_mode='max',
        pool_2_size=2,
        pool_2_stride=1,

        conv_3_n_features_out=5,
        conv_3_kernel_size=3,
        drop_3_prob=0.5,
        bnorm_3_momentum=0.01,
        pool_3_mode='max',
        pool_3_size=2,
        pool_3_stride=1,

        linear_1_in=500,
        linear_1_out=2
                 ):
        super(CNN2D_S3, self).__init__()
        self.__dict__.update(locals())

        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[self.pool_1_mode]

        #features extraction
        self.features = nn.Sequential()
        self.features.add_module("conv_1", nn.Conv2d(self.conv_1_n_features_in,
            self.conv_1_n_features_out, kernel_size=self.conv_1_kernel_size))
        self.features.add_module('drop_1', nn.Dropout2d(p=self.drop_1_prob))
        self.features.add_module('bnorm_1', nn.BatchNorm2d(self.conv_1_n_features_out, momentum=self.bnorm_1_momentum, affine=True, eps=1e-3))
        self.features.add_module('activation_1', nn.ReLU())

        self.features.add_module("conv_2", nn.Conv2d(self.conv_1_n_features_out,
            self.conv_2_n_features_out, kernel_size=self.conv_2_kernel_size))
        self.features.add_module('drop_2', nn.Dropout2d(p=self.drop_2_prob))
        self.features.add_module('bnorm_2', nn.BatchNorm2d(self.conv_2_n_features_out, momentum=self.bnorm_2_momentum, affine=True, eps=1e-3))
        self.features.add_module('activation_2', nn.ReLU())

        self.features.add_module("conv_3", nn.Conv2d(self.conv_2_n_features_out,
            self.conv_3_n_features_out, kernel_size=self.conv_3_kernel_size))
        self.features.add_module('drop_3', nn.Dropout2d(p=self.drop_3_prob))
        self.features.add_module('bnorm_3', nn.BatchNorm2d(self.conv_3_n_features_out, momentum=self.bnorm_3_momentum, affine=True, eps=1e-3))
        self.features.add_module('activation_3', nn.ReLU())

        #classifier
        self.classifier = nn.Sequential()
        self.classifier.add_module("linear_1", nn.Linear(self.linear_1_in, self.linear_1_out))
        self.classifier.add_module("log_softmax", nn.LogSoftmax())

    def forward(self, x, **kwargs):

        #x is shape (batch, h, w, time) permuted to (batch, time, h, w)
        x = x.permute(0, 3, 1, 2).float()
        features_out = self.features(x)

        #flatten features map
        features_flattened = features_out.view(features_out.size(0), -1)

        out = self.classifier(features_flattened)
        return out

    def init_weights(self):
        pass
