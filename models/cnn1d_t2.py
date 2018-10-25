from torch.autograd import Variable
import torch.nn as nn

class CNN1D_T2(nn.Module):
    def __init__(self,
        n_chans,
        pool_mode='max',
        conv_1_out_channels=10,
        conv_1_stride=2,
        conv_1_kernel_size=2,
        drop_1_p=0.5,
        linear_1_in=300,
        linear_1_out=2
                 ):
        super(CNN1D_T2, self).__init__()
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

        #classifier
        self.classifier = nn.Sequential()
        self.classifier.add_module('linear_1', nn.Linear(self.linear_1_in, self.linear_1_out, bias=True))
        self.classifier.add_module('softmax', nn.LogSoftmax())

    def forward(self, x, **kwargs):
        x = x.unsqueeze(1)
        features_out = self.features(x)
        #flatten features map
        features_flattened = features_out.view(features_out.size(0), -1)
        out = self.classifier(features_flattened)
        return out

    def init_weights(self):
        pass
