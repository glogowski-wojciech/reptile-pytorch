import torch
from torch import nn
from torch.autograd import Variable


class ProgressiveReptileModel(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)

    def point_grad_to(self, target):
        """
        Set .grad attribute of each parameter to be proportional
        to the difference between self and target
        """
        for named_p, target_p in zip(self.named_parameters(), target.parameters()):
            name, p = named_p
            if p.grad is None:
                if self.is_cuda():
                    p.grad = Variable(torch.zeros(p.size())).cuda()
                else:
                    p.grad = Variable(torch.zeros(p.size()))
            p.grad.data.zero_()  # not sure this is required
            p.grad.data.add_((p.data - target_p.data))

    def is_cuda(self):
        return next(self.parameters()).is_cuda


# Modules for Progressive Net architecture


class ConvModule(nn.Module):
    def __init__(self, channels_in, channels_out):
        nn.Module.__init__(self)
        self.conv = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, 2, 1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class LinearModule(nn.Module):
    def __init__(self, channels_in, channels_out):
        nn.Module.__init__(self)
        self.linear = nn.Sequential(
            nn.Linear(channels_in, channels_out),
            nn.LogSoftmax(1)
        )

    def forward(self, x):
        return self.linear(x)


# Adapters for Progressive Nets


class ConvToConvAdapter(nn.Module):
    def __init__(self, col_idx):
        nn.Module.__init__(self)
        self.col_idx = col_idx
        self.alphas = [nn.Parameter(torch.ones(1)) for _ in xrange(col_idx)]
        self.adapter = nn.Sequential(
            nn.Conv2d(64 * col_idx, 64, 1),
            nn.ReLU(True),
        )
        self.alphasRegister = nn.ParameterList(self.alphas)

    def forward(self, x, laterals):
        scaled_laterals = []
        for i, lateral in enumerate(laterals):
            alpha = self.alphas[i]
            scaled_laterals.append(lateral * alpha.expand_as(lateral))
        lateral = torch.cat(scaled_laterals, 1)
        lateral = self.adapter(lateral)
        return torch.cat([x, lateral], 1)


class ConvToLinearAdapter(nn.Module):
    def __init__(self, col_idx, num_classes):
        nn.Module.__init__(self)
        self.alphas = [nn.Parameter(torch.ones(1)) for _ in xrange(col_idx)]
        self.adapter0 = nn.Sequential(
            nn.Conv2d(64 * col_idx, 64, 1),
            nn.ReLU(True),
        )
        self.alphasRegister = nn.ParameterList(self.alphas)

    def forward(self, x, laterals):
        scaled_laterals = []
        for i, lateral in enumerate(laterals):
            alpha = self.alphas[i]
            scaled_laterals.append(lateral * alpha.expand_as(lateral))
        lateral = torch.cat(scaled_laterals, 1)
        lateral = self.adapter0(lateral)
        lateral = lateral.view(len(lateral), -1)
        x = x.view(len(x), -1)
        return torch.cat([x, lateral], 1)


class ProgressiveColumn(nn.Module):
    def __init__(self, col_idx, num_classes):
        nn.Module.__init__(self)
        self.col_idx = col_idx
        self.num_classes = num_classes

        num_inputs = 1 if col_idx == 0 else 2
        self.conv0 = ConvModule(1, 64)
        self.conv1 = ConvModule(num_inputs * 64, 64)
        self.conv2 = ConvModule(num_inputs * 64, 64)
        self.conv3 = ConvModule(num_inputs * 64, 64)
        self.linear = LinearModule(num_inputs * 256, self.num_classes)

        if col_idx == 0:
            self.conv1_adapt = None
            self.conv2_adapt = None
            self.conv3_adapt = None
            self.linear_adapt = None
        else:
            self.conv1_adapt = ConvToConvAdapter(self.col_idx)
            self.conv2_adapt = ConvToConvAdapter(self.col_idx)
            self.conv3_adapt = ConvToConvAdapter(self.col_idx)
            self.linear_adapt = ConvToLinearAdapter(self.col_idx, self.num_classes)

    def forward(self, x, laterals):
        if laterals is None:
            conv0_out = self.conv0(x)
            conv1_out = self.conv1(conv0_out)
            conv2_out = self.conv2(conv1_out)
            conv3_out = self.conv3(conv2_out)
            linear_out = self.linear(conv3_out.view(len(conv3_out), -1))
        else:
            laterals1, laterals2, laterals3, laterals4 = laterals
            conv0_out = self.conv0(x)
            lll = self.conv1_adapt(conv0_out, laterals1)
            conv1_out = self.conv1(lll)
            conv2_out = self.conv2(self.conv2_adapt(conv1_out, laterals2))
            conv3_out = self.conv3(self.conv3_adapt(conv2_out, laterals3))
            linear_out = self.linear(self.linear_adapt(conv3_out, laterals4))
        return [conv0_out, conv1_out, conv2_out, conv3_out, linear_out]


class OmniglotModel(ProgressiveReptileModel):
    def __init__(self, num_classes, num_columns):
        ProgressiveReptileModel.__init__(self)
        columns = [ProgressiveColumn(col_idx, num_classes) for col_idx in xrange(num_columns)]
        self.num_classes = num_classes
        self.columns = nn.ModuleList(columns)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        results = {}
        num_layers = None
        for col_idx, column in enumerate(self.columns):
            if col_idx == 0:
                laterals = None
            else:
                laterals = []
                for layer_idx in xrange(num_layers - 1):
                    layer_laterals = [results[i, layer_idx] for i in xrange(col_idx)]
                    laterals.append(layer_laterals)
            col_outs = column(x, laterals)
            num_layers = len(col_outs)
            for i, col_out in enumerate(col_outs):
                results[col_idx, i] = col_out
        out = {col_idx: results[col_idx, num_layers - 1] for col_idx in xrange(len(self.columns))}
        return out

    def predict(self, prob):
        __, argmax = prob.max(1)
        return argmax

    def clone(self):
        clone = OmniglotModel(self.num_classes, len(self.columns))
        clone.load_state_dict(self.state_dict())
        if self.is_cuda():
            clone.cuda()
        return clone


if __name__ == '__main__':
    model = OmniglotModel(20, 3)
    x = Variable(torch.zeros(5, 28*28))
    y = model(x)
    print 'x', type(x)
    print 'y', type(y)
    print 'x', x.size()
    print 'y', {k: y[k].shape for k in y}

