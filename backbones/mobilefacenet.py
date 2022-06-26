import copy

from torch.nn import (
    Linear,
    Conv2d,
    BatchNorm1d,
    BatchNorm2d,
    PReLU,
    ReLU,
    Sigmoid,
    Dropout2d,
    Dropout,
    AvgPool2d,
    MaxPool2d,
    AdaptiveAvgPool2d,
    Sequential,
    Module,
    Parameter,
)
import torch.nn.functional as F
import torch
import torch.nn as nn
from collections import namedtuple, OrderedDict
import math
#from .common import ECA_Layer, SEBlock, CbamBlock, Identity, GCT

##################################  Original Arcface Model #############################################################
from quantization_utils.quant_modules import Quant_Conv2d, Quant_Linear, QuantAct, QuantActPreLu


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
class Conv_block(Module):
    def __init__(
        self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1
    ):
        super(Conv_block, self).__init__()
        self.conv = Conv2d(
            in_c,
            out_channels=out_c,
            kernel_size=kernel,
            groups=groups,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = BatchNorm2d(out_c)
        self.prelu = PReLU(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class Linear_block(Module):
    def __init__(
        self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1
    ):
        super(Linear_block, self).__init__()
        self.conv = Conv2d(
            in_c,
            out_channels=out_c,
            kernel_size=kernel,
            groups=groups,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Depth_Wise(Module):
    def __init__(
        self,
        in_c,
        out_c,
        attention,
        residual=False,
        kernel=(3, 3),
        stride=(2, 2),
        padding=(1, 1),
        groups=1,
    ):
        super(Depth_Wise, self).__init__()
        self.conv = Conv_block(
            in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1)
        )
        self.conv_dw = Conv_block(
            groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride
        )
        self.project = Linear_block(
            groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1)
        )
        self.attention = attention
        self.residual = residual

        self.attention = attention  # se, eca, cbam

    def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.attention != "none":
            x = self.attention_layer(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output


class Residual(Module):
    def __init__(
        self,
        c,
        attention,
        num_block,
        groups,
        kernel=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
    ):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(
                Depth_Wise(
                    c,
                    c,
                    attention,
                    residual=True,
                    kernel=kernel,
                    padding=padding,
                    stride=stride,
                    groups=groups,
                )
            )
        self.model = Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class GNAP(Module):
    def __init__(self, embedding_size):
        super(GNAP, self).__init__()
        assert embedding_size == 512
        self.bn1 = BatchNorm2d(512, affine=False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.bn2 = BatchNorm1d(512, affine=False)

    def forward(self, x):
        x = self.bn1(x)
        x_norm = torch.norm(x, 2, 1, True)
        x_norm_mean = torch.mean(x_norm)
        weight = x_norm_mean / x_norm
        x = x * weight
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        feature = self.bn2(x)
        return feature


class GDC(Module):
    def __init__(self, embedding_size):
        super(GDC, self).__init__()
        self.conv_6_dw = Linear_block(
            512, 512, groups=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0)
        )
        self.conv_6_flatten = Flatten()
        self.linear = Linear(512, embedding_size, bias=False)
        # self.bn = BatchNorm1d(embedding_size, affine=False)
        self.bn = BatchNorm1d(embedding_size)
    def forward(self, x):
        x = self.conv_6_dw(x)
        x = self.conv_6_flatten(x)
        x = self.linear(x)
        x = self.bn(x)
        return x


class MobileFaceNet(Module):
    def __init__(
        self, input_size=(112,112), embedding_size=128, output_name="GDC", attention="none"
    ):
        super(MobileFaceNet, self).__init__()
        assert output_name in ["GNAP", "GDC"]
        assert input_size[0] in [112]

        self.conv1 = Conv_block(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Conv_block(
            64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64
        )
        self.conv_23 = Depth_Wise(
            64, 64, attention, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128
        )
        self.conv_3 = Residual(
            64,
            attention,
            num_block=4,
            groups=128,
            kernel=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.conv_34 = Depth_Wise(
            64, 128, attention, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256
        )
        self.conv_4 = Residual(
            128,
            attention,
            num_block=6,
            groups=256,
            kernel=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.conv_45 = Depth_Wise(
            128,
            128,
            attention,
            kernel=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            groups=512,
        )
        self.conv_5 = Residual(
            128,
            attention,
            num_block=2,
            groups=256,
            kernel=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.conv_6_sep = Conv_block(
            128, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0)
        )
        if output_name == "GNAP":
            self.output_layer = GNAP(512)
        else:
            self.output_layer = GDC(embedding_size)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)

        out = self.conv2_dw(out)

        out = self.conv_23(out)

        out = self.conv_3(out)

        out = self.conv_34(out)

        out = self.conv_4(out)

        out = self.conv_45(out)

        out = self.conv_5(out)

        conv_features = self.conv_6_sep(out)
        out = self.output_layer(conv_features)
        return out

def quantize_model(model, weight_bit=None, act_bit=None):
        """
		Recursively quantize a pretrained single-precision model to int8 quantized model
		model: pretrained single-precision model
		"""
        # if not (weight_bit) and not (act_bit ):
        #    weight_bit = self.settings.qw
        #    act_bit = self.settings.qa
        # quantize convolutional and linear layers
        if type(model) == nn.Conv2d:
            quant_mod = Quant_Conv2d(weight_bit=weight_bit)
            quant_mod.set_param(model)
            return quant_mod
        elif type(model) == nn.Linear:
            quant_mod = Quant_Linear(weight_bit=weight_bit)
            quant_mod.set_param(model)
            return quant_mod
        elif type(model) == nn.PReLU:
            quant_mod = QuantActPreLu(act_bit=act_bit)
            quant_mod.set_param(model)
            return quant_mod
        # quantize all the activation
        elif type(model) == nn.ReLU or type(model) == nn.ReLU6 or type(model) == nn.PReLU:
            return nn.Sequential(*[model, QuantAct(activation_bit=act_bit)])
        # recursively use the quantized module to replace the single-precision module
        elif type(model) == nn.Sequential or isinstance(model, nn.Sequential):
            mods = OrderedDict()
            for n, m in model.named_children():
                if isinstance(m, Depth_Wise) and m.residual:
                    mods[n] = nn.Sequential(
                        *[quantize_model(m, weight_bit=weight_bit, act_bit=act_bit), QuantAct(activation_bit=act_bit)])
                else:
                    mods[n] = quantize_model(m, weight_bit=weight_bit, act_bit=act_bit)
                # mods.append(self.quantize_model(m))
            return nn.Sequential(mods)
        else:
            q_model = copy.deepcopy(model)
            for attr in dir(model):
                mod = getattr(model, attr)
                if isinstance(mod, nn.Module) and 'norm' not in attr:
                    setattr(q_model, attr, quantize_model(mod, weight_bit=weight_bit, act_bit=act_bit))
            return q_model


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count
    
if __name__ == "__main__":

    net = MobileFaceNet()
    quant=quantize_model(net,8,8)
    print(quant)




