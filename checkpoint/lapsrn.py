import torch.nn as nn
import numpy as np
import math
from senet.se_module import SELayer, MALayer
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn.modules.utils import _single, _pair, _triple

def get_upsample_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * \
             (1 - abs(og[1] - center) / factor)
    return torch.from_numpy(filter).float()


def get_WB_filter(size):
    """make a 2D weight bilinear kernel suitable for WB_Conv"""
    ligne = []
    colonne = []
    for i in range(size):
        if (i + 1) <= np.floor(math.sqrt(16)):
            ligne.append(i + 1)
            colonne.append(i + 1)
        else:
            ligne.append(ligne[i - 1] - 1.0)
            colonne.append(colonne[i - 1] - 1.0)
    BilinearFilter = np.zeros(size * size)
    for i in range(size):
        for j in range(size):
            BilinearFilter[(j + i * size)] = (ligne[i] * colonne[j] / 16)
    filter0 = np.reshape(BilinearFilter, (7, 7))
    return torch.from_numpy(filter0).float()


class _Conv_Block(nn.Module):
    def __init__(self):
        super(_Conv_Block, self).__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.cov_block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
        )

    def forward(self, x):
        residual = x
        output = self.cov_block(x)
        output += residual
        output = self.relu(output)
        return output

class _Conv_attention_Block(nn.Module):
    def __init__(self):
        super(_Conv_attention_Block, self).__init__()
        self.ma = MALayer(64, 4)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.cov_block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
        )

    def forward(self, x):
        residual = x
        output = self.cov_block(x)
        output = self.ma(output)
        output += residual
        output = self.relu(output)
        return output

class branch_block_front(nn.Module):
    def __init__(self):
        super(branch_block_front, self).__init__()
        # self.relu = nn.LeakyReLU(0.2, inplace=False)
        self.se = MALayer(16, 4)
        # self.se = SELayer(16, 4)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        # self.front_conv_input = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        x = self.se(x)
        x = self.relu(x)
        # x = self.front_conv_input(x)
        return x


class branch_block_back(nn.Module):
    def __init__(self):
        super(branch_block_back, self).__init__()
        # self.relu = nn.LeakyReLU(0.2, inplace=True)
        # self.se = SELayer(64, 16)
        self.cov_block = nn.Sequential(
            # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True),
        )

    def forward(self, x):
        # x = self.relu(x)
        output = self.cov_block(x)
        return output


class Pos2Weight(nn.Module):
    def __init__(self, outC=16, kernel_size=5, inC=1):
        super(Pos2Weight, self).__init__()
        self.inC = inC
        self.kernel_size = kernel_size
        self.outC = outC
        self.meta_block = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.kernel_size * self.kernel_size * self.inC * self.outC)
        )

    def forward(self, x):
        output = self.meta_block(x)
        return output

class _ConvMultiW(nn.Module):

    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias']

    def __init__(self, in_channels, out_channels, kernel_size, msfa_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvMultiW, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.msfa_size = msfa_size
        self.weight = Parameter(torch.Tensor(msfa_size[0]*msfa_size[1], in_channels*kernel_size[0]*kernel_size[1], out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(msfa_size[0]*msfa_size[1], out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    # def extra_repr(self):
    #     s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
    #          ', stride={stride}')
    #     if self.padding != (0,) * len(self.padding):
    #         s += ', padding={padding}'
    #     if self.dilation != (1,) * len(self.dilation):
    #         s += ', dilation={dilation}'
    #     if self.output_padding != (0,) * len(self.output_padding):
    #         s += ', output_padding={output_padding}'
    #     if self.groups != 1:
    #         s += ', groups={groups}'
    #     if self.bias is None:
    #         s += ', bias=False'
    #     return s.format(**self.__dict__)

class ConvMosaic(_ConvMultiW):

    def __init__(self, in_channels, out_channels, kernel_size, msfa_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        msfa_size = _pair(msfa_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(ConvMosaic, self).__init__(
            in_channels, out_channels, kernel_size, msfa_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

    def forward(self, input):
        N, inC, H, W = input.size()
        cols = nn.functional.unfold(input, self.kernel_size, padding=self.padding)
        scale_int = 1
        cols = cols.contiguous().view(cols.size(0) // (scale_int ** 2), scale_int ** 2, cols.size(1), cols.size(2),
                                      1).permute(0, 1, 3, 4, 2).contiguous()
        spe_num, kernel_size_and_inC, outC = self.weight.size()
        self.weight1 = self.weight.view(1, spe_num, kernel_size_and_inC, outC)
        self.weight1 = torch.cat([self.weight1] * ((H*W)//spe_num), 0)
        self.weight2 = self.weight1.view(1, H*W, kernel_size_and_inC, outC)
        buff_weight = self.weight2.cpu().detach().numpy()
        # buff_weight_1 = buff_weight[0, 0, :, :]
        # buff_weight_2 = buff_weight[0, 1, :, :]
        # buff_weight_3 = buff_weight[0, 7, :, :]
        # buff_weight_4 = buff_weight[0, 8, :, :]
        # buff_weight_5 = buff_weight[0, 9, :, :]
        # buff_weight_1 = buff_weight[0, 0, :, :]
        # buff_weight_2 = buff_weight[0, 1, :, :]
        # buff_weight_3 = buff_weight[0, 15, :, :]
        # buff_weight_4 = buff_weight[0, 16, :, :]
        # buff_weight_5 = buff_weight[0, 17, :, :]
        Raw_conv = torch.matmul(cols, self.weight2).permute(0, 1, 4, 2, 3)
        Raw_conv = Raw_conv.contiguous().view(N, scale_int, scale_int, self.out_channels, H, W).permute(
            0, 3, 4, 1, 5, 2)
        Raw_conv = Raw_conv.contiguous().view(N, self.out_channels, scale_int * H, scale_int * W)
        self.bias1 = self.bias.view(1, spe_num, outC)
        self.bias1 = torch.cat([self.bias1] * ((H*W)//spe_num), 0)
        self.bias2 = self.bias1.view(1, H*W, outC)
        return Raw_conv

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.scale = 1
        self.outC = 16
        self.WB_Conv = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=7, stride=1, padding=3, bias=False, groups=16)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.front_conv_input = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1,
                                          bias=True)
        self.convt_br1_front = self.make_layer(branch_block_front)
        self.convt_F1 = self.make_layer(_Conv_attention_Block)
        self.convt_F2 = self.make_layer(_Conv_attention_Block)
        self.convt_br1_back = self.make_layer(branch_block_back)
        self.P2W = Pos2Weight(outC=self.outC)
        # self.mosaic_conv = ConvMosaic(in_channels=1, out_channels=16, kernel_size=5, msfa_size=4,stride=1, padding=2,bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.kernel_size[0] == 7:
                    c1, c2, h, w = m.weight.data.size()
                    WB = get_WB_filter(h)
                    for i in m.parameters():
                        i.requires_grad = False
                    m.weight.data = WB.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block):
        layers = []
        layers.append(block())
        return nn.Sequential(*layers)

    def forward_once(self, x):
        x = self.front_conv_input(x)
        out = self.convt_F1(x)
        out = self.convt_F2(out)
        return out

    def repeat_y(self, y):
        scale_int = math.ceil(self.scale)
        N, C, H, W = y.size()
        y = y.view(N, C, H, 1, W, 1)

        y = torch.cat([y] * scale_int, 3)
        y = torch.cat([y] * scale_int, 5).permute(0, 3, 5, 1, 2, 4)

        return y.contiguous().view(-1, C, H, W)

    def forward(self, data, pos_mat):
        x, y = data
        WB_norelu = self.WB_Conv(x)
        # y = torch.sum(x,1)
        # y = y.contiguous().view(y.size(0),1 ,y.size(1), y.size(2))
        # buff_x = x[0, 0, :, :].cpu().numpy()
        # buff_x1 = x[0, 3, :, :].cpu().numpy()
        # buff_y = y[0, 0, :, :].cpu().numpy()
        # buff_z = z[0, 0, :, :].cpu().numpy()
        local_weight = self.P2W(pos_mat.view(pos_mat.size(1), -1))
        up_y = self.repeat_y(y)
        cols = nn.functional.unfold(up_y, 5, padding=2)
        scale_int = math.ceil(self.scale)
        cols = cols.contiguous().view(cols.size(0) // (scale_int ** 2), scale_int ** 2, cols.size(1), cols.size(2),
                                      1).permute(0, 1, 3, 4, 2).contiguous()
        local_weight = local_weight.contiguous().view(y.size(2), scale_int, y.size(3), scale_int, -1,
                                                      self.outC).permute(1, 3, 0, 2, 4, 5).contiguous()
        local_weight = local_weight.contiguous().view(scale_int ** 2, y.size(2) * y.size(3), -1, self.outC)
        Raw_conv = torch.matmul(cols, local_weight).permute(0, 1, 4, 2, 3)
        Raw_conv = Raw_conv.contiguous().view(y.size(0), scale_int, scale_int, self.outC, y.size(2), y.size(3)).permute(
            0, 3, 4, 1, 5, 2)
        Raw_conv = Raw_conv.contiguous().view(y.size(0), self.outC, scale_int * y.size(2), scale_int * y.size(3))

        # Raw_conv1 = self.mosaic_conv(y)
        Raw_conv = self.relu(Raw_conv)
        convt_br1_front = self.convt_br1_front(Raw_conv)
        convt_br1_temp = self.forward_once(convt_br1_front)
        convt_br1_back = self.convt_br1_back(convt_br1_temp)
        HR_4x = convt_br1_back
        # return HR_4x
        return  torch.add(HR_4x, WB_norelu)
        # return WB_norelu


class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.sum(error)
        return loss