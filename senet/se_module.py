from torch import nn
import torch

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(MALayer, self).__init__()
        self.shuffledown = Shuffle_d(4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel*16, channel*16 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel*16 // reduction, channel*16, bias=False),
            nn.Sigmoid()
        )
        self.shuffleup = nn.PixelShuffle(4)

    def forward(self, x):
        ex_x = self.shuffledown(x)
        b, c, _, _ = ex_x.size()
        y = self.avg_pool(ex_x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        ex_x = ex_x * y.expand_as(ex_x)
        x = self.shuffleup(ex_x)
        # buff_error = buff_x - x
        # buff_error = buff_x - x
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        # return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
        # return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
        # return torch.mean(x,1)
        return torch.mean(x, 1).unsqueeze(1)

class Shuffle_d(nn.Module):
    def __init__(self, scale=2):
        super(Shuffle_d, self).__init__()
        self.scale = scale

    def forward(self, x):
        def _space_to_channel(x, scale):
            b, C, h, w = x.size()
            Cout = C * scale ** 2
            hout = h // scale
            wout = w // scale
            x = x.contiguous().view(b, C, hout, scale, wout, scale)
            x = x.contiguous().permute(0, 1, 3, 5, 2, 4)
            x = x.contiguous().view(b, Cout, hout, wout)
            return x
        return _space_to_channel(x, self.scale)