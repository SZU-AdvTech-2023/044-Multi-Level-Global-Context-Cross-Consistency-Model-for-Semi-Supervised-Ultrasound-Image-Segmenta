import torch
import torch.nn as nn

class BAGModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BAGModule, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, dilation=1)

        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=1, padding=1)

        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=2, padding=2)

        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=4, padding=4)

        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=6, padding=6)

        self.conv6 = nn.Conv2d(out_channels * 5, in_channels, kernel_size=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out4 = self.conv4(x)
        out5 = self.conv5(x)

        concatenated = torch.cat([out1, out2, out3, out4, out5], dim=1)

        out6 = self.conv6(concatenated)

        out7 = self.sigmoid(out6) * x

        return out7

