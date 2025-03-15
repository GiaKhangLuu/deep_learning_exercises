import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsamplingBlock, self).__init__()
        self.double_conv = DoubleConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        skip_connection = self.double_conv(x)
        p = self.pool(skip_connection)
        p = self.dropout(p)
        return skip_connection, p

class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsamplingBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.double_conv = DoubleConvBlock(in_channels, out_channels)

    def forward(self, x, skip_connection):
        x = self.upconv(x)
        x = torch.cat([x, skip_connection], dim=1)
        x = self.double_conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.down1 = DownsamplingBlock(in_channels, 64)
        self.down2 = DownsamplingBlock(64, 128)
        self.down3 = DownsamplingBlock(128, 256)
        self.down4 = DownsamplingBlock(256, 512)
        self.bottleneck = DoubleConvBlock(512, 1024)
        self.up1 = UpsamplingBlock(1024, 512)
        self.up2 = UpsamplingBlock(512, 256)
        self.up3 = UpsamplingBlock(256, 128)
        self.up4 = UpsamplingBlock(128, 64)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        f1, d1 = self.down1(x)
        f2, d2 = self.down2(d1)
        f3, d3 = self.down3(d2)
        f4, d4 = self.down4(d3)
        bottleneck = self.bottleneck(d4)
        u1 = self.up1(bottleneck, f4)
        u2 = self.up2(u1, f3)
        u3 = self.up3(u2, f2)
        u4 = self.up4(u3, f1)
        outputs = self.final_conv(u4)
        return outputs

if __name__ == "__main__":
    in_channels = 3
    num_classes = 10
    model = UNet(in_channels, num_classes)

    inputs = torch.randn(1, in_channels, 128, 128)
    outputs = model(inputs)

    print(outputs.shape)