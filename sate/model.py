import torch
from torch import nn

class UNetModel(nn.Module):

    def __init__(self, config):
        super(UNetModel, self).__init__()

        self.in_channels   = config['in_channels']
        self.out_channels  = config['out_channels']
        self.mul = config['mul_channels']

        def double_conv(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.GroupNorm(32, out_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.GroupNorm(32, out_channels),
                nn.SiLU(inplace=True),
            )

        def down(in_channels, out_channels):
            return nn.Sequential(
                nn.MaxPool2d(2),
                double_conv(in_channels, out_channels)
            )
   
        def up(in_channels, out_channels):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                double_conv(in_channels, out_channels)
            )

        self.inc = double_conv(self.in_channels, self.mul)
        self.down1 = down(self.mul, self.mul * 2)
        self.down2 = down(self.mul * 2, self.mul * 4)
        self.down3 = down(self.mul * 4, self.mul * 8)
        self.down4 = down(self.mul * 8, self.mul * 8)
        self.up0 = up(self.mul * 8, self.mul * 8) 
        self.up1 = up(self.mul * 16, self.mul * 4)
        self.up2 = up(self.mul * 8, self.mul * 2)
        self.up3 = up(self.mul * 4, self.mul)
        self.out = nn.Sequential(
                double_conv(self.mul * 2, self.mul * 2),
                nn.Conv2d(self.mul * 2, self.mul * 2, kernel_size=1),
                nn.Conv2d(self.mul * 2, self.out_channels, kernel_size=1)
        )
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up0(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.up1(x)
        x = torch.cat([x, x3], dim=1)
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dropout(x)
        x = self.out(x)

        return x 
