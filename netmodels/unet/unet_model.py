#!/usr/bin/python
# full assembly of the sub-parts to form the complete net

import torch
import torch.nn as nn
import torch.nn.functional as F

# python 3 confusing imports :(
from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear):
        super(UNet, self).__init__()
        self.inc = inconv(in_ch, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256, bilinear=bilinear)
        self.up2 = up(512, 128, bilinear=bilinear)
        self.up3 = up(256, 64, bilinear=bilinear)
        self.up4 = up(128, 64, bilinear=bilinear)
        self.outc = outconv(64, out_ch)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        x = self.tanh(x)
        return x


class AttriAiC(nn.Module):
    def __init__(self, in_ch, out_ch, n_attributes, bilinear, img_size=(320, 128)):
        super(AttriAiC, self).__init__()

        self.n_attributes = n_attributes
        self.img_size = img_size
        self.linear_attr = nn.Linear(self.n_attributes, self.n_attributes*(self.img_size[0]//16)*(self.img_size[1]//16))

        self.inc = inconv(in_ch, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024+self.n_attributes, 256, bilinear=bilinear)
        self.up2 = up(512, 128, bilinear=bilinear)
        self.up3 = up(256, 64, bilinear=bilinear)
        self.up4 = up(128, 64, bilinear=bilinear)
        self.outc = outconv(64, out_ch)
        self.tanh = nn.Tanh()

    def forward(self, x, x_attributes):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x_attributes = self.linear_attr(x_attributes)
        x_attributes = x_attributes.view(x.shape[0], self.n_attributes, self.img_size[0]//16, self.img_size[1]//16)
        x5 = torch.cat([x5, x_attributes], 1)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        x = self.tanh(x)
        return x


class AttriRAP(nn.Module):
    def __init__(self, n_channels, n_classes, number_attr, bilinear=True):
        super(AttriRAP, self).__init__()
        self.inc = inconv(n_channels, 64)
        # x1 = size(H,W) ==> (320, 128)
        self.down1 = down(64, 128)
        # x2 = size(H/2, W/2)
        self.down2 = down(128, 256)
        # x3 = size(H/4, W/4)
        self.down3 = down(256, 512)
        # x3 = size(H/8, W/8)
        self.down4 = down(512, 512)
        # x4 = size(H/16, W/16) ==>(20, 8)
        self.up1 = up(1024+number_attr, 256, bilinear=bilinear)
        self.up2 = up(512, 128, bilinear=bilinear)
        self.up3 = up(256, 64, bilinear=bilinear)
        self.up4 = up(128, 64, bilinear=bilinear)
        self.outc = outconv(64, n_classes)
        self.tanh = nn.Tanh()

    def forward(self, x, x_concat):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x6 = torch.cat([x5, x_concat], 1)

        x = self.up1(x6, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        x = self.tanh(x)
        return x
