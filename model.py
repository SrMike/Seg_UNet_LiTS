# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 18:05:31 2021

@author: Miguel
"""
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class SimpleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))
            
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)
## new 
class TripleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TripleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class TC_UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(TC_UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(TripleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(TripleConv(feature*2, feature))

        self.bottleneck = TripleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)


def test():
    x = torch.randn((3, 4, 161, 161))
    model = UNET(in_channels=4, out_channels=4)
    preds = model(x)
    print(preds.shape, x.shape)
    assert preds.shape == x.shape

# SegNet

class SEGNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(SEGNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices = True )
        self.unpool = nn.MaxUnpool2d(kernel_size = 2, stride=2)
        # Down part of SegNet
        
        self.downs.append(DoubleConv(in_channels, features[0]))
        self.downs.append(DoubleConv(features[0], features[1]))
        self.downs.append(TripleConv(features[1], features[2]))
        self.downs.append(TripleConv(features[2], features[3]))
        self.downs.append(TripleConv(features[3], features[3]*2))


        self.ups.append(TripleConv(features[3]*2,features[3]))
        self.ups.append(TripleConv(features[3],features[2]))
        self.ups.append(TripleConv(features[2],features[1]))
        self.ups.append(DoubleConv(features[1],features[0]))
        self.ups.append(SimpleConv(features[0], features[0]))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size = 1)
        # Down part of UNET
        #for feature in features:
        #    self.downs.append(DoubleConv(in_channels, feature))
        #    in_channels = feature

        # Up part of UNET
        #for feature in reversed(features):
        #    self.ups.append(
        #        nn.ConvTranspose2d(
        #            feature*2, feature, kernel_size=2, stride=2,
        #        )
        #    )
        #    self.ups.append(DoubleConv(feature*2, feature))

        #self.bottleneck = TripleConv(features[3], features[-1]*2)
        #self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        indices = []
        #skip_connections = []
        for down in self.downs:
            x = down(x)
            
            x, indice = self.pool(x)
            indices.append(indice)

        #x = self.bottleneck(x)
        #skip_connections = skip_connections[::-1]
        indices = indices[::-1]
        
        for i,up in enumerate(self.ups):
          x = self.unpool(x,indices[i])
          x = up(x)
        x = self.final_conv(x)
        #for idx in range(0, len(self.ups), 2):
            #x = self.ups[idx](x)
            #skip_connection = skip_connections[idx//2]

            #if x.shape != skip_connection.shape:
            #    x = TF.resize(x, size=skip_connection.shape[2:])

            #concat_skip = torch.cat((skip_connection, x), dim=1)
            #x = self.ups[idx+1](concat_skip)

        return x


if __name__ == "__main__":
    test()


