import torch
import torch.nn as nn
import torch.nn.functional as F

class FullyConvNetwork(nn.Module):

    def __init__(self, n_classes):
        super().__init__()
         # Encoder (Convolutional Layers)
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7, stride=1, padding=(4,5)),  # Input channels: 3, Output channels: 8
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=(3,3)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # pool1
        )

        ### FILL: add more CONV Layers
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # pool2
        ) 
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # pool3
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # pool4
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # pool5
        )

        self.layer6 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        ) # layer6, layer7 ==> score32

        self.score16 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.score8 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        # Decoder (Deconvolutional Layers)
        ### FILL: add ConvTranspose Layers
        ### None: since last layer outputs RGB channels, may need specific activation function
        self.upconv3 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.upconv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=[2,3]),
            #nn.BatchNorm2d(),
            #nn.ReLU(inplace=True),
        )

        # Final layer (No activation function, as it's used for segmentation)
        self.final_conv = nn.Sequential(nn.Conv2d(64, n_classes, kernel_size=1))  # Assuming 20 classes + 1 background

        self.initialize_weights()

    def initialize_weights(self):
        layers_to_init = [self.layer1, self.layer2, self.layer3, 
            self.layer4, self.layer5, self.layer6, 
            self.layer7, self.upconv1, self.upconv2, 
            self.upconv3, self.final_conv]
        for layers in layers_to_init:
            for module in layers:
                if isinstance(module, nn.ConvTranspose2d)or isinstance(module, nn.Conv2d):
                    nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # Encoder forward pass
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        #feat_8s = x

        x = self.layer4(x)
        #feat_16s = x

        x = self.layer5(x)
        #feat_32s = x

        # Decoder forward pass
        x = self.layer6(x)
        x = self.layer7(x)
        
        x = self.upconv3(x) #+ self.score16(feat_16s)
        x = self.upconv2(x) #+ self.score8(feat_8s)

        x = self.upconv1(x)
        ### FILL: encoder-decoder forward pass

        # Final layer
        output = self.final_conv(x)
        return output

class FullyConvNetwork_facades(nn.Module):

    def __init__(self, n_classes):
        super().__init__()
        # Final layer (No activation function, as it's used for segmentation)
        self.layer1 = self.conv_block_double(3, 64)
        self.layer2 = self.conv_block_double(64, 128)
        self.layer3 = self.conv_block_double(128, 256)
        self.layer4 = self.conv_block_double(256, 512)
        self.layer5 = self.conv_block_double(512, 1024)

        self.layer6 = self.conv_block(1024, 1024, 7, 3)
        self.layer7 = self.conv_block(1024, 1024, 1, 0) # layer6, layer7 ==> score32

        #self.score16 = self.conv_block(512, 512, 1, 0)
        #self.score8 = self.conv_block(256, 256, 1, 0)

        self.uplayer5 = self.deconv_block(1024, 512)
        self.uplayer4 = self.deconv_block(512 + 512, 256)
        self.uplayer3 = self.deconv_block(256 + 256, 128)
        self.uplayer2 = self.deconv_block(128, 64)
        self.uplayer1 = nn.Sequential(nn.Conv2d(64, n_classes, kernel_size=1))
        self.initialize_weights()

    def conv_block(self, in_channels, out_channels, kernel_size=3, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def conv_block_double(self, in_channels, out_channels, kernel_size=3, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            #nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            #nn.BatchNorm2d(out_channels),
            #nn.ReLU(inplace=True),
        )

    def deconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            #nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(out_channels),
            #nn.ReLU(inplace=True),
        )

    def initialize_weights(self):
        layers_to_init = [self.layer1, self.layer2, self.layer3, 
            self.layer4, self.layer5, self.layer6, 
            self.layer7, self.uplayer1, self.uplayer2, 
            self.uplayer3, self.uplayer4, self.uplayer5]
        for layers in layers_to_init:
            for module in layers:
                if isinstance(module, nn.ConvTranspose2d)or isinstance(module, nn.Conv2d):
                    nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # Encoder forward pass
        x = self.layer1(x)
        x = self.layer2(F.max_pool2d(x, 2))
        x = self.layer3(F.max_pool2d(x, 2))
        feat_8s = x

        x = self.layer4(F.max_pool2d(x, 2))
        feat_16s = x

        x = self.layer5(F.max_pool2d(x, 2))
        #feat_32s = x

        x = self.layer6(x)
        x = self.layer7(x)
        
        # Decoder forward pass
        x = torch.cat([self.uplayer5(x), feat_16s], dim=1)
        x = torch.cat([self.uplayer4(x), feat_8s], dim=1)

        x = self.uplayer3(x)
        x = self.uplayer2(x)
        x = self.uplayer1(x)
        ### FILL: encoder-decoder forward pass

        # Final layer
        output = F.sigmoid(x) * 2.0 - 1.0
        return output
