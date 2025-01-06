import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):

    def __init__(self):
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

        self.uplayer5 = self.deconv_block(1024, 512, dropout=True)
        self.uplayer4 = self.deconv_block(512 + 512, 256)
        self.uplayer3 = self.deconv_block(256 + 256, 128)
        self.uplayer2 = self.deconv_block(128, 64)
        self.uplayer1 = nn.Sequential(nn.Conv2d(64, 3, kernel_size=1))
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

    def deconv_block(self, in_channels, out_channels, dropout=False):
        if dropout:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.Dropout(0.5),
                nn.ReLU(inplace=True),
                #nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                #nn.BatchNorm2d(out_channels),
                #nn.ReLU(inplace=True),
            )
        else:
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

class Discriminator(nn.Module):
    def __init__(self, d=64):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(6, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 1, 1)
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, 1, 4, 1, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
    
    def initialize_weights(self):
        layers_to_init = [self.conv1, self.conv2, self.conv2_bn, 
            self.conv3, self.conv3_bn, self.conv4, 
            self.conv4, self.conv4_bn, self.conv5]
        for module in layers:
            if isinstance(module, nn.ConvTranspose2d)or isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, mode='fan_in')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                        
    # forward method
    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.sigmoid(self.conv5(x))

        return x

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()