import torch
import torch.nn as nn
import torch.nn.functional as F
from ops_pytorch import CustomConv2d, LADE_D



class D_net(nn.Module):
    def __init__(self,input, ch, sn=False):
        super(D_net, self).__init__()
        self.in_channels = input
        self.channel = ch
        self.sn = sn
        self.conv_0 = CustomConv2d(self.in_channels, ch, kernel_size=7, stride=1, sn=sn)
        self.conv_s2_0 = CustomConv2d(ch, ch, kernel_size=3, stride=2, sn=sn)
        self.LADE_D_0 = LADE_D(ch)
        self.conv_s1_0 = CustomConv2d(ch, ch*2, kernel_size=3, stride=1, sn=sn)
        self.LADE_D_1 = LADE_D(ch*2)
        self.conv_s2_1 = CustomConv2d(ch*2, ch*2, kernel_size=3, stride=2, sn=sn)
        self.LADE_D_2 = LADE_D(ch*2)
        self.conv_s1_1 = CustomConv2d(ch*2, ch*4, kernel_size=3, stride=1, sn=sn)
        self.LADE_D_3 = LADE_D(ch*4)
        self.conv_s2_2 = CustomConv2d(ch*4, ch*4, kernel_size=3, stride=2, sn=sn)
        self.LADE_D_4 = LADE_D(ch*4)
        self.conv_s1_2 = CustomConv2d(ch*4, ch*8, kernel_size=3, stride=1, sn=sn)
        self.LADE_D_5 = LADE_D(ch*8)
        self.conv_s2_3 = CustomConv2d(ch*8, ch*8, kernel_size=3, stride=2, sn=sn)
        self.LADE_D_6 = LADE_D(ch*8)
        self.D_logit = CustomConv2d(ch*8, 1, kernel_size=1, stride=1, sn=sn)

    def forward(self, x):
        x = F.leaky_relu(self.conv_0(x), 0.2)
        x = F.leaky_relu(self.LADE_D_0(self.conv_s2_0(x)), 0.2)
        x = F.leaky_relu(self.LADE_D_1(self.conv_s1_0(x)), 0.2)
        x = F.leaky_relu(self.LADE_D_2(self.conv_s2_1(x)), 0.2)
        x = F.leaky_relu(self.LADE_D_3(self.conv_s1_1(x)), 0.2)
        x = F.leaky_relu(self.LADE_D_4(self.conv_s2_2(x)), 0.2)
        x = F.leaky_relu(self.LADE_D_5(self.conv_s1_2(x)), 0.2)
        x = F.leaky_relu(self.LADE_D_6(self.conv_s2_3(x)), 0.2)
        x = self.D_logit(x)
        return x
    
    