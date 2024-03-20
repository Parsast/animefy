import torch
import torch.nn as nn
import torch.nn.functional as F
from ops_pytorch import CustomConv2d, LADE, ConvLADELReLU,ExternalAttentionV3

class G_net(nn.Module):
    def __init__(self, inputs_channels):
        super(G_net, self).__init__()
        # base
        self.x0 = ConvLADELReLU(inputs_channels, 32, 7)        # 256
        self.x1 = ConvLADELReLU(32, 64, stride=2)    # 128
        self.x1_1 = ConvLADELReLU(64, 64)
        self.x2 = ConvLADELReLU(64, 128, stride=2)
        self.x2_1 = ConvLADELReLU(128, 128)
        self.x3 = ConvLADELReLU(128,128, stride=2)
        self.x3_1 = ConvLADELReLU(128, 128)
        # support
        self.s_x3 = ExternalAttentionV3(128)
        self.s_x4 = ConvLADELReLU(128, 128)
        self.s_x4_1 = ConvLADELReLU(128, 128)
        self.s_x5 = ConvLADELReLU(128, 64)
        self.s_x5_1 = ConvLADELReLU(64, 64)
        self.s_x6 = ConvLADELReLU(64, 32)
        self.s_x6_1 = ConvLADELReLU(32, 32)
        self.s_final = CustomConv2d(32, 3, kernel_size=7, use_bias=True)
        # self.fake_s = torch.tanh(self.s_final, name='out_layer')
        # main
        self.m_x3 = ExternalAttentionV3(128)
        self.m_x4 = ConvLADELReLU(128, 128)
        self.m_x4_1 = ConvLADELReLU(128, 128)
        self.m_x5 = ConvLADELReLU(128, 64)
        self.m_x5_1 = ConvLADELReLU(64, 64)
        self.m_x6 = ConvLADELReLU(64, 32)
        self.m_x6_1 = ConvLADELReLU(32, 32)
        self.m_final = CustomConv2d(32, 3, kernel_size=7, use_bias=True)
        # self.fake_m = torch.tanh(self.m_final, name='out_layer')

    def forward(self, x):
        # Base layer sequence
        x0 = self.x0(x)
        x1 = self.x1(x0)
        x1_1 = self.x1_1(x1)
        x2 = self.x2(x1)
        x2_1 = self.x2_1(x2)
        x3 = self.x3(x2)
        x3_1 = self.x3_1(x3)
        
        # Support pathway
        s_x3 = self.s_x3(x3)
        s_x4 = F.interpolate(s_x3, scale_factor=2)
        s_x4 = self.s_x4(s_x4)
        s_x4 += x2  # Element-wise addition
        s_x4_1 = self.s_x4_1(s_x4)
        s_x5 = F.interpolate(s_x4, scale_factor=2)
        s_x5 = self.s_x5(s_x5)
        s_x5 += x1  # Element-wise addition
        s_x5_1 = self.s_x5_1(s_x5)
        s_x6 = F.interpolate(s_x5, scale_factor=2)
        s_x6 = self.s_x6(s_x6)
        s_x6 += x0  # Element-wise addition
        s_x6_1 = self.s_x6_1(s_x6)
        s_final = self.s_final(s_x6)
        fake_s = torch.tanh(s_final)
    
        # Main pathway
        m_x3 = self.m_x3(x3)
        m_x4 = F.interpolate(m_x3, scale_factor=2)
        m_x4 += x2
        m_x4_1 = self.m_x4(m_x4)
        m_x5 = F.interpolate(m_x4, scale_factor=2)
        m_x5 = self.m_x5(m_x5)
        m_x5 += x1
        m_x5 = self.m_x5_1(m_x5)
        m_x6 = F.interpolate(m_x5, scale_factor=2)
        m_x6 = self.m_x6(m_x6)
        m_x6 += x0
        m_x6_1 = self.m_x6_1(m_x6)
        m_final = self.m_final(m_x6_1)
        fake_m = torch.tanh(m_final)

        return fake_s, fake_m