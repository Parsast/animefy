import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.nn.utils.parametrizations import spectral_norm
from torchvision.models import vgg19
from torch.nn import L1Loss


class CustomConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding_type='reflect', use_bias=True, sn=False):
        super(CustomConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding_type = padding_type
        self.use_bias = use_bias
        self.sn = sn
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=use_bias, padding=0)  # No auto padding
        if sn:
            self.conv = spectral_norm(self.conv)

    def forward(self, x):
        # Calculate padding
        if (self.kernel_size - self.stride) % 2 == 0:
            pad = (self.kernel_size - self.stride) // 2
            padding = (pad, pad, pad, pad)  # Padding for left, right, top, bottom
        else:
            pad = (self.kernel_size - self.stride) // 2
            pad_extra = (self.kernel_size - self.stride) - pad
            padding = (pad, pad + pad_extra, pad, pad + pad_extra)  # Adjusted for asymmetric padding
        
        # Apply padding
        if self.padding_type == 'reflect':
            x = F.pad(x, padding, mode='reflect')
        elif self.padding_type == 'zero':
            x = F.pad(x, padding, mode='constant', value=0)
        
        x = self.conv(x)
        return x


class LADE(nn.Module):
    def __init__(self, channels):
        super(LADE, self).__init__()
        self.conv = CustomConv2d(channels, channels, kernel_size=1, use_bias=False)

    def forward(self, x):
        eps = 1e-5
        tx = self.conv(x)
        t_mean = tx.mean([2, 3], keepdim=True)
        t_var = tx.var([2, 3], keepdim=True, unbiased=False)
        x_mean = x.mean([2, 3], keepdim=True)
        x_var = x.var([2, 3], keepdim=True, unbiased=False)
        x_normalized = (x - x_mean) / torch.sqrt(x_var + eps)
        x_rescaled = x_normalized * torch.sqrt(t_var + eps) + t_mean
        return x_rescaled

class LADE_D(nn.Module):
    def __init__(self, channels):
        super(LADE_D, self).__init__()
        self.conv = CustomConv2d(channels, channels, kernel_size=1, use_bias=False , sn=True)

    def forward(self, x):
        eps = 1e-5
        tx = self.conv(x)
        t_mean = tx.mean([2, 3], keepdim=True)
        t_var = tx.var([2, 3], keepdim=True, unbiased=False)
        x_mean = x.mean([2, 3], keepdim=True)
        x_var = x.var([2, 3], keepdim=True, unbiased=False)
        x_normalized = (x - x_mean) / torch.sqrt(x_var + eps)
        x_rescaled = x_normalized * torch.sqrt(t_var + eps) + t_mean
        return x_rescaled

class ConvLADELReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, use_bias=True, alpha=0.2):
        super(ConvLADELReLU, self).__init__()
        self.conv = CustomConv2d(in_channels, out_channels, kernel_size, stride, use_bias=use_bias)
        self.lade = LADE(out_channels)
        self.alpha = alpha

    def forward(self, x):
        x = self.conv(x)
        x = self.lade(x)
        x = F.leaky_relu(x, negative_slope=self.alpha)
        return x



class ExternalAttentionV3(nn.Module):
    def __init__(self, channel, k=128):
        super(ExternalAttentionV3, self).__init__()
        self.k = k
        self.channel = channel
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=1)
        self.conv2 = nn.Conv1d(channel, k, kernel_size=1)  # Attention mechanism
        self.conv_transpose = nn.Conv1d(k, channel, kernel_size=1)  # Transposed operation
        self.conv3 = nn.Conv2d(channel, channel, kernel_size=1)
        self.batch_norm = nn.BatchNorm2d(channel)
        
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        b, c, h, w = x.size()
        x = x.view(b, c, -1)
        att = self.conv2(x)
        att = F.softmax(att, dim=1)
        att = att / (1e-9 + torch.sum(att, dim=2, keepdim=True))
        x = self.conv_transpose(att)
        x = x.view(b, c, h, w)
        x = self.conv3(x)
        x = self.batch_norm(x)
        x = F.leaky_relu(x + identity)  # 
        return x
        

# loss function



class VGG_FeatureExtractor(nn.Module):
    def __init__(self, layer, device='cpu'):
        super(VGG_FeatureExtractor, self).__init__()
        self.layer = layer
        self.model = vgg19(pretrained=True).features[:self.layer + 1]
        for param in self.model.parameters():
            param.requires_grad = False
        self.model = self.model.to(device).eval()

    def forward(self, x):
        return self.model(x)

class VGG_LOSS(nn.Module):
    def __init__(self, device='cpu'):
        super(VGG_LOSS, self).__init__()
        self.vgg = VGG_FeatureExtractor(25, device)
        self.criterion = L1Loss()

    def forward(self, x, y):
        x_vgg = self.vgg(x)
        y_vgg = self.vgg(y)
        loss = self.criterion(x_vgg, y_vgg) / x_vgg.shape[1]
        return loss

class con_loss(nn.Module):
    def __init__(self, device='cpu', weight=0.5):
        super(con_loss, self).__init__()
        self.weight = weight
        self.criterion = VGG_LOSS(device)
        self.device = device

    def forward(self, x, y):
        loss = self.criterion(x.to(self.device), y.to(self.device))
        return self.weight * loss

