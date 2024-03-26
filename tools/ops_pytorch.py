import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.nn.utils.parametrizations import spectral_norm
from torchvision.models import vgg19
from torch.nn import L1Loss, MSELoss
from pytorch_color_ops import  rgb_to_lab


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

class generator_loss(nn.Module):
    def __init__(self):
        super(generator_loss, self).__init__()
        self.criterion = nn.MSELoss()  # Use Mean Squared Error Loss

    def forward(self, fake):
        # The target tensor is a tensor full of 0.9 with the same shape as 'fake'
        target = 0.9 * torch.ones_like(fake) 
        loss = self.criterion(fake, target)
        return loss

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

def gram(x):
    b, c, h, w = x.size()
    f = x.view(b, c, h * w)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / float((c * h * w))
    return G

class style_loss_decentralization_3(nn.Module):
    def __init__(self, device='cpu', weight=[1.0, 1.0, 1.0]):
        super(style_loss_decentralization_3, self).__init__()
        self.device = device
        self.weight = weight
        self.criterion = L1Loss()

    def forward(self, style, fake):
        style_2 = VGG_FeatureExtractor(5, self.device)(style)
        style_3 = VGG_FeatureExtractor(10, self.device)(style)
        style_4 = VGG_FeatureExtractor(15, self.device)(style)
        fake_2 = VGG_FeatureExtractor(5, self.device)(fake)
        fake_3 = VGG_FeatureExtractor(10, self.device)(fake)
        fake_4 = VGG_FeatureExtractor(15, self.device)(fake)
        
        # Decentralization
        dim = [2, 3]
        style_2 = style_2 - style_2.mean(dim=dim, keepdim=True)
        fake_2 = fake_2 - fake_2.mean(dim=dim, keepdim=True)
        style_3 = style_3 - style_3.mean(dim=dim, keepdim=True)
        fake_3 = fake_3 - fake_3.mean(dim=dim, keepdim=True)
        style_4 = style_4 - style_4.mean(dim=dim, keepdim=True)
        fake_4 = fake_4 - fake_4.mean(dim=dim, keepdim=True)
        
        # Calculate loss
        loss4_4 = self.criterion(gram(style_4), gram(fake_4)) / style_4.size(1)
        loss3_3 = self.criterion(gram(style_3), gram(fake_3)) / style_3.size(1)
        loss2_2 = self.criterion(gram(style_2), gram(fake_2)) / style_2.size(1)
        
        # Apply weights
        return self.weight[0] * loss2_2, self.weight[1] * loss3_3, self.weight[2] * loss4_4

class region_smoothing_loss(nn.Module):
    def __init__(self, device='cpu', weight=0.5):
        super(region_smoothing_loss, self).__init__()
        self.device = device
        self.weight = weight
        self.criterion = VGG_LOSS(device)

    def forward(self, x, y):
        loss = self.criterion(x.to(self.device), y.to(self.device))
        return self.weight * loss
        
class color_loss(nn.Module):
    def __init__(self, device='cpu', weight=0.5):
        super(color_loss, self).__init__()
        self.device = device
        self.weight = weight
        self.criterion = L1Loss()

    def forward(self, photo, fake):
        # Normalize the photos to [0, 1] range
        photo = (photo + 1.0) / 2.0
        fake = (fake + 1.0) / 2.0

        # Convert RGB to LAB
        photo_lab = rgb_to_lab(photo)  
        fake_lab = rgb_to_lab(fake)

        # Calculate the loss
        loss = 2.0 * self.criterion(photo_lab[:, 0, :, :] / 100.0, fake_lab[:, 0, :, :] / 100.0) + \
               self.criterion(photo_lab[:, 1, :, :] + (128.0 / 255.0), fake_lab[:, 1, :, :] + (128.0 / 255.0)) + \
               self.criterion(photo_lab[:, 2, :, :] + (128.0 / 255.0), fake_lab[:, 2, :, :] + (128.0 / 255.0))
        
        # Apply the weight
        return self.weight * loss

class total_variation_loss(nn.Module):
    def __init__(self, weight=0.5):
        super(total_variation_loss, self).__init__()
        self.weight = weight
        self.criterion = MSELoss(reduction='mean')  
        
    def forward(self, x):
        # Calculate the total variation loss
        dh = x[:, :, :-1, :] - x[:, :, 1:, :]
        dw = x[:, :, :, :-1] - x[:, :, :, 1:]
        
        # The size variables are used to normalize the loss terms
        size_dh = dh.numel()
        size_dw = dw.numel()
        
        dh_loss = self.criterion(dh, torch.zeros_like(dh))
        dw_loss = self.criterion(dw, torch.zeros_like(dw))

        loss = (dh_loss / size_dh) + (dw_loss / size_dw)

        return self.weight * loss