import torch.nn.functional as F
import torch.nn as nn
import torch


class conv_LADE_Lrelu(nn.Module):
    def __init__(self, inputs, filters, kernel_size=3, strides=1, name='conv', padding='VALID', Use_bias = None,  alpha=0.2):
        super(nn.Module, self).__init__()
        self.inputs = inputs
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.Use_bias = Use_bias
        self.alpha = alpha
        
        
    def Conv2d(self, padding='VALID', Use_bias = None, activation_fn=None):
        if (self.kernel_size - self.strides) % 2 == 0:
            pad = (self.kernel_size - self.strides) // 2
            pad_top, pad_bottom, pad_left, pad_right= pad, pad, pad, pad
        else:
            pad = (self.kernel_size - self.strides) // 2
            pad_bottom, pad_right =pad, pad,
            pad_top, pad_left = self.kernel_size - self.strides - pad_bottom,  self.kernel_size - self.strides - pad_right
        
        F.pad(self.inputs, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')
        return nn.Conv2d(self.inputs, self.filters, self.kernel_size, self.strides, self.padding, self.Use_bias)
    
    def LADE(self, x):
        eps = 1e-5
        ch = x.shape[-1]
        tx = self.conv2d(x, ch, 1, 1)
        t_mean = tx.mean(dim= (1,2),keep_dims=True)
        x_mean = x.mean(dim= (1,2),keep_dims=True)
        t_var = tx.var(dim= (1,2),keep_dims=True, unbiased=False)
        x_var = x.var(dim= (1,2),keep_dims=True, unbiased=False)
        x_in = (x - in_mean) / (torch.sqrt(x_var + eps))
        x = x_in * (torch.sqrt(t_sigma + eps)) + t_mean
        return x
    
    def conv_LADE_Lrelu(self):
        x = self.Conv2d()
        x = self.LADE(x)
        return F.leaky_relu(x, self.alpha)



class External_attention_v3(nn.Module):
    def __init__(self, x, is_training, k=128):
        super(nn.Module, self).__init__()
        sef.idn = x
        self.x = x
        self.is_training = is_training
        self.k = k
        self.b, self.h, self.w, self.c = x.shape
        self.w_kernel = nn.Conv2d(self.c, self.k, 1, 1, padding='VALID')
        nn.init.xavier_uniform_(self.w_kernel.weight)
        self.x = Conv2D(self.x, self.c, 1, 1,)
        self.x = self.x.view(self.b, -1, self.c)
        self.attn = F.conv1d(self.x, self.w_kernel.weight, stride=1, padding='VALID')
        
    def forward(self):
        self.attn = F.softmax(self.attn, dim=1)
        self.attn = self.attn / (1e-9 + self.attn.sum(dim=2, keepdim=True))

        self.w_kernel = nn.Conv2d(self.k,self.c, 1, 1, padding='VALID')
        nn.init.xavier_uniform_(self.w_kernel.weight)
        self.x = Conv2D(self.x, self.c, 1, 1)
        bn_layer = x.batch_norm_wrapper(x, self.is_training)
        self.x = bn_layer.forward(self.x, self.is_training)
        self.x = self.x + self.idn
        self.x = F.leaky_relu(self.x, 0.2)
        return self.x

    
    

