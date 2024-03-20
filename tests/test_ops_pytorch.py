import unittest
import torch
import torch.nn as nn
from ops_pytorch import CustomConv2d, LADE, ConvLADELReLU, ExternalAttentionV3

class TestCustomConv2d(unittest.TestCase):
    def test_forward_shape(self):
        batch_size, in_channels, out_channels, height, width = 2, 3, 5, 64, 64
        x = torch.rand(batch_size, in_channels, height, width)
        model = CustomConv2d(in_channels, out_channels)
        out = model(x)
        self.assertEqual(out.shape, (batch_size, out_channels, height, width))

class TestLADE(unittest.TestCase):
    def test_forward_shape(self):
        batch_size, channels, height, width = 2, 3, 64, 64
        x = torch.rand(batch_size, channels, height, width)
        model = LADE(channels)
        out = model(x)
        self.assertEqual(out.shape, x.shape)

class TestConvLADELReLU(unittest.TestCase):
    def test_forward_shape(self):
        batch_size, in_channels, out_channels, height, width = 2, 3, 5, 64, 64
        x = torch.rand(batch_size, in_channels, height, width)
        model = ConvLADELReLU(in_channels, out_channels)
        out = model(x)
        self.assertEqual(out.shape, (batch_size, out_channels, height, width))

class TestExternalAttentionV3(unittest.TestCase):
    def test_forward_shape(self):
        batch_size, channel, height, width = 2, 3, 64, 64
        x = torch.rand(batch_size, channel, height, width)
        model = ExternalAttentionV3(channel)
        out = model(x)
        self.assertEqual(out.shape, x.shape)

if __name__ == '__main__':
    unittest.main()
