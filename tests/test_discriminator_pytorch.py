import unittest
import torch
from discriminator_pytorch import D_net



class TestDNet(unittest.TestCase):
    def test_forward(self):
        x = torch.randn(1, 3, 128, 128)  # Assuming input channels is 3
        ch = 64  # Initial channel size
        model = D_net(x, ch, sn=False)
        out = model(x)
        print(out.shape)
        # # Assuming the output is a single channel (e.g., for binary classification)
        # self.assertEqual(out.shape, (1, 1, 1, 1))  # Update expected shape based on your model's architecture

if __name__ == '__main__':
    unittest.main()
