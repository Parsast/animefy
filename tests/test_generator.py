import unittest
import torch
from generator_pytorch import G_net

class TestGNet(unittest.TestCase):
    def setUp(self):
        self.input_channels = 3  # Assuming RGB images
        self.batch_size = 4
        self.height = 256
        self.width = 256
        self.model = G_net(inputs_channels=self.input_channels)

    def test_initialization(self):
        """Test if the model initializes correctly."""
        self.assertIsInstance(self.model, G_net)

    def test_forward_pass(self):
        """Test the forward pass of the model."""
        input_tensor = torch.randn(self.batch_size, self.input_channels, self.height, self.width)
        output = self.model(input_tensor)
        # Assuming the output is a tuple of two tensors (fake_s, fake_m)
        self.assertIsInstance(output, tuple)
        self.assertEqual(len(output), 2)

        # Check if the output tensors have the expected shape
        for out in output:
            self.assertEqual(out.size(0), self.batch_size)
            self.assertEqual(out.size(1), 3)  # Assuming output channels (e.g., RGB)
            self.assertEqual(out.size(2), self.height)  # Assuming output height matches input
            self.assertEqual(out.size(3), self.width)  # Assuming output width matches input

    def test_training_mode(self):
        """Test if the model can switch between training and evaluation modes."""
        self.assertTrue(self.model.training, "Model should be in training mode by default.")
        self.model.eval()
        self.assertFalse(self.model.training, "Model should be in evaluation mode after calling .eval().")
        self.model.train()
        self.assertTrue(self.model.training, "Model should be back in training mode after calling .train().")

if __name__ == '__main__':
    unittest.main()
