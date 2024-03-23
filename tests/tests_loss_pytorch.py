import unittest
import torch
from torchvision.transforms import functional as F
from torch.testing import assert_allclose
from ops_pytorch import VGG_FeatureExtractor, VGG_LOSS, con_loss


class TestVGGFunctions(unittest.TestCase):
    def setUp(self):
        # Setup a dummy image (batch size 1, 3 channels, 224x224 pixels)
        self.dummy_image = torch.rand(1, 3, 224, 224)
        self.device ='cpu'
        self.dummy_image = self.dummy_image.to(self.device)

    def test_feature_extractor_shape(self):
        """Test if the VGG_FeatureExtractor returns features of the correct shape."""
        extractor = VGG_FeatureExtractor(24, self.device)
        features = extractor(self.dummy_image)
        self.assertTrue(features.shape[1] == 512)  # The number of channels for conv4_4 should be 512

    def test_vgg_loss_value(self):
        """Test if the VGG_LOSS returns a non-negative scalar."""
        vgg_loss = VGG_LOSS(self.device)
        loss_value = vgg_loss(self.dummy_image, self.dummy_image)
        self.assertTrue(loss_value.item() >= 0)

    def test_con_loss_value_and_weight(self):
        """Test if the con_loss returns a non-negative scalar and respects the weight."""
        weight = 0.5
        content_loss = con_loss(self.device, weight=weight)
        loss_value = content_loss(self.dummy_image, self.dummy_image)
        self.assertTrue(loss_value.item() >= 0)

        # Check if weight is applied correctly
        content_loss_no_weight = con_loss(self.device, weight=1.0)
        loss_value_no_weight = content_loss_no_weight(self.dummy_image, self.dummy_image)
        assert_allclose(loss_value, loss_value_no_weight * weight, atol=1e-6, rtol=1e-6)

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
