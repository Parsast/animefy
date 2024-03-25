import unittest
import torch
from torchvision.transforms import functional as F
from torch.testing import assert_allclose
from ops_pytorch import VGG_FeatureExtractor, VGG_LOSS, con_loss, style_loss_decentralization_3, color_loss


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

class TestStyleLossDecentralization3(unittest.TestCase):
    def setUp(self):
        # Setup a dummy image (batch size 1, 3 channels, 224x224 pixels)
        self.dummy_style = torch.rand(1, 3, 224, 224)
        self.dummy_fake = torch.rand(1, 3, 224, 224)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dummy_style = self.dummy_style.to(self.device)
        self.dummy_fake = self.dummy_fake.to(self.device)
        self.weights = [1.0, 1.0, 1.0]
        
        # Instantiate the style loss class
        self.style_loss_module = style_loss_decentralization_3(self.device, self.weights)

    def test_non_negative_losses(self):
        """Test if the style losses are non-negative."""
        losses = self.style_loss_module(self.dummy_style, self.dummy_fake)
        for loss in losses:
            self.assertTrue(loss.item() >= 0, "Loss is negative.")

    def test_weighted_losses(self):
        """Test if the weights are applied correctly to the losses."""
        weighted_losses = self.style_loss_module(self.dummy_style, self.dummy_fake)
        # Check each loss term
        for i, weight in enumerate(self.weights):
            unweighted_loss = weighted_losses[i] / weight
            self.assertTrue(unweighted_loss.item() >= 0, "Unweighted loss is negative.")
            
            # Test with different weights
            new_weights = [w * 0.5 for w in self.weights]
            self.style_loss_module.weight = new_weights
            new_weighted_losses = self.style_loss_module(self.dummy_style, self.dummy_fake)
            assert_allclose(new_weighted_losses[i], weighted_losses[i] * 0.5, atol=1e-6, rtol=1e-6)

    def test_expected_input_dimensions(self):
        """Test the style loss with expected input dimensions."""
        try:
            self.style_loss_module(self.dummy_style, self.dummy_fake)
        except Exception as e:
            self.fail(f"Execution failed for input with expected dimensions: {e}")

class TestColorLoss(unittest.TestCase):
    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.photo = torch.randn(2, 3, 256, 256, device=self.device)  # A batch of 2 photos
        self.fake = torch.randn(2, 3, 256, 256, device=self.device)  # A batch of 2 fake images

    def test_color_loss_value(self):
        """Test if the color_loss returns a non-negative value."""
        weight = 0.5
        criterion = color_loss(device=self.device, weight=weight)
        loss = criterion(self.photo, self.fake)
        self.assertTrue(loss.item() >= 0, "Loss should be non-negative.")

    def test_color_loss_weight(self):
        """Test if the color_loss applies the weight correctly."""
        weight = 0.5
        criterion = color_loss(device=self.device, weight=weight)
        loss_weighted = criterion(self.photo, self.fake)
        
        criterion_unweighted = color_loss(device=self.device, weight=1.0)
        loss_unweighted = criterion_unweighted(self.photo, self.fake)
        
        self.assertAlmostEqual(loss_weighted.item(), weight * loss_unweighted.item(),
                               msg="Weighted loss should be equal to unweighted loss times the weight.")

    def test_color_loss_gradient(self):
        """Test if the color_loss allows for gradient computation."""
        self.photo.requires_grad_()
        self.fake.requires_grad_()
        weight = 0.5
        criterion = color_loss(device=self.device, weight=weight)
        loss = criterion(self.photo, self.fake)
        loss.backward()
        
        self.assertIsNotNone(self.photo.grad, "Gradients should be computed for photo.")
        self.assertIsNotNone(self.fake.grad, "Gradients should be computed for fake.")


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
