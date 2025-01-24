import unittest

import torch

from mirai_chile.models.cumulative_probability_layer import CumulativeProbabilityLayer


class CPLayerTestCase(unittest.TestCase):
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    cpl_layer = CumulativeProbabilityLayer(612)
    cpl_layer.to_device(device)

    def test_pmf_layer_forward_batched(self):
        input = torch.zeros((10, 612), device=self.device)
        logit = self.cpl_layer(input)

        self.assertEqual(torch.Size((10, 5)), logit.shape)

    def test_pmf_layer_forward_single(self):
        input = torch.zeros((1, 612), device=self.device)
        logit = self.cpl_layer(input)

        self.assertEqual(torch.Size((1, 5)), logit.shape)


if __name__ == '__main__':
    unittest.main()
