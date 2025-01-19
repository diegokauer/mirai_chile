import unittest

import numpy as np
import torch
from mirai_chile.models.pmf_layer import PMFLayer


class PMFLayerTestCase(unittest.TestCase):
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    pmf_layer = PMFLayer(612)
    pmf_layer.to_device(device)

    def test_pmf_layer_forward(self):
        input = torch.zeros((10, 612), device=self.device)
        logit = self.pmf_layer(input)

        self.assertEqual(torch.Size((10, 5)), logit.shape)

    def test_pmf_logit_to_probs_zeros(self):
        logit = torch.zeros((1, 5), device=self.device)
        pmf, s = self.pmf_layer.logit_to_cancer_prob(logit)

        expected_pmf = torch.tensor([[np.exp(0) / (np.exp(0) * 5 + 1)] * 5], dtype=torch.float32)
        expected_s = torch.tensor([[1 - np.exp(0) / (np.exp(0) * 5 + 1) * (i + 1) for i in range(5)]],
                                  dtype=torch.float32)

        self.assertTrue(torch.isclose(expected_pmf, pmf.cpu()).all())
        self.assertTrue(torch.isclose(expected_s, s.cpu()).all())

    def test_pmf_logit_to_probs_ones(self):
        logit = torch.ones((1, 5), device=self.device)
        pmf, s = self.pmf_layer.logit_to_cancer_prob(logit)

        expected_pmf = torch.tensor([[np.exp(1) / (np.exp(1) * 5 + 1)] * 5], dtype=torch.float32)
        expected_s = torch.tensor([[1 - np.exp(1) / (np.exp(1) * 5 + 1) * (i + 1) for i in range(5)]],
                                  dtype=torch.float32)

        self.assertTrue(torch.isclose(expected_pmf, pmf.cpu()).all())
        self.assertTrue(torch.isclose(expected_s, s.cpu()).all())


if __name__ == '__main__':
    unittest.main()
