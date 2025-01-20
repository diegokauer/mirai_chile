import unittest

import torch
from mirai_chile.models.loss.pmf_loss import PMFLoss
from mirai_chile.models.pmf_layer import PMFLayer


class PMFLossTestCase(unittest.TestCase):
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    loss = PMFLoss().to_device(device)
    pmf_layer = PMFLayer(10).to_device(device)

    logit = torch.tensor([[-0.810, -0.724, -0.724, -0.724, -0.724]]).to(device)
    pmf = torch.tensor([[0.1, 0.2, 0.1, 0.3, 0.05]]).to(device)
    s = torch.tensor([[0.9, 0.7, 0.6, 0.3, 0.25]]).to(device)

    def test_pmf_only_0_single(self):
        logit = torch.zeros((1, self.loss.args.max_followup), device=self.device)
        pmf = torch.zeros((1, self.loss.args.max_followup), device=self.device)
        s = torch.zeros((1, self.loss.args.max_followup), device=self.device)

        for d_i in [0, 1]:
            for t_i in range(self.loss.args.max_followup):
                t = torch.tensor([[t_i]], device=self.device)
                d = torch.tensor([[d_i]], device=self.device)

                loss = self.loss(logit, pmf, s, t, d).cpu()

                # self.assertEqual(-torch.log(torch.tensor(1e-9)), loss)

    def test_pmf_only_0_batched(self):
        batch_size = 10

        logit = torch.zeros((batch_size, self.loss.args.max_followup), device=self.device)
        pmf = torch.zeros((batch_size, self.loss.args.max_followup), device=self.device)
        s = torch.zeros((batch_size, self.loss.args.max_followup), device=self.device)

        for d_i in [0, 1]:
            for t_i in range(self.loss.args.max_followup):
                t = torch.tensor([[t_i]], device=self.device).repeat(batch_size, 1)
                d = torch.tensor([[d_i]], device=self.device).repeat(batch_size, 1)
                loss = self.loss(logit, pmf, s, t, d).cpu()

                # self.assertEqual(-torch.log(torch.tensor(1e-9)), loss)

    def test_pmf_only_1_single(self):
        logit = torch.ones((1, self.loss.args.max_followup), device=self.device)
        pmf = torch.ones((1, self.loss.args.max_followup), device=self.device)
        s = torch.ones((1, self.loss.args.max_followup), device=self.device)

        for d_i in [0, 1]:
            for t_i in range(self.loss.args.max_followup):
                t = torch.tensor([[t_i]], device=self.device)
                d = torch.tensor([[d_i]], device=self.device)

                loss = self.loss(logit, pmf, s, t, d).cpu()

                # self.assertEqual(-torch.log(torch.tensor(1)), loss)

    def test_pmf_only_0_batched(self):
        batch_size = 10

        logit = torch.ones((batch_size, self.loss.args.max_followup), device=self.device)
        pmf = torch.ones((batch_size, self.loss.args.max_followup), device=self.device)
        s = torch.ones((batch_size, self.loss.args.max_followup), device=self.device)

        for d_i in [0, 1]:
            for t_i in range(self.loss.args.max_followup):
                t = torch.tensor([[t_i]], device=self.device).repeat(batch_size, 1)
                d = torch.tensor([[d_i]], device=self.device).repeat(batch_size, 1)
                loss = self.loss(logit, pmf, s, t, d).cpu()

                # self.assertEqual(-torch.log(torch.tensor(1)), loss)

    def test_pmf_synth_example(self):
        for d_i in [0, 1]:
            for t_i in range(self.loss.args.max_followup):
                t = torch.tensor([[t_i]], device=self.device)
                d = torch.tensor([[d_i]], device=self.device)
                loss = self.loss(self.logit, self.pmf, self.s, t, d).cpu()

                # if d_i == 1:
                #     self.assertEqual(-torch.log(self.pmf[0, t_i]).cpu(), loss)
                # else:
                #     self.assertEqual(-torch.log(self.s[0, t_i]).cpu(), loss)

    def test_pmf_backwards_pass_synth_example(self):
        for d_i in [0, 1]:
            for t_i in range(self.loss.args.max_followup):
                logit = torch.ones((1, 5)).to(self.device).requires_grad_()
                pmf, s = self.pmf_layer.logit_to_cancer_prob(logit)

                t = torch.tensor([[t_i]], device=self.device)
                d = torch.tensor([[d_i]], device=self.device)
                loss = self.loss(logit, pmf, s, t, d)

                loss.backward()
            self.assertEqual(torch.Size((1, 5)), logit.grad.shape)

    def test_pmf_excedes_max_followup_d_1(self):
        logit = torch.ones((2, 5), device=self.device)
        pmf = torch.zeros((2, 5), device=self.device)
        s = torch.ones((2, 5), device=self.device)
        t = torch.tensor([[10], [5]], device=self.device)
        d = torch.tensor([[1], [1]], device=self.device)
        loss = self.loss(logit, pmf, s, t, d).cpu()

        # self.assertEqual(-torch.log(torch.tensor(1)), loss)

    def test_pmf_excedes_max_followup_d_1(self):
        logit = torch.ones((2, 5), device=self.device)
        pmf = torch.zeros((2, 5), device=self.device)
        s = torch.ones((2, 5), device=self.device)
        t = torch.tensor([[10], [5]], device=self.device)
        d = torch.tensor([[0], [0]], device=self.device)
        loss = self.loss(logit, pmf, s, t, d).cpu()

        # self.assertEqual(-torch.log(torch.tensor(1)), loss)


if __name__ == '__main__':
    unittest.main()
