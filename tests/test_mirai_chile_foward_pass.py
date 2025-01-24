import unittest

import torch

from mirai_chile.configs.mirai_base_config import MiraiBaseConfig, MiraiBaseConfigEval
from mirai_chile.configs.mirai_chile_config import MiraiChileConfig
from mirai_chile.data.generate_dataset import create_dataloader, PNGDataset
from mirai_chile.data.pre_processing import pre_process_images
from mirai_chile.models.cumulative_probability_layer import CumulativeProbabilityLayer
from mirai_chile.models.mirai_model import MiraiChile
from mirai_chile.models.pmf_layer import PMFLayer


class TestMiraiForwardPass(unittest.TestCase):
    pmf_args = MiraiChileConfig()
    cpl_args = MiraiBaseConfig()
    eval_args = MiraiBaseConfigEval()

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    mirai_chile_pmf = MiraiChile(pmf_args, PMFLayer(612)).eval()
    mirai_base_cpl = MiraiChile(cpl_args, CumulativeProbabilityLayer(612)).eval()
    mirai_eval = MiraiChile(eval_args, CumulativeProbabilityLayer(612)).eval()

    batch = {
        "side_seq": torch.tensor([0, 0, 0, 0]),
        "time_seq": torch.tensor([1, 0, 1, 0]),
        "view_seq": torch.tensor([1, 1, 0, 0])
    }
    single_input = torch.ones((1, 3, 4, 20, 16))
    batched_input = torch.ones((7, 3, 4, 20, 16))

    single_unprocessed_exam = [
        "../mirai_chile/data/examples/png/000001_1_CC_L.png",
        "../mirai_chile/data/examples/png/000001_1_CC_R.png",
        "../mirai_chile/data/examples/png/000001_1_MLO_L.png",
        "../mirai_chile/data/examples/png/000001_1_MLO_R.png"
    ]

    dataset = PNGDataset("../mirai_chile/data/examples/png", pmf_args)
    dataloader = create_dataloader(dataset)
    dataloader_input = next(iter(dataloader))

    mirai_chile_pmf.to_device(device)
    mirai_base_cpl.to_device(device)
    mirai_eval.to_device(device)

    single_input.to(device)
    batched_input.to(device)
    for key in batch:
        batch[key] = batch[key].to(device)

    def test_mirai_chile_pmf_single_observation(self):
        logit, transformer_hidden, encoder_hidden = self.mirai_chile_pmf(self.single_input, self.batch)
        self.assertEqual(torch.Size([1, 5]), logit.shape)
        self.assertEqual(torch.Size([1, 612]), transformer_hidden.shape)
        self.assertEqual(torch.Size([1, 512 * 4]), encoder_hidden.shape)

    def test_mirai_chile_pmf_batched_observation(self):
        logit, transformer_hidden, encoder_hidden = self.mirai_chile_pmf(self.batched_input, self.batch)
        self.assertEqual(torch.Size([7, 5]), logit.shape)
        self.assertEqual(torch.Size([7, 612]), transformer_hidden.shape)
        self.assertEqual(torch.Size([7, 512 * 4]), encoder_hidden.shape)

    def test_mirai_chile_cpl_single_observation(self):
        logit, transformer_hidden, encoder_hidden = self.mirai_base_cpl(self.single_input, self.batch)
        self.assertEqual(torch.Size([1, 5]), logit.shape)
        self.assertEqual(torch.Size([1, 612]), transformer_hidden.shape)
        self.assertEqual(torch.Size([1, 512 * 4]), encoder_hidden.shape)

    def test_mirai_chile_cpl_batched_observation(self):
        logit, transformer_hidden, encoder_hidden = self.mirai_base_cpl(self.batched_input, self.batch)
        self.assertEqual(torch.Size([7, 5]), logit.shape)
        self.assertEqual(torch.Size([7, 612]), transformer_hidden.shape)
        self.assertEqual(torch.Size([7, 512 * 4]), encoder_hidden.shape)

    def test_pre_processing_mirai_single_observation(self):
        x, batch = pre_process_images(self.single_unprocessed_exam, self.cpl_args, direct_call=True)
        logit, transformer_hidden, encoder_hidden = self.mirai_base_cpl(x, batch)
        self.assertEqual(torch.Size([1, 5]), logit.shape)
        self.assertEqual(torch.Size([1, 612]), transformer_hidden.shape)
        self.assertEqual(torch.Size([1, 512 * 4]), encoder_hidden.shape)

    def test_pre_processing_mirai_single_dataloader(self):
        input = self.dataloader_input
        logit, transformer_hidden, encoder_hidden = self.mirai_base_cpl(input["images"], input["batch"])
        self.assertEqual(torch.Size([1, 5]), logit.shape)
        self.assertEqual(torch.Size([1, 612]), transformer_hidden.shape)
        self.assertEqual(torch.Size([1, 512 * 4]), encoder_hidden.shape)

    def test_pre_processing_mirai_eval_single_dataloader(self):
        input = self.dataloader_input
        logit, transformer_hidden, encoder_hidden = self.mirai_base_cpl(input["images"], input["batch"])
        self.assertEqual(torch.Size([1, 5]), logit.shape)
        self.assertEqual(torch.Size([1, 612]), transformer_hidden.shape)
        self.assertEqual(torch.Size([1, 512 * 4]), encoder_hidden.shape)


if __name__ == '__main__':
    unittest.main()
