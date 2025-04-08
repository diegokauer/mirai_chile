import argparse
import os
import pprint

import torch

from mirai_chile.configs.mirai_base_config import MiraiBaseConfigEval
from mirai_chile.data.pre_processing import pre_process_images
from mirai_chile.models.cumulative_probability_layer import CumulativeProbabilityLayer
from mirai_chile.models.mirai_model import MiraiChile


def predict_images(imgs, view_saliency, grad_dir):
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    model_args = MiraiBaseConfigEval()
    head = CumulativeProbabilityLayer(612, args=model_args)
    model = MiraiChile(args=model_args, head=head)
    model.to_device(device)
    # model.load_state_dict(
    #     torch.load(os.path.expanduser(os.path.join('~/.mirai_chile', model_args.model_path)), map_location='cpu'))
    model.eval()

    # model = torch.load(os.path.join(__module_path__, model_args.model_path))

    model.to_device(device)

    x, batch = pre_process_images(imgs, model_args, direct_call=True)

    if view_saliency:
        x.requires_grad_(True)

    logit, _, _ = model(x, batch)

    pmf, s = model.head.logit_to_cancer_prob(logit)
    s_inv = 1 - s.squeeze()
    s_inv = [i.item() for i in s_inv]

    if view_saliency:
        pmf.sum().backward()
        torch.save(x.grad.data, os.path.join(grad_dir, 'grad.pt'))

    results = {
        'predictions': {f'Year {i + 1}': str(round(s, 4)) + ' %' for i, s in enumerate(s_inv)}
    }

    pprint.pprint(results)

    return results


def predict_png_imgs(args):
    return predict_images(args.pngs, args.view_saliency, args.grad_dir)


def main():
    parser = argparse.ArgumentParser(description="PNG images to predict using Mirai")
    parser.add_argument('pngs', nargs="*", help="Path to DICOM files (from a single exam) to run inference on.")
    parser.add_argument('--view-saliency', default=False, action="store_true", )
    parser.add_argument('--grad-dir', default='')
    args = parser.parse_args()
    predict_png_imgs(args)


if __name__ == '__main__':
    main()
