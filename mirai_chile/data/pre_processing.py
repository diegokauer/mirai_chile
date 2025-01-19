import os
import traceback

import mirai_chile.models.transformers.factory as transformer_factory
import numpy as np
import torch
from PIL import Image
from mirai_chile.models.transformers.basic import ComposeTrans
from mirai_chile.models.utils import parsing


def pre_process_images(png_list, args, direct_call=False):
    images = read_pngs(png_list)

    test_image_transformers = parsing.parse_transformers(args.test_image_transformers)
    test_tensor_transformers = parsing.parse_transformers(args.test_tensor_transformers)
    test_transformers = transformer_factory.get_transformers(test_image_transformers, test_tensor_transformers, args)

    transforms = ComposeTrans(test_transformers)

    x, batch = collate_batch(images, transforms)

    x.to(args.device)
    for key, val in batch.items():
        batch[key] = val.to(args.device)

    if direct_call:
        x = x.unsqueeze(0)
        for k in batch:
            batch[k] = batch[k].unsqueeze(0)
    return x, batch


def collate_batch(images, transforms):
    batch = {}
    batch['side_seq'] = torch.cat([torch.tensor(b['side_seq']).unsqueeze(0) for b in images], dim=0)
    batch['view_seq'] = torch.cat([torch.tensor(b['view_seq']).unsqueeze(0) for b in images], dim=0)
    batch['time_seq'] = torch.zeros_like(batch['view_seq'])

    x = torch.cat(
        (lambda imgs: [transforms(b['x']).unsqueeze(0) for b in imgs])(images), dim=0
    ).transpose(0, 1)

    return x, batch


def read_pngs(png_list):
    files = list(map(os.path.expanduser, png_list))
    assert len(png_list) == 4, "Expected 4 files, got {}".format(len(png_list))
    for file in files:
        # assert dicom_file.endswith('.dcm'), f"DICOM files must have extension 'dcm'"
        assert os.path.exists(file), f"File not found: {file}"

    file_info = {}
    images = []
    for png in files:
        view_str, side_str = png.replace('.png', '').split('_')[-2:]

        view = 0 if view_str == 'CC' else 1
        side = 0 if side_str == 'R' else 1

        file_info[(view, side)] = png

    for k in file_info:
        try:
            png = file_info[k]
            view, side = k
            image = png_to_arr(png)
            images.append({'x': image, 'side_seq': side, 'view_seq': view})
        except Exception as e:
            print(f"{type(e).__name__}: {e}")
            print(f"{traceback.format_exc()}")
    return images


def png_to_arr(png):
    image = Image.open(png)
    image = np.array(image).astype(np.int32)
    return Image.fromarray(image, mode='I')
