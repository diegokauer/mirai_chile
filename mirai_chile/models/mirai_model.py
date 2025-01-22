import os

import torch
from mirai_chile.configs.abstract_config import AbstractConfig
from mirai_chile.models.abstract_layer import AbstractLayer
from mirai_chile.models.loss.abstract_loss import AbstractLoss
from torch import nn


class MiraiChile(nn.Module):
    def __init__(
            self,
            args=AbstractConfig(),
            head=AbstractLayer(),
            encoder=None,
            transformer=None,
            loss_function=AbstractLoss()
    ):
        super().__init__()
        self.args = args
        self._encoder = encoder
        if encoder is None:
            self.load_encoder(args.encoder_path)
        self.head = head
        if transformer is None:
            self.load_transformer(args.transformer_path)
        self.loss_function = loss_function

        # Freeze backbone of model
        for param in self._encoder.parameters():
            param.requires_grad = not args.freeze_encoder
        for param in self._transformer.parameters():
            param.requires_grad = not args.freeze_transformer
        # for param in self._transformer.pool.parameters():
        #     param.requires_grad = not args.freeze_risk_factor_layer
        for param in self._encoder.pool.parameters():
            param.requires_grad = not args.freeze_risk_factor_layer

    def forward(self, x, batch=None):
        if hasattr(self.args, "use_precomputed_encoder_hidden") and self.args.use_precomputed_encoder_hidden:
            B = x.size(0)
            encoder_hidden = x
            encoder_hidden = encoder_hidden.view(B, 4, -1)
        elif not self.args.precompute_mode:
            B, C, N, H, W = x.size()
            x = x.transpose(1, 2).contiguous().view(B * N, C, H, W)
            x = self.encoder_forward(x)
            encoder_hidden = self.aggregate_and_classify_encoder(x)

        if hasattr(self.args, "use_precomputed_transformer_hidden") and self.args.use_precomputed_transformer_hidden:
            B = x.size(1)
            encoder_hidden = torch.zeros((B, 4, 1))
            transformer_hidden = x  # [:, :512]
        else:  # elif not self.args.precompute_mode:
            encoder_hidden = encoder_hidden.view(B, 4, -1)
            transformer_hidden = self.transformer_forward(encoder_hidden, batch)
            transformer_hidden = transformer_hidden[:, :512]
            transformer_hidden = self.aggregate_and_classify_transformer(transformer_hidden)

        if hasattr(self.args, "use_original_aggregate") and self.args.use_original_aggregate:
            logit, transformer_hidden = self._transformer.aggregate_and_classify(transformer_hidden)
        else:
            logit = self.head(transformer_hidden)

        return logit, transformer_hidden, encoder_hidden.view(B, -1)

    def encoder_forward(self, x):
        for gpu, layers in enumerate(self._encoder.gpu_to_layer_assignments):
            x = x.to(self.args.device)
            for name in layers:
                layer = self._encoder._modules[name]
                x = layer(x)

        return x

    def aggregate_and_classify_encoder(self, x):
        _, hidden = self._encoder.pool.internal_pool(x)
        if not self._encoder.pool.replaces_fc():
            # self.fc is always on last gpu, so direct call of fc(x) is safe
            try:
                # placed in try catch for back compatbility.
                hidden = self._encoder.relu(hidden)
            except:
                pass
            hidden = self._encoder.dropout(hidden)

        return hidden

    def transformer_forward(self, x, batch):
        time_seq, view_seq, side_seq = batch['time_seq'], batch['view_seq'], batch['side_seq']

        masked_x, is_mask = self._transformer.mask_input(x, view_seq.to(self.args.device))
        masked_x = self._transformer.projection_layer(masked_x)
        transformer_hidden = self._transformer.transformer(masked_x,
                                                           time_seq.to(self.args.device),
                                                           view_seq.to(self.args.device),
                                                           side_seq.to(self.args.device))

        img_like_hidden = transformer_hidden.transpose(1, 2).unsqueeze(-1)

        return img_like_hidden

    def aggregate_and_classify_transformer(self, x):
        _, hidden = self._transformer.pool(x, risk_factors=None)
        if not self._transformer.pool.replaces_fc():
            # self.fc is always on last gpu, so direct call of fc(x) is safe
            try:
                # placed in try catch for back compatbility.
                hidden = self._transformer.relu(hidden)
            except:
                pass
            hidden = self._transformer.dropout(hidden)

        return hidden

    def load_encoder(self, path):
        model_path = os.path.expanduser(path)
        self._encoder = torch.load(model_path, map_location='cpu').module._model
        self._encoder.args.use_pred_risk_factors_at_test = True

    def load_transformer(self, path):
        model_path = os.path.expanduser(path)
        self._transformer = torch.load(model_path, map_location='cpu')

    def to_device(self, device):
        self.args.device = device
        self.head.to_device(device)
        self.loss_function.to_device(device)
        self.to(device)
        self._transformer.to(device)
        self._encoder.to(device)
        return self
