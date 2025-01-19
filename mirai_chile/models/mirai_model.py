import os

import torch
from mirai_chile.configs.generic_config import GenericConfig
from mirai_chile.models.generic_layer import GenericLayer
from mirai_chile.models.loss.generic_loss import GenericLoss
from torch import nn


class MiraiChile(nn.Module):
    def __init__(
            self,
            args=GenericConfig(),
            head=GenericLayer(),
            encoder=None,
            transformer=None,
            loss_function=GenericLoss()
    ):
        super(MiraiChile, self).__init__()
        self.args = args
        self._encoder = encoder
        if encoder is None:
            self.load_encoder(args.encoder_path)
        self.head = head(612, args)
        if transformer is None:
            self.load_transformer(args.transformer_path)
        self.loss_function = loss_function

        # Freeze backbone of model
        for param in self._encoder.parameters():
            param.requires_grad = not args.freeze_encoder
        for param in self._transformer.parameters():
            param.requires_grad = not args.freeze_transformer
        for param in self._transformer.pool.parameters():
            param.requires_grad = not args.freeze_risk_factor_layer

    def forward(self, x, batch):

        if hasattr(self.args, "use_precomputed_encoder_hiddens") and self.use_precomputed_encoder_hiddens:
            encoder_hidden = x
        else:
            B, C, N, H, W = x.size()
            x = x.transpose(1, 2).contiguous().view(B * N, C, H, W)
            x = self.encoder_forward(x)
            encoder_hidden = self.aggregate_and_classify_encoder(x)

        if hasattr(self.args, "use_precomputed_trasnformer_hiddens") and self.use_precomputed_trasnformer_hiddens:
            transformer_hidden = x
        else:
            encoder_hidden = encoder_hidden.view(B, N, -1)
            transformer_hidden = self.transformer_forward(encoder_hidden, batch)

        if hasattr(self.args, "use_original_aggregate") and self.args.use_original_aggregate:
            logit, transformer_hidden = self._transformer.aggregate_and_classify(transformer_hidden)
        else:
            transformer_hidden = self.aggregate_and_classify_transformer(transformer_hidden)
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
        self._encoder = torch.load(model_path, map_location='cpu').module._model.to(self.args.device)
        self._encoder.args.use_pred_risk_factors_at_test = True

    def load_transformer(self, path):
        model_path = os.path.expanduser(path)
        self._transformer = torch.load(model_path, map_location='cpu').to(self.args.device)

    def to_device(self, device):
        self.args.device = device
        self.head.to_device(device)
        self.loss_function.to_device(device)
        self.to(device)
        return self
