from mirai_chile.configs.abstract_config import AbstractConfig


class TrainTransformerHiddenConfig(AbstractConfig):
    use_precomputed_transformer_hidden = True
    precompute_mode = True
    freeze_encoder = True
    freeze_transformer = True
    freeze_risk_factor_layer = True


class TrainEncoderHiddenConfig(AbstractConfig):
    use_precomputed_encoder_hidden = True
    precompute_mode = True
    freeze_encoder = True
    freeze_transformer = False
    freeze_risk_factor_layer = True


class TrainTransformerCPFHiddenConfig(AbstractConfig):
    use_precomputed_transformer_hidden = True
    use_original_aggregate = True
    precompute_mode = True
    freeze_encoder = True
    freeze_transformer = False
    freeze_risk_factor_layer = True
