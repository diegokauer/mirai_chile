from mirai_chile.configs.generic_config import GenericConfig


class TrainTransformerHiddenConfig(GenericConfig):
    use_precomputed_transformer_hidden = True
    precompute_mode = True
    freeze_encoder = True
    freeze_transformer = True
    freeze_risk_factor_layer = True
