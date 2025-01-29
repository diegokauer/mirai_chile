from mirai_chile.configs.abstract_config import AbstractConfig


class MiraiChileConfig(AbstractConfig):
    pmf = True
    model_path = 'checkpoints/good_models/mirai_transformer_pmf_final.pt'
