from mirai_chile.configs.abstract_config import AbstractConfig


class MiraiBaseConfig(AbstractConfig):
    make_probs_indep = False
    use_original_aggregate = True


class MiraiBaseConfigEval(AbstractConfig):
    make_probs_indep = False
    use_original_aggregate = True
    freeze_encoder = True
    freeze_transformer = True
    freeze_risk_factor_layer = True
