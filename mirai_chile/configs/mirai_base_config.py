from mirai_chile.configs.generic_config import GenericConfig


class MiraiBaseConfig(GenericConfig):
    make_probs_indep = False
    use_original_aggregate = True


class MiraiBaseConfigEval(GenericConfig):
    make_probs_indep = False
    use_original_aggregate = True
    freeze_encoder = True
    freeze_transformer = True
    freeze_risk_factor_layer = True
