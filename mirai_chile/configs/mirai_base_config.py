from mirai_chile.configs.abstract_config import AbstractConfig


class MiraiBaseConfig(AbstractConfig):
    make_probs_indep = False
    use_original_aggregate = False


class MiraiBaseConfigEval(AbstractConfig):
    make_probs_indep = False
    use_original_aggregate = True
    freeze_encoder = True
    freeze_transformer = True
    freeze_risk_factor_layer = True
    use_precomputed_transformer_hidden = False
    precompute_mode = False
    use_calibrator = True
    train = False


class TrainFromTransformerHidden(AbstractConfig):
    make_probs_indep = False
    use_original_aggregate = True
    freeze_encoder = True
    freeze_transformer = True
    freeze_risk_factor_layer = True
    freeze_additive_hazard_layer = False
    precompute_mode = True
    use_precomputed_transformer_hidden = True
    use_calibrator = False


class TrainFromEncoderHidden(AbstractConfig):
    make_probs_indep = False
    use_original_aggregate = True
    freeze_encoder = True
    freeze_transformer = False
    freeze_risk_factor_layer = True
    freeze_additive_hazard_layer = False
    precompute_mode = True
    use_precomputed_encoder_hidden = True
    use_calibrator = False


class TrainEnd2End(AbstractConfig):
    make_probs_indep = False
    use_original_aggregate = True
    freeze_encoder = False
    freeze_transformer = False
    freeze_additive_hazard_layer = False
    freeze_risk_factor_layer = True
    use_calibrator = False
