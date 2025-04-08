from scipy.stats import entropy


class AbstractConfig:
    train = True
    dropout_rate = 0.3
    max_followup = 5
    manufacturer_count = 8
    manufacturer_entropy = entropy(manufacturer_count)
    num_adv_steps = 3
    model_parallel = False
    device = None
    precompute_mode = False
    encoder_path = "~/.mirai/snapshots/mgh_mammo_MIRAI_Base_May20_2019.p"
    transformer_path = "~/.mirai/snapshots/mgh_mammo_cancer_MIRAI_Transformer_Jan13_2020.p"
    calibrator_path = "~/.mirai/snapshots/calibrators/Mirai_calibrator_mar12_2022.p"
    freeze_encoder = True
    freeze_transformer = True
    freeze_risk_factor_layer = True
    freeze_additive_hazard_layer = True
    make_probs_indep = False
    img_mean = [
        7047.99
    ]
    img_std = [
        12005.5
    ]
    img_size = [
        1664,
        2048
    ]
    num_chan = 3
    image_transformers = [
        "scale_2d",
        "align_to_left",
        "rand_ver_flip",
        "rotate_range/min=-20/max=20"
    ]
    tensor_transformers = [
        "force_num_chan_2d",
        "normalize_2d"
    ]
    test_image_transformers = [
        "scale_2d",
        "align_to_left"
    ]
    test_tensor_transformers = [
        "force_num_chan_2d",
        "normalize_2d"
    ]
    video = False
