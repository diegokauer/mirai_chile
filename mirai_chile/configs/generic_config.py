class GenericConfig:
    max_followup = 5
    model_parallel = False
    device = None
    encoder_path = "~/.mirai/snapshots/mgh_mammo_MIRAI_Base_May20_2019.p"
    transformer_path = "~/.mirai/snapshots/mgh_mammo_cancer_MIRAI_Transformer_Jan13_2020.p"
    freeze_encoder = True
    freeze_transformer = False
    freeze_risk_factor_layer = True
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
    test_image_transformers = [
        "scale_2d",
        "align_to_left"
    ]
    test_tensor_transformers = [
        "force_num_chan_2d",
        "normalize_2d"
    ]
    video = False
