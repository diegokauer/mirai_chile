import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mirai_chile.configs.abstract_config import AbstractConfig
from mirai_chile.models.abstract_layer import AbstractLayer


class CumulativeProbabilityLayer(AbstractLayer):
    def __init__(self, num_features, args=AbstractConfig(), calibrator=None):
        super().__init__()
        if not (args is None):
            self.args = args
        self.hazard_fc = nn.Linear(num_features, args.max_followup)
        self.base_hazard_fc = nn.Linear(num_features, 1)
        self.relu = nn.ReLU(inplace=True)
        mask = torch.ones([args.max_followup, args.max_followup])
        mask = torch.tril(mask, diagonal=0)
        mask = torch.nn.Parameter(torch.t(mask), requires_grad=False)
        self.register_parameter('upper_triagular_mask', mask)

        # logit_to_prob necessary components
        self.sigmoid = F.sigmoid
        if calibrator is None and hasattr(args, "use_calibrator") and args.use_calibrator:
            self.get_calibrator()
        dif_matrix = torch.zeros((args.max_followup, args.max_followup), dtype=torch.float32)
        dif_matrix[0, 0] = 1
        for i in range(1, args.max_followup):
            dif_matrix[i, i - 1] = -1
            dif_matrix[i, i] = 1
        self.register_buffer('dif_matrix', dif_matrix)

    def hazards(self, x):
        raw_hazard = self.hazard_fc(x)
        pos_hazard = self.relu(raw_hazard)
        return pos_hazard

    def forward(self, x):
        if self.args.make_probs_indep:
            return self.hazards(x)
        #        hazards = self.hazard_fc(x)
        hazards = self.hazards(x)
        B, T = hazards.size()  # hazards is (B, T)
        expanded_hazards = hazards.unsqueeze(-1).expand(B, T, T)  # expanded_hazards is (B,T, T)
        masked_hazards = expanded_hazards * self.upper_triagular_mask  # masked_hazards now (B,T, T)
        cum_prob = torch.sum(masked_hazards, dim=1) + self.base_hazard_fc(x)
        return cum_prob

    def logit_to_cancer_prob(self, logit):
        device = self.args.device

        probs = self.sigmoid(logit)
        pred_y = np.zeros(probs.shape[1])

        if hasattr(self.args, "use_calibrator") and self.args.use_calibrator:
            for i in self._calibrator.keys():
                pred_y[i] = self._calibrator[i].predict_proba(probs[0, i].reshape(-1, 1)).flatten()[1]

        s = 1 - pred_y
        pmf = torch.matmul(s_inv, self.dif_matrix.to(device))

        return s, pmf

    def get_calibrator(self):
        with open(self.args.calibrator_path, 'rb') as infi:
            calibrator = pickle.load(infi)
        self._calibrator = calibrator
