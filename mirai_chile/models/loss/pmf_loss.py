import torch

from mirai_chile.models.loss.abstract_loss import AbstractLoss


class PMFLoss(AbstractLoss):
    def __init__(self, args=None):
        super().__init__()
        if not (args is None):
            self.args = args
        # Difference matrix for Mirai Backbone
        dif_matrix = torch.zeros((self.args.max_followup, self.args.max_followup), dtype=torch.float32)
        dif_matrix[0, 0] = 1
        for i in range(1, self.args.max_followup):
            dif_matrix[i, i - 1] = -1
            dif_matrix[i, i] = 1
        self.register_buffer('dif_matrix', dif_matrix)

        # Lower triangular matrix without diagonal
        l_t_mat = torch.tril(torch.ones([self.args.max_followup, self.args.max_followup], dtype=torch.float32),
                             diagonal=-1)
        self.register_buffer('lower_triangular_matrix_no_diag', l_t_mat)

        # Ones vector for cumulative probability calculation
        self.register_buffer('ones', torch.ones(self.args.max_followup, dtype=torch.float32))

    def forward(self, logit, pmf, s, t, d):
        # """
        # logit: Tensor of shape (B, T)
        # t: Indices for the time step, shape (B,)
        # d: Indicators for event occurrence, shape (B,)
        # """
        device = self.args.device

        # d[(t >= self.args.max_followup) & (d == 1)] = 0
        t = torch.clamp(t, max=self.args.max_followup)  # Make t in range[0, 4]

        # Ensure t is used correctly
        batch_indices = torch.arange(logit.size(0), device=logit.device)
        pmf_t = pmf[batch_indices, t]
        s_t = s[batch_indices, t]

        # Numerical stability in log computation
        loss = d * torch.log(torch.clamp(pmf_t, min=1e-9)) + (1 - d) * torch.log(torch.clamp(s_t, min=1e-9))
        return -torch.mean(loss)  # Negative log likelihood

        # d[(t >= self.args.max_followup) & (d == 1)] = 0
        # t = torch.clamp(t, max=self.args.max_followup)  # Make t in range[0, 4]

        # events = d
        # idx_durations = t
        # phi = logit
        #
        # events = events.view(-1)
        # idx_durations = idx_durations.view(-1, 1)
        # phi = torch.cat([phi, torch.zeros((phi.size(0), 1), device=phi.device)], dim=1)
        # gamma = phi.max(1)[0]
        # cumsum = phi.sub(gamma.view(-1, 1)).exp().cumsum(1)
        # sum_ = cumsum[:, -1]
        # part1 = phi.gather(1, idx_durations).view(-1).sub(gamma).mul(events)
        # part2 = - sum_.relu().add(1e-10).log()
        # part3 = sum_.sub(cumsum.gather(1, idx_durations).view(-1)).relu().add(1e-10).log().mul(1. - events)
        # # need relu() in part3 (and possibly part2) because cumsum on gpu has some bugs and we risk getting negative numbers.
        # loss = - part1.add(part2).add(part3)
        # return loss.mean()
