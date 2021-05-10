import torch
import torch.nn as nn
import logging
logger = logging.getLogger("default_log")
class MLP(nn.Module):
    """
    projection head for contrastive learning.
    """
    def __init__(self, dim_mlp, dim_nce, l2_norm=True):
        super(MLP, self).__init__()
        self.l2_norm = l2_norm
        self.mlp = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim_nce))

    def forward(self, x):
        if self.l2_norm:
            x = nn.functional.normalize(x, dim=1)
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=1)
        return x


class SupClusterConLoss(nn.Module):
    """
    modified from supConLoss:  https://arxiv.org/pdf/2004.11362.pdf
    """
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(SupClusterConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, centers, labels):
        device = features.device
        centers= torch.cat(centers, dim=0).reshape(5, -1).to(device)
        features = torch.cat([features, centers], dim=0)
        labels = torch.cat([labels, torch.Tensor([0,1,2,3,4]).long().to(device)], dim=0)

        batch_size = features.shape[0]
        if batch_size < 5:
            return 0


        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)  # N * N, mark positive samples

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)  # N * N
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )  # N * N, eyes are 0, others 1
        mask = mask * logits_mask  # N * N, mark positive samples except themselves

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        if torch.sum(exp_logits).isnan():
            print()
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        log_prob_np = log_prob.detach().cpu().numpy()

        # compute mean of log-likelihood over positive
        # assert sum(mask.sum(1) == 0) != 0  # if
        # a = (mask * log_prob).sum(1)
        # b = (mask.sum(1) + 0.01)
        # mean_log_prob_pos = a / b
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 0.00001)  # add 0.0001 to avoid div 0 problem
        assert not mean_log_prob_pos.sum().isnan()



        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(batch_size).mean()
        if torch.isnan(loss):
            print()
        # logger.info(f"B:{batch_size} nce loss {loss:.2f}")

        return loss















