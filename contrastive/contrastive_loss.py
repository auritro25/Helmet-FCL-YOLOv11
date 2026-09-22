import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):

    def __init__(self, temperature=0.07):
        super().__init__()

        self.temperature = temperature


    def forward(self, features, labels):

        device = features.device


        features = F.normalize(
            features,
            dim=1
        )


        similarity = torch.matmul(
            features,
            features.T
        ) / self.temperature


        labels = labels.contiguous().view(-1,1)


        mask = torch.eq(
            labels,
            labels.T
        ).float().to(device)


        # remove self comparison
        logits_mask = torch.ones_like(mask)

        logits_mask.fill_diagonal_(0)


        mask = mask * logits_mask


        # remove samples without positive pairs
        valid = mask.sum(dim=1) > 0


        if valid.sum() == 0:
            return torch.tensor(
                0.0,
                device=device,
                requires_grad=True
            )


        similarity = similarity[valid][:,valid]

        mask = mask[valid][:,valid]


        logits_mask = logits_mask[valid][:,valid]


        exp_logits = (
            torch.exp(similarity)
            *
            logits_mask
        )


        log_prob = similarity - torch.log(
            exp_logits.sum(
                dim=1,
                keepdim=True
            )
        )


        mean_log_prob = (
            mask * log_prob
        ).sum(1) / mask.sum(1)


        loss = -mean_log_prob.mean()


        return loss