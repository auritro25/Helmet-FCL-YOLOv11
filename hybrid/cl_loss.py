import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):

    def __init__(self, temperature=0.07):
        super().__init__()

        self.temperature = temperature


    def forward(self, features, labels):

        features = F.normalize(
            features,
            dim=1
        )


        similarity = torch.matmul(
            features,
            features.T
        ) / self.temperature


        labels = labels.reshape(-1,1)

        mask = torch.eq(
            labels,
            labels.T
        ).float().to(features.device)


        mask.fill_diagonal_(0)


        valid = mask.sum(1)>0


        if valid.sum()==0:
            return torch.tensor(
                0.0,
                device=features.device,
                requires_grad=True
            )


        similarity = similarity[valid][:,valid]

        mask = mask[valid][:,valid]


        exp = torch.exp(similarity)

        log_prob = similarity - torch.log(
            exp.sum(
                dim=1,
                keepdim=True
            )
        )


        loss = (
            -(mask*log_prob).sum(1)
            /
            mask.sum(1)
        ).mean()


        return loss