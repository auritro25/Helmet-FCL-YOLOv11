import torch
import torch.nn as nn
import torch.nn.functional as F



class NTXentLoss(nn.Module):

    def __init__(
        self,
        temperature=0.07
    ):

        super().__init__()

        self.temperature = temperature


    def forward(
        self,
        z1,
        z2
    ):


        batch_size = z1.size(0)


        z1 = F.normalize(
            z1,
            dim=1
        )

        z2 = F.normalize(
            z2,
            dim=1
        )


        representations = torch.cat(
            [
                z1,
                z2
            ],
            dim=0
        )


        similarity = torch.matmul(
            representations,
            representations.T
        )


        similarity /= self.temperature


        mask = torch.eye(
            2*batch_size,
            device=z1.device
        ).bool()


        similarity.masked_fill_(
            mask,
            -9e15
        )


        positives = torch.cat(
            [
                torch.arange(
                    batch_size,
                    2*batch_size
                ),

                torch.arange(
                    0,
                    batch_size
                )
            ]
        ).to(z1.device)


        loss = F.cross_entropy(
            similarity,
            positives
        )


        return loss