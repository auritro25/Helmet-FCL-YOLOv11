import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics import YOLO


class ProjectionHead(nn.Module):

    def __init__(self, input_dim=512, output_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )


    def forward(self, x):

        x = self.net(x)

        return F.normalize(
            x,
            dim=1
        )


class YOLOContrastive(nn.Module):

    def __init__(self, weights="yolo11s.pt"):

        super().__init__()

        self.model = YOLO(weights).model

        self.features = None


        self.model.model[-2].register_forward_hook(
            self.hook
        )


        self.projector = ProjectionHead()


    def hook(self, module, input, output):

        self.features = output


    def forward(self, x):

        _ = self.model(x)

        feature = self.features


        if len(feature.shape) == 4:

            feature = torch.mean(
                feature,
                dim=[2,3]
            )


        embedding = self.projector(
            feature
        )


        return embedding