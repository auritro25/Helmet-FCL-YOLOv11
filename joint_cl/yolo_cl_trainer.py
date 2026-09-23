import torch
import torch.nn as nn

from ultralytics import YOLO

from .cl_loss import NTXentLoss



class YOLOCLTrainer:


    def __init__(
        self,
        model_path="yolo11s.pt",
        proj_dim=128,
        temperature=0.07
    ):


        self.yolo = YOLO(
            model_path
        )


        self.model = self.yolo.model


        self.features = None


        # Hook a backbone feature layer
        # YOLO11s layers: use layer before Detect head

        target_layer = self.model.model[-2]


        target_layer.register_forward_hook(
            self.save_features
        )


        self.projector = nn.Sequential(

            nn.Linear(
                512,
                256
            ),

            nn.ReLU(),

            nn.Linear(
                256,
                proj_dim
            )

        )


        self.criterion = NTXentLoss(
            temperature
        )



    def save_features(
        self,
        module,
        input,
        output
    ):

        self.features = output



    def forward_features(
        self,
        x
    ):


        self.features = None


        # Run complete YOLO forward
        # Hook captures feature map

        _ = self.model(
            x
        )


        features = self.features


        if isinstance(
            features,
            list
        ):

            features = features[-1]


        # Global average pooling

        features = torch.mean(
            features,
            dim=(2,3)
        )


        return features



    def training_step(
        self,
        batch
    ):


        view1, view2, _ = batch


        z1 = self.forward_features(
            view1
        )


        z2 = self.forward_features(
            view2
        )


        z1 = self.projector(
            z1
        )


        z2 = self.projector(
            z2
        )


        loss = self.criterion(
            z1,
            z2
        )


        return loss