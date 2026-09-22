from ultralytics.models.yolo.detect import DetectionTrainer

import torch
import torch.nn as nn
import torch.nn.functional as F

from cl_loss import SupConLoss



class ProjectionHead(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                input_dim,
                256
            ),

            nn.ReLU(),

            nn.Linear(
                256,
                128
            )
        )


    def forward(self, x):

        return F.normalize(
            self.net(x),
            dim=1
        )



class YOLOCLTrainer(DetectionTrainer):


    def setup_model(self):

        super().setup_model()

        self.features = None

        self.supcon = SupConLoss().to(
            self.device
        )

        self.projection = None

        print("Model loaded")



    def _setup_train(self):

        super()._setup_train()


        print("Attaching feature hook...")


        self.model.model[-2].register_forward_hook(
            self.feature_hook
        )


        print("Feature hook attached")



    def feature_hook(
        self,
        module,
        input,
        output
    ):

        self.features = output



    def get_embedding(self, feature):


        if len(feature.shape) == 4:

            feature = torch.mean(
                feature,
                dim=[2,3]
            )


        if self.projection is None:

            self.projection = ProjectionHead(
                feature.shape[1]
            ).to(
                self.device
            )


            print(
                "Projection dimension:",
                feature.shape[1]
            )


        return self.projection(
            feature
        )



    def criterion(
        self,
        preds,
        batch
    ):


        yolo_loss = super().criterion(
            preds,
            batch
        )


        contrastive_loss = torch.tensor(
            0.0,
            device=self.device
        )


        if self.features is not None:


            embedding = self.get_embedding(
                self.features
            )


            labels = batch["cls"]


            labels = labels.reshape(
                -1
            )


            labels = labels[:embedding.shape[0]]


            contrastive_loss = self.supcon(
                embedding,
                labels
            )


        total_loss = (

            yolo_loss[0]

            +

            0.1 * contrastive_loss

        )


        return (

            total_loss,

            yolo_loss[1]

        )