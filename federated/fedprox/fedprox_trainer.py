from ultralytics.models.yolo.detect.train import DetectionTrainer
import torch



class FedProxTrainer(DetectionTrainer):


    def __init__(
        self,
        cfg=None,
        overrides=None,
        _callbacks=None
    ):


        if overrides is None:
            overrides = {}


        self.global_weights = overrides.pop(
            "global_weights",
            None
        )


        self.mu = overrides.pop(
            "mu",
            0.01
        )


        super().__init__(
            cfg,
            overrides,
            _callbacks
        )



    def compute_loss(
        self,
        model,
        batch
    ):


        loss, loss_items = super().compute_loss(
            model,
            batch
        )


        if self.global_weights is None:

            return loss, loss_items



        prox_loss = torch.zeros(
            1,
            device=self.device
        )


        for name, param in model.named_parameters():


            if name in self.global_weights:


                global_param = (
                    self.global_weights[name]
                    .to(self.device)
                )


                prox_loss += torch.sum(
                    (param - global_param) ** 2
                )



        loss = (

            loss

            +

            (self.mu / 2)

            *

            prox_loss

        )


        return loss, loss_items