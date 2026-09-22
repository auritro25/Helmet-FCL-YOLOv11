from ultralytics.models.yolo.detect import DetectionTrainer
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

        # save FedProx parameters
        self.global_weights = overrides.pop(
            "global_weights",
            None
        )

        self.mu = overrides.pop(
            "mu",
            0.01
        )


        # IMPORTANT FIX
        if cfg is None:
            cfg = {}


        super().__init__(
            cfg=cfg,
            overrides=overrides,
            _callbacks=_callbacks
        )


    def criterion(
        self,
        preds,
        batch
    ):

        # YOLO original loss
        loss, loss_items = super().criterion(
            preds,
            batch
        )


        prox_loss = torch.tensor(
            0.0,
            device=self.device
        )


        if self.global_weights is not None:

            for name, param in self.model.named_parameters():

                if name in self.global_weights:

                    global_param = (
                        self.global_weights[name]
                        .to(self.device)
                    )

                    prox_loss += torch.sum(
                        (param - global_param) ** 2
                    )


        total_loss = (
            loss
            +
            (self.mu / 2) * prox_loss
        )


        return (
            total_loss,
            loss_items
        )