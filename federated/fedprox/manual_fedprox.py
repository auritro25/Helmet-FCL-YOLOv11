import torch

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.cfg import get_cfg



class FedProxTrainer(DetectionTrainer):


    def __init__(
        self,
        cfg=None,
        overrides=None,
        _callbacks=None,
        **kwargs
    ):


        if cfg is None:

            cfg = get_cfg()



        if overrides is None:

            overrides = {}



        super().__init__(

            cfg=cfg,

            overrides=overrides,

            _callbacks=_callbacks

        )



        self.global_weights = None

        self.mu = 0.01





    def compute_loss(

        self,

        model,

        batch,

        preds=None

    ):


        loss, loss_items = super().compute_loss(

            model,

            batch,

            preds

        )



        prox = torch.zeros(

            1,

            device=self.device

        )



        if self.global_weights is not None:



            for name,param in model.named_parameters():


                if name in self.global_weights:


                    global_param = (

                        self.global_weights[name]

                        .to(self.device)

                    )


                    prox += torch.sum(

                        (

                            param

                            -

                            global_param

                        )

                        **2

                    )



        fedprox_loss = (

            loss

            +

            (

                self.mu / 2

            )

            *

            prox

        )



        return fedprox_loss, loss_items